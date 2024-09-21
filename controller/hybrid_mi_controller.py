import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import entropy

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.random_grid_map import RandomGridMap

from controller.base_controller import BaseController
from controller.adaptive_mpc_controller import AdaptiveMPCController
from controller.pure_pursuit_controller import PurePursuitController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

# Mutual Information을 계산하는 함수
def mutual_information(state1, state2):
    """두 상태 간의 Mutual Information 계산"""
    mi_list = []
    
    # x, y, theta, velocity 각각에 대해 Mutual Information 계산
    for i in range(4):  # [x, y, theta, velocity]
        hist1, _ = np.histogram(state1[:, i], bins=5, density=True)
        hist2, _ = np.histogram(state2[:, i], bins=5, density=True)
        print(f"State {i} Histograms: {hist1}, {hist2}")
        
        entropy1 = entropy(hist1)
        entropy2 = entropy(hist2)
        print(f"State {i} Entropy: {entropy1}, {entropy2}")
        
        joint_hist, _, _ = np.histogram2d(state1[:, i], state2[:, i], bins=5, density=True)
        joint_entropy = entropy(joint_hist.flatten())
        print(f"Joint Entropy: {joint_entropy}")
        
        # Mutual Information 계산
        mi = entropy1 + entropy2 - joint_entropy
        mi_list.append(mi)
    
    # 각 상태에 대한 Mutual Information 반환
    return np.array(mi_list)

# 두 알고리즘의 상태를 결합하는 함수
def combine_states(state1, state2, mi):
    """Mutual Information 기반으로 상태를 결합"""
    combined_state = np.zeros_like(state1)
    
    # Mutual Information을 바탕으로 각 상태에 가중치를 부여하여 결합
    for i in range(4):  # [x, y, theta, velocity]
        weight1 = mi[i] / (mi[i] + 1)  # Mutual Information 비율에 따른 가중치
        weight2 = 1 - weight1
        combined_state[:, i] = weight1 * state1[:, i] + weight2 * state2[:, i]
    
    return combined_state

class HybridMIController(BaseController):
    def __init__(self, horizon, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.mpc_controller = AdaptiveMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
        # self.pure_pursuit_controller = PurePursuitController(lookahead_distance=5.0, dt=dt, wheelbase=wheelbase, map_instance=map_instance)

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # 초기 상태 설정
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        for i in range(len(ref_trajectory)):
            if np.all(self.mpc_controller.is_goal_reached(current_state, goal_position)):
                print("Goal reached successfully!")
                break

            # MPC와 Pure Pursuit에서 각각의 상태 예측
            ref_segment = ref_trajectory[i:i + self.mpc_controller.horizon]
            ref_segment_interpole = transform_trajectory_with_angles(ref_segment[:3], num_points=4, last_segment_factor=1)
            
            control_input, best_predicted_states = self.mpc_controller.optimize_control(current_state, ref_segment)
            # next_state_mpc = self.mpc_controller.apply_control(current_state, control_input)

            # target_state = self.pure_pursuit_controller.find_target_state(current_state, ref_trajectory)
            # steering_angle_pure_pursuit = self.pure_pursuit_controller.compute_control(current_state, target_state)
            # next_state_pure_pursuit = self.pure_pursuit_controller.apply_control(current_state, steering_angle_pure_pursuit, velocity=0.5)

            print(f"Ref Trajectory: {ref_segment_interpole}")
            print(f"Predicted State MPC: {best_predicted_states}")

            # 현재 스텝에서 두 경로 간의 Mutual Information 계산
            mi = mutual_information(np.array(ref_segment_interpole), np.array(best_predicted_states))
            print(f"Mutual Information: {mi}")

            # 두 경로를 결합하여 최적 경로 생성
            combined_state = combine_states(np.array(ref_segment_interpole), np.array(best_predicted_states), mi)

            # 다음 상태로 업데이트
            current_state = combined_state[0]
            trajectory.append(current_state)

            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        total_distance = calculate_trajectory_distance(trajectory)

        print("Trajectory following completed.")
        return True, total_distance, np.array(trajectory)

def main(map_type="RandomGridMap"):
    if map_type == "ParkingLot":
        map_instance = ParkingLot(width=20, height=20)
    else:
        map_instance = RandomGridMap(width=20, height=20)

    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Path Planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map(title="Path Planning with MPC and Pure Pursuit")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    if not isReached:
        print("TRRT* was unable to generate a valid path.")
        return

    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)

    horizon = 10  # MPC horizon
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = HybridMIController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)

    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
