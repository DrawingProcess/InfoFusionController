import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import entropy

from utils import calculate_angle, transform_arrays_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from controller.base_controller import BaseController
from controller.mpc_controller import MPCController
from controller.pure_pursuit_controller import PurePursuitController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

def mutual_information(route1, route2):
    """두 경로 간의 상호 정보를 계산"""
    hist1, _ = np.histogram(route1, bins=20, density=True)
    hist2, _ = np.histogram(route2, bins=20, density=True)
    entropy1 = entropy(hist1)
    entropy2 = entropy(hist2)
    joint_hist, _, _ = np.histogram2d(route1[:, 0], route2[:, 0], bins=20, density=True)
    joint_entropy = entropy(joint_hist.flatten())
    mi = entropy1 + entropy2 - joint_entropy
    return mi

def optimize_combined_path(state_mpc, state_pure_pursuit, mi):
    """현재 스텝에서 두 경로를 상호 정보에 따라 결합하여 최적 경로 상태 생성"""
    if mi > 0.5:  # 상호 정보가 낮을 때 하나의 경로를 더 선호
        weight1 = 0.5
        weight2 = 0.5
    elif mi > 0.2:
        weight1 = 0.8
        weight2 = 0.2
    elif mi < 0.2:
        weight1 = 1
        weight2 = 0

    combined_x = weight1 * state_mpc[0] + weight2 * state_pure_pursuit[0]
    combined_y = weight1 * state_mpc[1] + weight2 * state_pure_pursuit[1]
    combined_theta = weight1 * state_mpc[2] + weight2 * state_pure_pursuit[2]

    return np.array([combined_x, combined_y, combined_theta, state_mpc[3]])  # 속도는 MPC를 기준으로

class HybridController(BaseController):
    def __init__(self, mpc_controller, pure_pursuit_controller):
        self.mpc_controller = mpc_controller
        self.pure_pursuit_controller = pure_pursuit_controller

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # 초기 상태 설정
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        for i in range(len(ref_trajectory)):
            if self.mpc_controller.is_goal_reached(current_state, goal_position):
                print("Goal reached successfully!")
                break

            # MPC와 Pure Pursuit에서 각각의 상태 예측
            ref_segment = ref_trajectory[i:i + self.mpc_controller.horizon]
            state_mpc = self.mpc_controller.optimize_control(current_state, ref_segment)
            next_state_mpc = self.mpc_controller.apply_control(current_state, state_mpc)

            target_point = self.pure_pursuit_controller.find_target_point(current_state, ref_trajectory)
            steering_angle_pure_pursuit = self.pure_pursuit_controller.compute_control(current_state, target_point)
            next_state_pure_pursuit = self.pure_pursuit_controller.apply_control(current_state, steering_angle_pure_pursuit, velocity=0.5)

            # 현재 스텝에서 두 경로 간의 Mutual Information 계산
            mi = mutual_information(np.array([next_state_mpc]), np.array([next_state_pure_pursuit]))

            # 두 경로를 결합하여 최적 경로 생성
            combined_state = optimize_combined_path(next_state_mpc, next_state_pure_pursuit, mi)

            # 다음 상태로 업데이트
            current_state = combined_state
            trajectory.append(current_state)

            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        total_distance = self.calculate_trajectory_distance(np.array(trajectory))

        print("Trajectory following completed.")
        return True, total_distance, np.array(trajectory)

def main(map_type="ComplexGridMap"):
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=20, lot_height=20)
    else:
        map_instance = ComplexGridMap(lot_width=20, lot_height=20)

    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Path Planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map()
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Path Planning with MPC and Pure Pursuit")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    try:
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    if len(rx_opt) == 0 or len(ry_opt) == 0:
        print("TRRT* was unable to generate a valid path.")
        return

    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

    mpc_controller = MPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=2.5)
    pure_pursuit_controller = PurePursuitController(lookahead_distance=5.0, dt=0.1, wheelbase=2.5, map_instance=map_instance)
    controller = HybridController(mpc_controller, pure_pursuit_controller)

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
