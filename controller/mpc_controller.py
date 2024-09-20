import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from controller.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class MPCController(BaseController):
    def __init__(self, horizon, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.horizon = horizon

    def apply_control(self, state, control_input):
        x, y, theta, v = state
        v_ref, delta_ref = control_input

        # Update the state using the kinematic bicycle model
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += v / self.wheelbase * np.tan(delta_ref) * self.dt
        v += v_ref * self.dt

        return np.array([x, y, theta, v])

    def compute_cost(self, predicted_states, ref_trajectory):
        cost = 0
        for i in range(len(predicted_states)):
            if i >= len(ref_trajectory):
                break
            state = predicted_states[i]
            ref_state = ref_trajectory[i]
            cost += np.sum((state - ref_state) ** 2)
        return cost

    def optimize_control(self, current_state, ref_trajectory):
        best_control = None
        best_predicted_states = None
        min_cost = float('inf')

        for v_ref in np.linspace(-1, 1, 7):  # 속도 범위를 세밀하게 설정
            for delta_ref in np.linspace(-np.pi/6, np.pi/6, 7):  # 조향 각도 범위 설정
                predicted_states = []
                state = current_state
                for _ in range(self.horizon):
                    state = self.apply_control(state, (v_ref, delta_ref))
                    predicted_states.append(list(state))

                # predicted states와 reference trajectory 길이를 맞춤
                if len(predicted_states) > len(ref_trajectory):
                    predicted_states = predicted_states[:len(ref_trajectory)]

                # 충돌 여부를 확인
                if all(self.is_collision_free(state, s) for s in predicted_states):
                    cost = self.compute_cost(predicted_states, ref_trajectory)
                    if cost < min_cost:
                        min_cost = cost
                        best_control = (v_ref, delta_ref)
                        best_predicted_states = predicted_states

        # best_control이 없을 때, 기본 움직임을 설정하고 최소한의 예측 상태 생성
        if best_control is None:
            best_control = (0.1, 0.0)  # 기본값: 천천히 직진
            # best_predicted_states = []
            # state = current_state
            # for _ in range(self.horizon):
            #     state = self.apply_control(state, best_control)
            #     best_predicted_states.append(list(state))

        return best_control, best_predicted_states

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        is_reached = True

        # Follow the reference trajectory
        for i in range(len(ref_trajectory)):
            if self.is_goal_reached(current_state, goal_position):
                print("Goal reached successfully!")
                break
            
            self.horizon = min(self.horizon, len(ref_trajectory) - i)

            ref_segment = ref_trajectory[i:i + self.horizon]
            control_input = self.optimize_control(current_state, ref_segment)
            next_state = self.apply_control(current_state, control_input)

            adjusted_states = self.avoid_obstacle(current_state, next_state)
            if not self.is_collision_free(current_state, next_state):
                print(f"Collision detected at step {i}. Attempting to avoid obstacle...")
                is_reached, next_state = self.select_best_path(current_state, adjusted_states, goal_position)
                if not is_reached:
                    print("Goal not reachable.")
                    return is_reached, 0, np.array(trajectory)

            current_state = next_state
            trajectory.append(current_state)

            # Plot current state
            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        # If the goal is still not reached, adjust the final position
        if not self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)
    
        total_distance = calculate_trajectory_distance(trajectory)

        print("Trajectory following completed.")
        return is_reached, total_distance, np.array(trajectory)

def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 맵과 장애물 및 시작/목표 지점을 표시
    map_instance.plot_map(title="MPC Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    # Create Informed TRRT* planner
    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)

    # Ensure the route generation is completed
    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
        if not isReached:
            print("TRRT* was unable to generate a valid path.")
            return

    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)
    print(ref_trajectory)

    # Plot Theta* Path
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path 
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    # MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    mpc_controller = MPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Follow the trajectory using the MPC controller
    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory  = mpc_controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
