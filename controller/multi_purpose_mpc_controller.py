import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from controller.mpc_controller import MPCController
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar


class MultiPurposeMPCController(MPCController):
    def __init__(self, horizon, dt, wheelbase, map_instance, mode="path_tracking"):
        super().__init__(horizon, dt, wheelbase, map_instance)
        self.mode = mode  # 모드 설정 (path_tracking, time_optimal, obstacle_avoidance)
        self.weight_matrices = self.get_weight_matrices()

    def get_weight_matrices(self):
        # 모드에 따른 가중치 행렬 설정
        if self.mode == "path_tracking":
            return {
                "state_deviation": 10.0,  # 경로 추적에서 위치 편차의 가중치
                "control_effort": 1.0     # 제어 입력의 가중치
            }
        elif self.mode == "time_optimal":
            return {
                "state_deviation": 1.0,   # 시간 최적화 주행에서 위치 편차의 가중치
                "control_effort": 0.1,    # 제어 입력의 가중치 (속도 최대화)
                "time_penalty": 10.0      # 시간 페널티 가중치
            }
        elif self.mode == "obstacle_avoidance":
            return {
                "state_deviation": 5.0,   # 장애물 회피에서 위치 편차의 가중치
                "control_effort": 1.0,    # 제어 입력의 가중치
                "obstacle_penalty": 10.0  # 장애물 페널티 가중치
            }
        else:
            raise ValueError("Invalid mode selected for MultiPurposeMPC.")

    def compute_cost(self, predicted_states, ref_trajectory):
        cost = 0
        for i in range(len(predicted_states)):
            if i >= len(ref_trajectory):
                break
            state = predicted_states[i]
            ref_state = ref_trajectory[i]
            # 경로 추적에서의 위치 편차 가중치
            state_deviation = np.sum((state - ref_state) ** 2) * self.weight_matrices["state_deviation"]
            control_effort = np.sum((state[3] - ref_state[3]) ** 2) * self.weight_matrices["control_effort"]

            cost += state_deviation + control_effort

            # 시간 최적화 모드의 경우
            if self.mode == "time_optimal":
                time_penalty = state[3] * self.weight_matrices["time_penalty"]  # 속도 페널티 추가
                cost += time_penalty

            # 장애물 회피 모드의 경우
            if self.mode == "obstacle_avoidance" and self.map_instance.is_obstacle_near(state):
                obstacle_penalty = self.weight_matrices["obstacle_penalty"]
                cost += obstacle_penalty

        return cost

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # 초기 상태와 경로 초기화
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        # 경로를 따르며 목표 지점에 도달
        for i in range(len(ref_trajectory) - self.horizon):
            ref_segment = ref_trajectory[i:i + self.horizon]
            control_input, predicted_states = self.optimize_control(current_state, ref_segment)
            next_state = self.apply_control(current_state, control_input)

            if not self.is_collision_free(current_state, next_state):
                print(f"Collision detected at step {i}. Attempting to avoid obstacle...")
                adjusted_targets = self.avoid_obstacle(current_state, next_state)
                is_reached, next_state = self.select_best_path(current_state, adjusted_targets, goal_position)
                if not is_reached:
                    print("Goal not reachable.")
                    return is_reached, 0, np.array(trajectory)

            current_state = next_state
            trajectory.append(current_state)

            # 현재 상태 시각화
            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        # 목표 지점에 도달하지 않은 경우 위치 조정
        if not self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        total_distance = calculate_trajectory_distance(trajectory)

        print("Trajectory following completed.")
        return True, total_distance, np.array(trajectory)



def main():
    parser = argparse.ArgumentParser(description="Adaptive MPC Route Planner with configurable map, route planner, and controller.")
    parser.add_argument('--map', type=str, default='fixed_grid', choices=['parking_lot', 'fixed_grid', 'random_grid'], help='Choose the map type.')
    parser.add_argument('--conf', help='Path to configuration JSON file', default=None)
    args = parser.parse_args()

    if args.conf:
        # Read the JSON file and extract parameters
        with open(args.conf, 'r') as f:
            config = json.load(f)

        start_pose = Pose(config['start_pose'][0], config['start_pose'][1], config['start_pose'][2])
        goal_pose = Pose(config['goal_pose'][0], config['goal_pose'][1], config['goal_pose'][2])
        width = config.get('width', 50)
        height = config.get('height', 50)
        obstacles = config.get('obstacles', [])
    else:
        # Use default parameters
        width = 50
        height = 50
        start_pose = Pose(2, 2, 0)
        goal_pose = Pose(width - 5, height - 5, 0)
        obstacles = None  # Will trigger default obstacles in the class

    # Map selection using dictionary
    map_options = {
        'parking_lot': ParkingLot,
        'fixed_grid': FixedGridMap,
        'random_grid': RandomGridMap
    }
    map_instance = map_options[args.map](width, height, obstacles)

    if args.map == "random_grid":
        start_pose = map_instance.get_random_valid_start_position()
        goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 맵과 장애물 및 시작/목표 지점을 표시
    map_instance.plot_map(title="Multi-Purpose MPC Route Planner")
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

    # Plot Theta* Path
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path 
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    # MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = MultiPurposeMPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Follow the trajectory using the MPC controller
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
