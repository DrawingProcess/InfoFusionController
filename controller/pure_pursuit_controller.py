import numpy as np
import math
import matplotlib.pyplot as plt
import json
import argparse

from utils import calculate_angle, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from controller.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class PurePursuitController(BaseController):
    def __init__(self, lookahead_distance, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.lookahead_distance = lookahead_distance  # Lookahead distance for Pure Pursuit

    def find_target_state(self, state, ref_trajectory):
        x, y, theta = state[:3]
        min_distance = float('inf')
        target_index = 0

        # Find the closest point in the trajectory
        for i in range(len(ref_trajectory)):
            distance = np.hypot(ref_trajectory[i, 0] - x, ref_trajectory[i, 1] - y)
            if distance < min_distance:
                min_distance = distance
                target_index = i

        # Move forward to find the target point based on lookahead distance
        while target_index < len(ref_trajectory) and np.hypot(ref_trajectory[target_index, 0] - x, ref_trajectory[target_index, 1] - y) < self.lookahead_distance:
            target_index += 1

        if target_index >= len(ref_trajectory):
            target_point = ref_trajectory[-1]
        else:
            target_point = ref_trajectory[target_index]

        # target_point를 target_state로 변환 (theta와 velocity 추가)
        velocity = state[3] if len(state) > 3 else 0.0  # 속도가 없을 경우 기본값 0.0 사용
        target_state = [target_point[0], target_point[1], theta, velocity]

        return target_state
    
    def compute_control(self, current_state, target_state):
        x, y, theta, v = current_state
        x_next, y_next, theta_next, v_next = target_state

        # 원하는 속도 계산
        a_ref = (v_next - v) / self.dt  # This is acceleration

        # 위치 변화량 계산
        dx = x_next - x
        dy = y_next - y
        distance = np.hypot(dx, dy)

        # 목표 각도 계산
        desired_theta = np.arctan2(dy, dx)

        # 진행 방향 오차 계산
        theta_error = desired_theta - theta
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))  # [-pi, pi] 범위로 정규화

        # 스티어링 각도 계산
        delta_ref = np.arctan2(2 * self.wheelbase * np.sin(theta_error), self.lookahead_distance)

        # 스티어링 각도 제한 적용
        max_steering_angle = np.radians(30)  # 최대 스티어링 각도 (라디안)
        delta_ref = np.clip(delta_ref, -max_steering_angle, max_steering_angle)

        return a_ref, delta_ref

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

    map_instance.plot_map(title="Pure Pursuit Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    # Create Informed TRRT* planner
    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)

    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)

    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    lookahead_distance = 5.0  # Example lookahead distance in meters
    dt = 0.1  # Assume a fixed time step of 0.1 seconds
    controller = PurePursuitController(lookahead_distance, dt, wheelbase, map_instance)

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
