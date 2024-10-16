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

class StanleyController(BaseController):
    def __init__(self, k, dt, wheelbase, map_instance):
        super().__init__(dt=dt, wheelbase=wheelbase, map_instance=map_instance)
        self.k = k  # Control gain for the cross-track error

    def compute_control(self, current_state, target_state):
        # 현재 상태와 목표 지점의 좌표를 추출
        x, y, theta, v = current_state
        x_next, y_next, theta_next, v_next = target_state
    
        # 원하는 속도 계산
        a_ref = (v_next - v) / self.dt  # This is acceleration

        # 경로 각도 계산
        dx = x_next - x
        dy = y_next - y
        path_angle = math.atan2(dy, dx)
    
        # Heading error (차량의 방향과 경로의 방향 간의 차이)
        heading_error = path_angle - theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
    
        # Cross-track error 계산
        # 경로와 차량 위치 사이의 수직 거리를 계산합니다.
        cross_track_error = np.hypot(dx, dy) * math.sin(heading_error)
    
        # Stanley 조향각 계산
        velocity = max(current_state[3], 0.1)  # 속도가 0에 가까울 때를 방지
        delta_ref = heading_error + math.atan2(self.k * cross_track_error, velocity)
    
        # 차량이 올바른 방향으로 가고 있는지 확인하고, 뒤로 가는 경우를 방지
        if np.cos(heading_error) < 0:  # 만약 헤딩 에러가 90도 이상이면 차량이 뒤로 가려는 경우
            delta_ref += math.pi  # 조향각을 반대로 뒤집어 전진하도록 함
    
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

    print(f"Start Stanley Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map(title="Stanley Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    # Create Informed TRRT* planner
    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)

    # Ensure the route generation is completed
    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)

    # Plot Theta* Path
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    # Stanley Controller
    k = 0.5  # Example gain for the Stanley controller
    dt = 0.1  # Time step for the controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = StanleyController(k, dt, wheelbase, map_instance)

    # Follow the trajectory using the Stanley controller
    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory, steering_angles, accelations  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Stanley Path")
        plt.legend(fontsize=14)
        plt.show()

if __name__ == "__main__":
    main()
