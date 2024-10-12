import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from utils import calculate_trajectory_distance, transform_trajectory

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from route_planner.geometry import Pose, Node
from route_planner.informed_rrt_star_planner import InformedRRTStar

class InformedRRTSmoothStar(InformedRRTStar):
    def __init__(self, start, goal, map_instance, max_iter=700, search_radius=10, show_eclipse=False):
        super().__init__(start, goal, map_instance, max_iter, search_radius)

    def search_route(self, show_process=True):
        # Generate and optimize the final course
        isReached, total_distance, route_trajectory = super().search_route(show_process)
        if not isReached:
            return isReached, 0, [], []

        # Apply smoothing to the final path
        route_trajectory_opt = self.smooth_path(route_trajectory)
        total_distance = calculate_trajectory_distance(route_trajectory_opt)
        return isReached, total_distance, route_trajectory, route_trajectory_opt

    def smooth_path(self, trajectory):
        # 첫 번째 점을 바로 추가
        smooth_trajectory = [trajectory[0]]
        i = 0

        while i < len(trajectory) - 1:
            for j in range(len(trajectory) - 1, i, -1):
                # trajectory에서 x, y를 직접 꺼내서 Node를 생성
                if self.is_collision_free(Node(trajectory[i][0], trajectory[i][1], 0.0), 
                                          Node(trajectory[j][0], trajectory[j][1], 0.0)):
                    # 충돌이 없다면 해당 좌표를 smooth_trajectory에 추가
                    smooth_trajectory.append(trajectory[j])
                    i = j
                    break

        return np.array(smooth_trajectory)

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

    map_instance.plot_map(title="Informed RRT* Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    route_planner = InformedRRTSmoothStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=True)

    if not isReached:
        print("Goal not reached. No path found.")
    else:
        # Plot Informed RRT* Path
        plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Informed RRT* Path")  # Green dashed line

        # Plot Optimized Path
        plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Optimized Path")  # Red solid line

        plt.legend()
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
