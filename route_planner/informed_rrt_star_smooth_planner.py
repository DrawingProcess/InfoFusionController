import math
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import calculate_trajectory_distance, transform_trajectory

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node
from route_planner.informed_rrt_star_planner import InformedRRTStar

class InformedRRTSmoothStar(InformedRRTStar):
    def __init__(self, start, goal, map_instance, max_iter=300, search_radius=10, show_eclipse=False):
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

def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Informed RRT* Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

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
