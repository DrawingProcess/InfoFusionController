import math
import random
import numpy as np
import matplotlib.pyplot as plt

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node
from route_planner.informed_rrt_star_planner import InformedRRTStar

class InformedRRTSmoothStar(InformedRRTStar):
    def __init__(self, start, goal, map_instance, max_iter=300, search_radius=10, show_eclipse=False):
        super().__init__(start, goal, map_instance, max_iter, search_radius)

    def search_route(self, show_process=True):
        # Generate and optimize the final course
        rx, ry = super().search_route(show_process)
        if not rx or not ry:
            return [], [], [], []

        # Apply smoothing to the final path
        rx_opt, ry_opt = self.smooth_path(rx, ry)

        return rx, ry, rx_opt, ry_opt

    def smooth_path(self, path_x, path_y):
        smooth_x, smooth_y = [path_x[0]], [path_y[0]]
        i = 0
        while i < len(path_x) - 1:
            for j in range(len(path_x) - 1, i, -1):
                # Pass a default cost value when creating Node instances
                if self.is_collision_free(Node(path_x[i], path_y[i], 0.0), Node(path_x[j], path_y[j], 0.0)):
                    smooth_x.append(path_x[j])
                    smooth_y.append(path_y[j])
                    i = j
                    break
        return smooth_x, smooth_y

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

    map_instance.plot_map()
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Informed RRT* Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    informed_rrt_star = InformedRRTSmoothStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    rx_rrt, ry_rrt, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=True)

    if not rx_rrt and not ry_rrt:
        print("Goal not reached. No path found.")
    else:
        # Plot Informed RRT* Path
        plt.plot(rx_rrt, ry_rrt, "g--", label="Informed RRT* Path")  # Green dashed line

        # Plot Optimized Path
        plt.plot(rx_opt, ry_opt, "-r", label="Optimized Path")  # Red solid line

        plt.legend()
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
