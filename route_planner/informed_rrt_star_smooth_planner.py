import math
import random
import numpy as np
import matplotlib.pyplot as plt

from space.parking_lot import ParkingLot
from space.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node
from route_planner.informed_rrt_star_planner import InformedRRTStar

class InformedRRTSmoothStar(InformedRRTStar):
    def __init__(self, start, goal, map_instance, max_iter=300, search_radius=10, show_eclipse=False):
        self.start = Node(start.x, start.y, 0.0)
        self.goal = Node(goal.x, goal.y, 0.0)
        self.map_instance = map_instance
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.c_best = float("inf")
        self.x_center = np.array([(self.start.x + self.goal.x) / 2.0, (self.start.y + self.goal.y) / 2.0])
        self.c_min = np.linalg.norm(np.array([self.start.x, self.start.y]) - np.array([self.goal.x, self.goal.y]))
        self.C = self.rotation_to_world_frame()
        self.show_eclipse = show_eclipse
        self.goal_reached = False  # Goal reached flag

    def search_route(self, show_process=True):
        super().search_route(show_process)

        # Generate and optimize the final course
        rx, ry = self.generate_final_course()
        theta_star = ThetaStar(self.map_instance)
        path_nodes = [Node(x, y, 0) for x, y in zip(rx, ry)]
        optimized_path_nodes = theta_star.optimize_path(path_nodes)

        rx_opt, ry_opt = [node.x for node in optimized_path_nodes], [node.y for node in optimized_path_nodes]
        return rx, ry, rx_opt, ry_opt


class ThetaStar:
    def __init__(self, map_instance):
        self.map_instance = map_instance

    def is_collision_free(self, node1, node2):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        
        # Ensure the segment is within the parking lot boundaries
        if not (0 <= x1 <= self.map_instance.lot_width and 0 <= y1 <= self.map_instance.lot_height 
        and 0 <= x2 <= self.map_instance.lot_width and 0 <= y2 <= self.map_instance.lot_height):
            return False
        
        # Use a high-resolution check along the line segment
        num_checks = int(math.hypot(x2 - x1, y2 - y1) * 10)
        for i in range(num_checks):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            # Ensure the point is within the parking lot and not on an obstacle
            if not self.map_instance.is_not_crossed_obstacle((round(x1), round(y1)), (round(x), round(y))):
                return False
        return True

    def optimize_path(self, path):
        optimized_path = [path[0]]
        for i in range(1, len(path) - 1):
            if not self.is_collision_free(optimized_path[-1], path[i + 1]):
                # If the segment from the previous node to the next node is not collision-free, add the current node to the path
                optimized_path.append(path[i])
        optimized_path.append(path[-1])
        return optimized_path


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
