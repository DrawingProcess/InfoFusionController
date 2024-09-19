# Improving the Path:
# Try applying these adjustments step-by-step and observing how the path improves:

# 1. Narrow Sampling Region More Aggressively: In narrow_sample, reduce the margin around the initial Theta* path:
# margin = self.search_radius / 2  # Previously, it was set to the full search_radius.

# 2. Increase Heuristic Weight in Parent Selection: Modify the parent selection logic as mentioned earlier:
# heuristic_cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y) + 1.5 * self.heuristic(new_node, self.goal)

# 3. Reduce Search Radius: Lower the search radius:
# self.search_radius = max(5, self.search_radius / 1.5)

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import transform_trajectory, calculate_trajectory_distance

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node
from route_planner.theta_star_planner import ThetaStar  # Assume Theta* is implemented here
from route_planner.informed_rrt_star_planner import InformedRRTStar

class InformedTRRTStar(InformedRRTStar):
    def __init__(self, start, goal, map_instance, max_iter=300, search_radius=10, show_eclipse=False):
        super().__init__(start, goal, map_instance, max_iter, search_radius)

    def narrow_sample(self, trajectory):
        # Narrow the sampling region based on the initial Theta* path
        path_region = []
        margin = self.search_radius / 2  # You can define the margin to your liking
        for i in range(len(trajectory) - 1):
            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]
            path_region.append((x1, y1, x2, y2, margin))
        return path_region

    def calculate_transformation_matrix(self):
        # Calculate the matrix L using the Cholesky decomposition as described in the text
        a1 = self.c_best / 2.0
        a2 = math.sqrt(self.c_best**2 - self.c_min**2) / 2.0
        D = np.diag([a1, a2])

        # Cholesky decomposition to calculate L
        L = np.linalg.cholesky(D)
        return L

    def sample(self, path_region=None):
        if self.c_best < float("inf"):
            while True:
                x_ball = self.sample_unit_ball()
                L = self.calculate_transformation_matrix()
                x_rand = np.dot(self.C, np.dot(L, x_ball)) + self.x_center
                x_rand_node = Node(x_rand[0], x_rand[1], 0.0)
                if self.is_within_map_instance(x_rand_node) and (path_region is None or self.is_within_region(x_rand_node, path_region)):
                    return x_rand_node
        else:
            return self.get_random_node(path_region)

    def is_within_region(self, node, path_region):
        for (x1, y1, x2, y2, margin) in path_region:
            d = abs((y2 - y1) * node.x - (x2 - x1) * node.y + x2 * y1 - y2 * x1) / math.hypot(y2 - y1, x2 - x1)
            if d <= margin:
                return True
        return False

    def get_random_node(self, path_region=None):
        while True:
            x = random.uniform(0, self.map_instance.lot_width)
            y = random.uniform(0, self.map_instance.lot_height)
            node = Node(x, y, 0.0)
            if path_region is None or self.is_within_region(node, path_region):
                return node

    def search_best_parent(self, new_node, near_nodes):
        best_parent = None
        min_cost = float("inf")
        for near_node in near_nodes:
            if self.is_collision_free(near_node, new_node):
                # Improved heuristic: combine distance to goal with path cost
                # heuristic_cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y) + self.heuristic(new_node, self.goal)
                heuristic_cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y) + 1.5 * self.heuristic(new_node, self.goal)
                if heuristic_cost < min_cost:
                    best_parent = near_node
                    min_cost = heuristic_cost
        return best_parent

    def heuristic(self, node, goal):
        # Simple Euclidean distance as a heuristic
        return math.hypot(goal.x - node.x, goal.y - node.y)

    def search_route(self, show_process=True):
        # Step 1: Use Theta* to find an initial path
        theta_star = ThetaStar(self.start, self.goal, self.map_instance)
        isReached, total_distance, route_trajectory = theta_star.search_route()
        if not isReached:
            return isReached, 0, [], []

        path_region = self.narrow_sample(route_trajectory)

        # Return both the Theta* path and the final path generated by Informed TRRT*
        isReached, total_distance, route_trajectory_opt = super().search_route(show_process, path_region)
        if not isReached:
            return isReached, 0, [], []

        # Apply smoothing to the final path
        route_trajectory_opt = self.smooth_path(route_trajectory_opt)
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
    print(f"Start Informed-TRRT* Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map(title="Informed-TRRT* Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    informed_trrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    isReached, total_distance, route_trajectory, route_trajectory_opt = informed_trrt_star.search_route(show_process=False)

    if not isReached:
        print("Goal not reached. No path found.")
    else:
        # Plot Theta* Path
        plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line

        # Plot Optimized Path
        plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT Path")  # Red solid line

        plt.legend()
        plt.pause(0.001)
        plt.show()

if __name__ == "__main__":
    main()

