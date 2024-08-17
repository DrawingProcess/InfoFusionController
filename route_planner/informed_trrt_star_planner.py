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

from space.parking_lot import ParkingLot
from space.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node
from route_planner.theta_star_planner import ThetaStar  # Assume Theta* is implemented here

class InformedTRRTStar:
    def __init__(self, start, goal, map_instance, max_iter=300, search_radius=10, show_eclipse=False):
        self.start = Node(start.x, start.y, 0.0)
        self.goal = Node(goal.x, goal.y, 0.0)
        self.map_instance = map_instance
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.c_best = float("inf")  # Initialize the cost to reach the goal
        self.x_center = np.array([(self.start.x + self.goal.x) / 2.0, (self.start.y + self.goal.y) / 2.0])
        self.c_min = np.linalg.norm(np.array([self.start.x, self.start.y]) - np.array([self.goal.x, self.goal.y]))
        self.C = self.rotation_to_world_frame()
        self.show_eclipse = show_eclipse  # Flag to enable/disable eclipse drawing

    def rotation_to_world_frame(self):
        a1 = np.array([self.goal.x - self.start.x, self.goal.y - self.start.y])
        a1 = a1 / np.linalg.norm(a1)
        a2 = np.array([-a1[1], a1[0]])
        return np.vstack((a1, a2)).T

    def narrow_sample(self, x_path, y_path):
        # Narrow the sampling region based on the initial Theta* path
        path_region = []
        margin = self.search_radius / 2  # You can define the margin to your liking
        for i in range(len(x_path) - 1):
            x1, y1 = x_path[i], y_path[i]
            x2, y2 = x_path[i + 1], y_path[i + 1]
            path_region.append((x1, y1, x2, y2, margin))
        return path_region

    def sample(self, path_region=None):
        if self.c_best < float("inf"):
            while True:
                x_ball = self.sample_unit_ball()
                x_rand = np.dot(np.dot(self.C, np.diag([self.c_best / 2.0, math.sqrt(self.c_best ** 2 - self.c_min ** 2) / 2.0])), x_ball)
                x_rand = x_rand + self.x_center
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

    def sample_unit_ball(self):
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        sample = np.array([b * math.cos(2 * math.pi * a / b), b * math.sin(2 * math.pi * a / b)])
        return sample

    def is_within_map_instance(self, node):
        return 0 <= node.x <= self.map_instance.lot_width and 0 <= node.y <= self.map_instance.lot_height

    def get_random_node(self, path_region=None):
        while True:
            x = random.uniform(0, self.map_instance.lot_width)
            y = random.uniform(0, self.map_instance.lot_height)
            node = Node(x, y, 0.0)
            if path_region is None or self.is_within_region(node, path_region):
                return node

    def get_nearest_node_index(self, node):
        dlist = [(n.x - node.x) ** 2 + (n.y - node.y) ** 2 for n in self.nodes]
        min_index = dlist.index(min(dlist))
        return min_index

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.cost, from_node)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        extend_length = min(d, extend_length)

        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.cost += extend_length

        return new_node

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        return distance, angle

    def is_collision_free(self, node1, node2):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

        # Check if the path crosses any obstacle line
        for line in self.map_instance.obstacle_lines:
            if self.map_instance.intersect(line, [(x1, y1), (x2, y2)]):
                return False

        # Check if the path crosses any obstacle grid cell
        return self.map_instance.is_not_crossed_obstacle((round(x1), round(y1)), (round(x2), round(y2)))

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

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.is_collision_free(new_node, near_node):
                cost = new_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = cost

    def search_route(self, show_process=True):
        # Step 1: Use Theta* to find an initial path
        theta_star = ThetaStar(self.start, self.goal, self.map_instance)
        rx, ry = theta_star.find_path()
        if not rx or not ry:
            # If Theta* cannot find a path, return empty routes
            return [], [], [], []

        path_region = self.narrow_sample(rx, ry)

        for _ in range(self.max_iter):
            rand_node = self.sample(path_region)
            nearest_node = self.nodes[self.get_nearest_node_index(rand_node)]
            new_node = self.steer(nearest_node, rand_node, extend_length=self.search_radius)

            if not self.is_collision_free(nearest_node, new_node):
                continue

            # Dynamic adjustment of search radius based on current best cost
            adaptive_radius = self.search_radius * (self.c_best / self.c_min)
            near_nodes = [node for node in self.nodes if math.hypot(node.x - new_node.x, node.y - new_node.y) <= adaptive_radius]
            best_parent = self.search_best_parent(new_node, near_nodes)

            if best_parent:
                new_node = self.steer(best_parent, new_node)
                new_node.parent = best_parent

            self.nodes.append(new_node)
            self.rewire(new_node, near_nodes)

            if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.search_radius:
                final_node = self.steer(new_node, self.goal)
                if self.is_collision_free(new_node, final_node):
                    self.goal = final_node
                    self.goal.parent = new_node
                    self.nodes.append(self.goal)
                    self.c_best = self.goal.cost  # Update the best cost
                    print("Goal Reached")
                    break

            if show_process:
                self.plot_process(new_node)

        if self.goal.parent is None:
            # If goal is not reached, return empty paths
            return [], [], [], []

        # Return both the Theta* path and the final path generated by Informed TRRT*
        rx_opt, ry_opt = self.generate_final_course()

        # Apply smoothing to the final path
        rx_opt, ry_opt = self.smooth_path(rx_opt, ry_opt)

        return rx, ry, rx_opt, ry_opt

    def generate_final_course(self):
        rx, ry = [], []
        node = self.goal
        while node.parent is not None:
            rx.append(node.x)
            ry.append(node.y)
            node = node.parent
        rx.append(self.start.x)
        ry.append(self.start.y)
        return rx[::-1], ry[::-1]

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

    def plot_process(self, node):
        plt.plot(node.x, node.y, "xc")
        if self.show_eclipse and self.c_best < float("inf"):
            self.plot_ellipse()
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if len(self.nodes) % 10 == 0:
            plt.pause(0.001)

    def plot_ellipse(self):
        U, s, Vt = np.linalg.svd(np.dot(self.C, np.diag([self.c_best / 2.0, math.sqrt(self.c_best**2 - self.c_min**2) / 2.0])))
        angle = math.atan2(U[1, 0], U[0, 0])
        angle = angle * 180.0 / math.pi

        a = s[0]  # Semi-major axis
        b = s[1]  # Semi-minor axis

        # Plot the ellipse representing the sampling area
        ellipse = plt.matplotlib.patches.Ellipse(xy=self.x_center, width=a * 2.0, height=b * 2.0, angle=angle, edgecolor='b', fc='None', lw=1, ls='--')
        plt.gca().add_patch(ellipse)

def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    obstacle_x = [obstacle[0] for obstacle in map_instance.obstacles]
    obstacle_y = [obstacle[1] for obstacle in map_instance.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Informed-TRRT* Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Informed-TRRT* Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    informed_trrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    rx, ry, rx_opt, ry_opt = informed_trrt_star.search_route(show_process=False)

    if not rx and not ry:
        print("Goal not reached. No path found.")
    else:
        # Plot Theta* Path
        plt.plot(rx, ry, "g--", label="Theta* Path")  # Green dashed line

        # Plot Optimized Path
        plt.plot(rx_opt, ry_opt, "-r", label="Informed TRRT Path")  # Red solid line

        plt.legend()
        plt.pause(0.001)
        plt.show()

if __name__ == "__main__":
    main()

