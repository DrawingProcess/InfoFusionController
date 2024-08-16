import math
import random
import numpy as np
import matplotlib.pyplot as plt

from space.parking_lot import ParkingLot
from space.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node


class InformedRRTStar:
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
        self.goal_reached = False

    def rotation_to_world_frame(self):
        a1 = np.array([self.goal.x - self.start.x, self.goal.y - self.start.y])
        a1 = a1 / np.linalg.norm(a1)
        a2 = np.array([-a1[1], a1[0]])
        return np.vstack((a1, a2)).T

    def sample(self):
        if self.c_best < float("inf"):
            while True:
                x_ball = self.sample_unit_ball()
                x_rand = np.dot(np.dot(self.C, np.diag([self.c_best / 2.0, math.sqrt(self.c_best**2 - self.c_min**2) / 2.0])), x_ball)
                x_rand = x_rand + self.x_center
                x_rand_node = Node(x_rand[0], x_rand[1], 0.0)
                if self.is_within_map_instance(x_rand_node):
                    return x_rand_node
        else:
            return self.get_random_node()

    def sample_unit_ball(self):
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        sample = np.array([b * math.cos(2 * math.pi * a / b), b * math.sin(2 * math.pi * a / b)])
        return sample

    def is_within_map_instance(self, node):
        return 0 <= node.x <= self.map_instance.lot_width and 0 <= node.y <= self.map_instance.lot_height

    def get_random_node(self):
        x = random.uniform(0, self.map_instance.lot_width)
        y = random.uniform(0, self.map_instance.lot_height)
        return Node(x, y, 0.0)

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
        return self.map_instance.is_not_crossed_obstacle((round(x1), round(y1)), (round(x2), round(y2)))

    def search_best_parent(self, new_node, near_nodes):
        best_parent = None
        min_cost = float("inf")
        for near_node in near_nodes:
            if self.is_collision_free(near_node, new_node):
                cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
                if cost < min_cost:
                    best_parent = near_node
                    min_cost = cost
        return best_parent

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.is_collision_free(new_node, near_node):
                cost = new_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = cost

    def search_route(self, show_process=True):
        for _ in range(self.max_iter):
            rand_node = self.sample()
            nearest_node = self.nodes[self.get_nearest_node_index(rand_node)]
            new_node = self.steer(nearest_node, rand_node, extend_length=self.search_radius)

            if not self.is_collision_free(nearest_node, new_node):
                continue

            near_nodes = [node for node in self.nodes if math.hypot(node.x - new_node.x, node.y - new_node.y) <= self.search_radius]
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
                    self.goal_reached = True  # Mark goal as reached
                    print("Goal Reached")
                    break

            if show_process:
                self.plot_process(new_node)

        # 목표에 도달하지 못한 경우 빈 경로 반환
        if not self.goal_reached:
            print("Goal Not Reached")
            return [], []

        return self.generate_final_course()

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

        a = s[0]
        b = s[1]

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
    print(f"Start Informed RRT* Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Informed RRT* Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    informed_rrt_star = InformedRRTStar(start_pose, goal_pose, map_instance, show_eclipse=True)
    rx, ry = informed_rrt_star.search_route(show_process=True)

    if not rx and not ry:
        print("Goal not reached. No path found.")
    else:
        plt.plot(rx, ry, "-r", label="Planned Path")

        plt.legend()
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()