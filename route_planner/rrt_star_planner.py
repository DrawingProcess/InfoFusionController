import math
import random
import matplotlib.pyplot as plt

from adlab_planning.space.parking_lot import ParkingLot

class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class Node:
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class RRTStar:
    def __init__(self, start, goal, parking_lot, max_iter=300, search_radius=10):
        self.start = Node(start.x, start.y, 0.0)
        self.goal = Node(goal.x, goal.y, 0.0)
        self.parking_lot = parking_lot
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]

    def get_random_node(self):
        x = random.uniform(0, self.parking_lot.lot_width)
        y = random.uniform(0, self.parking_lot.lot_height)
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
        return self.parking_lot.is_not_crossed_obstacle((round(x1), round(y1)), (round(x2), round(y2)))

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

    def search_route(self):
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
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
                    print("Goal Reached")
                    break

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


def main():
    parking_lot = ParkingLot()
    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")

    # Start and goal pose
    start_pose = Pose(14.0, 4.0, math.radians(0))
    goal_pose = Pose(69.0, 59.0, math.radians(90))
    print(f"Start RRT* Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, parking_lot.lot_width + 1)
    plt.ylim(-1, parking_lot.lot_height + 1)
    plt.title("RRT* Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    rrt_star = RRTStar(start_pose, goal_pose, parking_lot)
    rx, ry = rrt_star.search_route()
    plt.plot(rx, ry, "-r")
    plt.pause(0.001)
    plt.show()


if __name__ == "__main__":
    main()
