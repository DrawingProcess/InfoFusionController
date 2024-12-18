import math
import random
import matplotlib.pyplot as plt
import json
import argparse

from utils import calculate_trajectory_distance, transform_trajectory

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from route_planner.geometry import Pose, Node


class RRTStar:
    def __init__(self, start, goal, map_instance, max_iter=700, search_radius=10):
        self.start = Node(start.x, start.y, 0.0)
        self.goal = Node(goal.x, goal.y, 0.0)
        self.map_instance = map_instance
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.goal_reached = False

    def is_within_map_instance(self, node):
        return 0 <= node.x <= self.map_instance.width and 0 <= node.y <= self.map_instance.height

    def sample(self, path_region=None):
        x = random.uniform(0, self.map_instance.width)
        y = random.uniform(0, self.map_instance.height)
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

    def search_route(self, show_process=False, path_region=None):
        self.goal_reached = False
        for _ in range(self.max_iter):
            rand_node = self.sample(path_region)
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
                    self.goal_reached = True
                    print("Goal Reached")
                    break

            if show_process:
                self.plot_process(new_node)

        if not self.goal_reached:
            print("Goal Not Reached")
            return False, 0, []
        
        rx, ry = self.generate_final_course()
        route_trajectory = transform_trajectory(rx, ry)
        total_distance = calculate_trajectory_distance(route_trajectory)

        return True, total_distance, route_trajectory

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
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if len(self.nodes) % 10 == 0:
            plt.pause(0.001)

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

    map_instance.plot_map(title="RRT* Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    rrt_star = RRTStar(start_pose, goal_pose, map_instance)
    isReached, total_distance, route_trajectory = rrt_star.search_route()

    if not isReached:
        print("Goal not reached. No path found.")
    else:
        plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "-r", label="Planned Path")

        plt.legend(fontsize=14)
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
