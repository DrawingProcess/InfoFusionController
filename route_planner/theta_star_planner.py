import heapq
import math
import random
import matplotlib.pyplot as plt
from space.parking_lot import ParkingLot

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
        self.f_score = float("inf")  # f_score for A* or Theta*

    def __lt__(self, other):
        return self.f_score < other.f_score

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class ThetaStar:
    def __init__(self, start, goal, parking_lot):
        self.start = start
        self.goal = goal
        self.parking_lot = parking_lot
        self.open_set = []
        self.closed_set = set()
        self.g_score = {self.start: 0}
        self.f_score = {self.start: self.heuristic(self.start, self.goal)}
        self.came_from = {}

    def heuristic(self, node1, node2):
        return math.hypot(node2.x - node1.x, node2.y - node1.y)

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            x = node.x + dx
            y = node.y + dy
            if 0 <= x <= self.parking_lot.lot_width and 0 <= y <= self.parking_lot.lot_height:
                neighbor = Node(x, y, node.cost + math.hypot(dx, dy))
                if self.parking_lot.is_not_crossed_obstacle((node.x, node.y), (x, y)):
                    neighbors.append(neighbor)
        return neighbors

    def line_of_sight(self, node1, node2):
        return self.parking_lot.is_not_crossed_obstacle((node1.x, node1.y), (node2.x, node2.y))

    def update_vertex(self, current_node, neighbor):
        if self.line_of_sight(self.came_from[current_node], neighbor):
            parent = self.came_from[current_node]
            new_g = self.g_score[parent] + self.heuristic(parent, neighbor)
            if new_g < self.g_score.get(neighbor, float('inf')):
                self.came_from[neighbor] = parent
                self.g_score[neighbor] = new_g
                self.f_score[neighbor] = new_g + self.heuristic(neighbor, self.goal)
                heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))
        else:
            new_g = self.g_score[current_node] + self.heuristic(current_node, neighbor)
            if new_g < self.g_score.get(neighbor, float('inf')):
                self.came_from[neighbor] = current_node
                self.g_score[neighbor] = new_g
                self.f_score[neighbor] = new_g + self.heuristic(neighbor, self.goal)
                heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))

    def find_path(self):
        heapq.heappush(self.open_set, (self.f_score[self.start], self.start))
        self.came_from[self.start] = self.start

        while self.open_set:
            _, current_node = heapq.heappop(self.open_set)

            if current_node.x == self.goal.x and current_node.y == self.goal.y:
                return self.reconstruct_path(current_node)

            self.closed_set.add((current_node.x, current_node.y))

            for neighbor in self.get_neighbors(current_node):
                if (neighbor.x, neighbor.y) in self.closed_set:
                    continue

                self.update_vertex(current_node, neighbor)

        return []

    def reconstruct_path(self, current_node):
        path = []
        while current_node != self.start:
            path.append((current_node.x, current_node.y))
            current_node = self.came_from[current_node]
        path.append((self.start.x, self.start.y))
        return path[::-1]

def main():
    # Create a parking lot instance
    parking_lot = ParkingLot()

    # Define start and goal positions
    start_pose = Pose(14.0, 4.0, math.radians(0))
    goal_pose = Pose(50.0, 38.0, math.radians(90))
    
    print(f"Start Theta* Pathfinding (start {start_pose.x, start_pose.y}, goal {goal_pose.x, goal_pose.y})")

    # Plot obstacles
    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k", label="Obstacles")

    # Plot start and goal
    plt.plot(start_pose.x, start_pose.y, "og", label="Start")
    plt.plot(goal_pose.x, goal_pose.y, "xb", label="Goal")

    # Initialize Theta* planner
    start_node = Node(start_pose.x, start_pose.y, 0.0)
    goal_node = Node(goal_pose.x, goal_pose.y, 0.0)
    theta_star = ThetaStar(start_node, goal_node, parking_lot)

    # Find the path
    path = theta_star.find_path()

    if path:
        rx, ry = zip(*path)
        plt.plot(rx, ry, "-r", label="Theta* Path")
        plt.legend()
        plt.title("Theta* Pathfinding")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
