import heapq
import math
import matplotlib.pyplot as plt

from space.parking_lot import ParkingLot
from route_planner.geometry import Pose, Node


class ThetaStar:
    def __init__(self, start, goal, parking_lot):
        # start와 goal이 Pose 객체라면 Node 객체로 변환
        if isinstance(start, Pose):
            start = Node(start.x, start.y, 0.0, -1)
        if isinstance(goal, Pose):
            goal = Node(goal.x, goal.y, 0.0, -1)
            
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
                neighbor = Node(x, y, node.cost + math.hypot(dx, dy), None)
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

        return [], []  # 빈 경로 반환

    def reconstruct_path(self, current_node):
        x_path = []
        y_path = []
        while current_node != self.start:
            x_path.append(current_node.x)
            y_path.append(current_node.y)
            current_node = self.came_from[current_node]
        x_path.append(self.start.x)
        y_path.append(self.start.y)
        x_path.reverse()
        y_path.reverse()
        return x_path, y_path

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
    start_node = Node(start_pose.x, start_pose.y, 0.0, -1)
    goal_node = Node(goal_pose.x, goal_pose.y, 0.0, -1)
    theta_star = ThetaStar(start_node, goal_node, parking_lot)

    # Find the path
    rx, ry = theta_star.find_path()

    if rx and ry:
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