import heapq
import math
import matplotlib.pyplot as plt

from utils import transform_trajectory, calculate_trajectory_distance

from map.parking_lot import ParkingLot
from map.random_grid_map import RandomGridMap
from route_planner.geometry import Pose, Node

class ThetaStar:
    def __init__(self, start, goal, map_instance):
        # start와 goal이 Pose 객체라면 Node 객체로 변환
        if isinstance(start, Pose):
            start = Node(start.x, start.y, 0.0, -1)
        if isinstance(goal, Pose):
            goal = Node(goal.x, goal.y, 0.0, -1)
            
        self.start = start
        self.goal = goal
        self.map_instance = map_instance
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
            if 0 <= x < self.map_instance.width and 0 <= y < self.map_instance.height:
                neighbor = Node(x, y, node.cost + math.hypot(dx, dy), None)
                if self.map_instance.is_not_crossed_obstacle((node.x, node.y), (x, y)):
                    neighbors.append(neighbor)
        return neighbors

    def line_of_sight(self, node1, node2):
        return self.map_instance.is_not_crossed_obstacle((node1.x, node1.y), (node2.x, node2.y))

    def update_vertex(self, current_node, neighbor):
        if current_node in self.came_from and self.line_of_sight(self.came_from[current_node], neighbor):
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

    def search_route(self, show_process=False):
        heapq.heappush(self.open_set, (self.f_score[self.start], self.start))
        self.came_from[self.start] = self.start

        while self.open_set:
            _, current_node = heapq.heappop(self.open_set)

            if show_process:
                self.plot_process(current_node)
            
            if current_node.x == self.goal.x and current_node.y == self.goal.y:
                rx, ry = self.reconstruct_path(current_node)
                route_trajectory = transform_trajectory(rx, ry)
                total_distance = calculate_trajectory_distance(route_trajectory)
                return True, total_distance, route_trajectory

            self.closed_set.add((current_node.x, current_node.y))

            for neighbor in self.get_neighbors(current_node):
                if (neighbor.x, neighbor.y) in self.closed_set:
                    continue

                self.update_vertex(current_node, neighbor)

        return False, 0, []

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

    @staticmethod
    def plot_process(current_node):
        # show graph
        plt.plot(current_node.x, current_node.y, "xc")
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

def main(map_type="RandomGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(width=100, height=75)
    else:  # Default to RandomGridMap
        map_instance = RandomGridMap(width=100, height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    
    print(f"Start Theta* Pathfinding (start {start_pose}, goal {goal_pose}) with {map_type}")

    # 장애물 시각화
    obstacle_x = [obstacle[0] for obstacle in map_instance.obstacles]
    obstacle_y = [obstacle[1] for obstacle in map_instance.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k", label="Obstacles")

    # 시작 및 목표 위치 시각화
    plt.plot(start_pose[0], start_pose[1], "og", label="Start")
    plt.plot(goal_pose[0], goal_pose[1], "xb", label="Goal")

    # Theta* 알고리즘 초기화
    start_node = Node(start_pose[0], start_pose[1], 0.0, -1)
    goal_node = Node(goal_pose[0], goal_pose[1], 0.0, -1)
    theta_star = ThetaStar(start_node, goal_node, map_instance)

    # 경로 찾기
    isReached, total_distance, route_trajectory = theta_star.search_route()

    if isReached:
        map_instance.plot_map(title="Theta* Route Planner")
        plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "-r", label="Theta* Path")
    else:
        print("No path found!")

if __name__ == "__main__":
    # 사용할 맵 클래스 선택: "ParkingLot" 또는 "RandomGridMap"
    main()
