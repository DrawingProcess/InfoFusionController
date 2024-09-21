import math
import matplotlib.pyplot as plt

from utils import transform_trajectory, calculate_trajectory_distance

from map.parking_lot import ParkingLot
from map.random_grid_map import RandomGridMap

from route_planner.geometry import Pose, Node

class AStarRoutePlanner:
    def __init__(self, start_pose, goal_pose, map_instance):
        self.start_node = Node(start_pose.x, start_pose.y, 0.0, -1)
        self.goal_node = Node(goal_pose.x, goal_pose.y, 0.0, -1)
        self.map_instance: ParkingLot = map_instance

        # Motion Model: dx, dy, cost
        self.motions = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

    def search_route(self, show_process=False):
        open_set = {self.map_instance.get_grid_index(self.start_node.x, self.start_node.y): self.start_node}
        closed_set = {}

        while open_set:
            current_node_index = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calculate_heuristic_cost(open_set[o]),
            )
            current_node = open_set[current_node_index]

            if show_process:
                self.plot_process(current_node, closed_set)

            if current_node.x == self.goal_node.x and current_node.y == self.goal_node.y:
                print("Find goal")
                # goal_node의 parent_node_index를 현재 노드의 parent_node_index로 설정
                self.goal_node = current_node
                self.goal_node.parent = current_node.parent
                
                rx, ry = self.process_route(closed_set)
                route_trajectory = transform_trajectory(rx, ry)
                total_distance = calculate_trajectory_distance(route_trajectory)
                return True, total_distance, route_trajectory

            # Remove the item from the open set
            del open_set[current_node_index]

            # Add it to the closed set
            closed_set[current_node_index] = current_node

            # expand_grid search grid based on motion model
            for motion in self.motions:
                next_node = Node(
                    current_node.x + motion[0],
                    current_node.y + motion[1],
                    current_node.cost + motion[2],
                    current_node_index,
                )
                next_node_index = self.map_instance.get_grid_index(
                    next_node.x, next_node.y
                )

                if self.map_instance.is_not_crossed_obstacle(
                        (current_node.x, current_node.y),
                        (next_node.x, next_node.y),
                ):
                    if next_node_index in closed_set:
                        continue

                    if next_node_index not in open_set:
                        open_set[next_node_index] = next_node  # discovered a new node
                    else:
                        if open_set[next_node_index].cost > next_node.cost:
                            # This path is the best until now. record it
                            open_set[next_node_index] = next_node

        print("Cannot find Route")
        return False, 0, []

    def process_route(self, closed_set):
        rx = [round(self.goal_node.x)]
        ry = [round(self.goal_node.y)]
        parent = self.goal_node.parent
        while parent != -1:
            node = closed_set[parent]
            rx.append(node.x)
            ry.append(node.y)
            parent = node.parent
        rx.reverse()
        ry.reverse()
        return rx, ry

    def calculate_heuristic_cost(self, node):
        distance = math.sqrt(
            (node.x - self.goal_node.x) ** 2
            + (node.y - self.goal_node.y) ** 2
        )

        cost = distance
        return cost

    @staticmethod
    def plot_process(current_node, closed_set):
        # show graph
        plt.plot(current_node.x, current_node.y, "xc")
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if len(closed_set.keys()) % 10 == 0:
            plt.pause(0.001)

def main(map_type="RandomGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(width=100, height=75)
    else:  # Default to RandomGridMap
        map_instance = RandomGridMap(width=100, height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start A Star Route Planner (start ({start_pose.x}, {start_pose.y}), end ({goal_pose.x}, {goal_pose.y}))")

    map_instance.plot_map(title="A Star Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    a_star = AStarRoutePlanner(start_pose, goal_pose, map_instance)
    isReached, total_distance, route_trajectory = a_star.search_route(False)

    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "-r")
    plt.pause(0.001)
    plt.show()

if __name__ == "__main__":
    main()
