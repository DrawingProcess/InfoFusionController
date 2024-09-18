import math
import random
import numpy as np
import matplotlib.pyplot as plt

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose, Node
from route_planner.rrt_star_planner import RRTStar

class InformedRRTStar(RRTStar):
    def __init__(self, start, goal, map_instance, max_iter=300, search_radius=10, show_eclipse=False):
        super().__init__(start, goal, map_instance, max_iter, search_radius)
        self.c_best = float("inf")  # Initialize the cost to reach the goal
        self.x_center = np.array([(self.start.x + self.goal.x) / 2.0, (self.start.y + self.goal.y) / 2.0])
        self.c_min = np.linalg.norm(np.array([self.start.x, self.start.y]) - np.array([self.goal.x, self.goal.y]))
        self.C = self.rotation_to_world_frame()
        self.show_eclipse = show_eclipse  # Flag to enable/disable eclipse drawing

    def rotation_to_world_frame(self):
        # Calculate the rotation matrix to align the ellipse with the path
        direction = np.array([self.goal.x - self.start.x, self.goal.y - self.start.y])
        direction = direction / np.linalg.norm(direction)
        perpendicular = np.array([-direction[1], direction[0]])
        return np.vstack((direction, perpendicular)).T

    def sample(self, path_region=None):
        if self.c_best < float("inf"):
            while True:
                x_ball = self.sample_unit_ball()
                x_rand = np.dot(np.dot(self.C, np.diag([self.c_best / 2.0, math.sqrt(self.c_best**2 - self.c_min**2) / 2.0])), x_ball)
                x_rand = x_rand + self.x_center
                x_rand_node = Node(x_rand[0], x_rand[1], 0.0)
                if self.is_within_map_instance(x_rand_node):
                    return x_rand_node
        else:
            return super().sample(path_region)

    def sample_unit_ball(self):
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        sample = np.array([b * math.cos(2 * math.pi * a / b), b * math.sin(2 * math.pi * a / b)])
        return sample

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

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Informed RRT* Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map(title="Informed RRT* Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

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