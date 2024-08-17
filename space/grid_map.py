import matplotlib.pyplot as plt
import random
import math

from route_planner.geometry import Pose

class GridMap:
    def __init__(self, lot_width=82, lot_height=63):
        self.lot_width = lot_width
        self.lot_height = lot_height
        self.grid = [[0 for _ in range(self.lot_width)] for _ in range(self.lot_height)]

        self.obstacles = []
        self.obstacle_lines = []

        # 외벽 생성
        self.create_outer_walls()

    def create_outer_walls(self):
        # 주차장의 외벽을 생성하는 함수
        for x in range(self.lot_width):
            self.obstacles.append((x, 0))  # 아래쪽 외벽
            self.obstacles.append((x, self.lot_height - 1))  # 위쪽 외벽
        for y in range(1, self.lot_height - 1):
            self.obstacles.append((0, y))  # 왼쪽 외벽
            self.obstacles.append((self.lot_width - 1, y))  # 오른쪽 외벽
        self.obstacle_lines.extend([
            [(0, 0), (0, self.lot_height - 1)],
            [(0, 0), (self.lot_width - 1, 0)],
            [(self.lot_width - 1, 0), (self.lot_width - 1, self.lot_height - 1)],
            [(0, self.lot_height - 1), (self.lot_width - 1, self.lot_height - 1)],
        ])

    def get_grid_index(self, x, y):
        return x + y * self.lot_width

    def is_obstacle(self, x, y):
        return self.grid[y][x] == 1

    def is_valid_position(self, x, y):
        # 주어진 좌표가 장애물이 아니고 맵 범위 내에 있는지 확인
        return 0 <= x < self.lot_width and 0 <= y < self.lot_height and not self.is_obstacle(x, y)

    def get_random_valid_position(self):
        while True:
            x = random.randint(0, self.lot_width - 1)
            y = random.randint(0, self.lot_height - 1)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(0))
            
    def get_random_valid_start_position(self):
        while True:
            x = random.randint(0, self.lot_width // 4)
            y = random.randint(0, self.lot_height // 4)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(0))
            
    def get_random_valid_goal_position(self):
        while True:
            x = random.randint(self.lot_width * 3 // 4, self.lot_width - 1)
            y = random.randint(self.lot_height * 3 // 4, self.lot_height - 1)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(90))

    def is_not_crossed_obstacle(self, previous_node, current_node):
        is_cross_line = any(
            [
                self.intersect(obstacle_line, [previous_node, current_node])
                for obstacle_line in self.obstacle_lines
            ]
        )
        return (
            current_node not in set(self.obstacles)
            and 0 < current_node[0] < self.lot_width
            and 0 < current_node[1] < self.lot_height
            and not is_cross_line
        )

    def intersect(self, line1, line2):
        A = line1[0]
        B = line1[1]
        C = line2[0]
        D = line2[1]
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def plot_map(self, path=None):
        obstacle_x = [x for x, y in self.obstacles]
        obstacle_y = [y for x, y in self.obstacles]
        plt.plot(obstacle_x, obstacle_y, ".k")  # 장애물은 검은색 사각형으로 표시
        # plt.plot(obstacle_x, obstacle_y, "sk")  # 장애물은 검은색 사각형으로 표시

        # Plot the obstacle lines
        for line in self.obstacle_lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, "k-")  # 장애물 라인은 검은 실선으로 표시

        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            plt.plot(path_x, path_y, "-or")  # 경로는 빨간색 원으로 연결된 선으로 표시

if __name__ == "__main__":
    # 맵 크기를 지정하여 GridMap 생성 (예: 100x75)
    grid_map = GridMap(lot_width=100, lot_height=80)
    grid_map.plot_map()

    plt.xlim(-1, grid_map.lot_width + 1)
    plt.ylim(-1, grid_map.lot_height + 1)
    plt.title("Complex Grid Map with Path")
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.show()
