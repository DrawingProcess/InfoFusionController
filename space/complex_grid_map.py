import matplotlib.pyplot as plt
import random
import math

from space.grid_map import GridMap

from route_planner.geometry import Pose

class ComplexGridMap(GridMap):
    def __init__(self, lot_width=82, lot_height=63):
        self.lot_width = lot_width
        self.lot_height = lot_height
        self.grid = [[0 for _ in range(self.lot_width)] for _ in range(self.lot_height)]

        self.obstacles = []
        self.obstacle_lines = []

        # 외벽 생성
        self.create_outer_walls()

        # 장애물 생성 (맵 크기에 비례하여 장애물 수 결정)
        self.create_random_obstacles()

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

    def create_random_obstacles(self):
        num_obstacles = int(self.lot_width * self.lot_height * 0.005)
        for _ in range(num_obstacles):
            self.add_random_shape()

    def add_random_shape(self):
        start_x = random.randint(1, self.lot_width - 10)
        start_y = random.randint(1, self.lot_height - 10)
        width = random.randint(3, 10)
        height = random.randint(3, 10)
        for x in range(start_x, min(start_x + width, self.lot_width)):
            for y in range(start_y, min(start_y + height, self.lot_height)):
                self.grid[y][x] = 1
                self.obstacles.append((x, y))
        self.obstacle_lines.extend([
            [(start_x, start_y), (start_x + width - 1, start_y)],
            [(start_x, start_y), (start_x, start_y + height - 1)],
            [(start_x + width - 1, start_y), (start_x + width - 1, start_y + height - 1)],
            [(start_x, start_y + height - 1), (start_x + width - 1, start_y + height - 1)],
        ])

if __name__ == "__main__":
    # 맵 크기를 지정하여 GridMap 생성 (예: 100x75)
    grid_map = ComplexGridMap(lot_width=100, lot_height=80)
    grid_map.plot_map()

    plt.xlim(-1, grid_map.lot_width + 1)
    plt.ylim(-1, grid_map.lot_height + 1)
    plt.title("Complex Grid Map with Path")
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.show()
