import matplotlib.pyplot as plt
import random
import math

from map.grid_map import GridMap

from route_planner.geometry import Pose

class ComplexGridMap(GridMap):
    def __init__(self, lot_width=82, lot_height=63):
        self.lot_width = lot_width
        self.lot_height = lot_height
        self.grid = [[0 for _ in range(self.lot_width)] for _ in range(self.lot_height)]

        self.obstacles = []
        self.obstacle_lines = []
        self.circular_obstacles = []

        # 외벽 생성
        self.create_outer_walls()

        # 장애물 생성 (맵 크기에 비례하여 장애물 수 결정)
        self.create_random_obstacles()

    def create_random_obstacles(self):
        num_obstacles = int(self.lot_width * self.lot_height * 0.005)
        for _ in range(num_obstacles):
            if random.choice([True, False]):  # 50% 확률로 사각형 또는 원형 장애물 생성
                self.add_random_rectangle()
            else:
                self.add_random_circle()

    def add_random_rectangle(self):
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


    def add_random_circle(self):
        center_x = random.randint(10, self.lot_width - 10)
        center_y = random.randint(10, self.lot_height - 10)
        radius = random.randint(3, 7)

        # 원의 좌표를 저장
        self.circular_obstacles.append((center_x, center_y, radius))

        # 그리드 맵에 원의 내부를 장애물로 추가
        for x in range(center_x - radius, center_x + radius + 1):
            for y in range(center_y - radius, center_y + radius + 1):
                if 0 <= x < self.lot_width and 0 <= y < self.lot_height:
                    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance <= radius:
                        self.grid[y][x] = 1
                        self.obstacles.append((x, y))

    def plot_map(self, path=None):
        obstacle_x = [x for x, y in self.obstacles]
        obstacle_y = [y for x, y in self.obstacles]
        plt.plot(obstacle_x, obstacle_y, ".k")  # 장애물은 검은색 점으로 표시

        # 사각형 장애물의 경계선 표시
        for line in self.obstacle_lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, "k-")  # 장애물 라인은 검은 실선으로 표시

        # 원형 장애물 표시
        for center_x, center_y, radius in self.circular_obstacles:
            circle = plt.Circle((center_x, center_y), radius, color='black', fill=False)
            plt.gca().add_patch(circle)

        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            plt.plot(path_x, path_y, "-or")  # 경로는 빨간색 원으로 연결된 선으로 표시

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
