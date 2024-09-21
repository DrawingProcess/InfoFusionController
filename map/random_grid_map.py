import matplotlib.pyplot as plt
import random
import math

from map.grid_map import GridMap

from route_planner.geometry import Pose

class RandomGridMap(GridMap):
    def __init__(self, width=82, height=63):
        super().__init__(width=width, height=height)
        # 장애물 생성 (맵 크기에 비례하여 장애물 수 결정)
        self.create_random_obstacles()

    def create_random_obstacles(self):
        num_obstacles = int(self.width * self.height * 0.005)
        for _ in range(num_obstacles):
            if random.choice([True, False]):  # 50% 확률로 사각형 또는 원형 장애물 생성
                self.add_random_rectangle()
            else:
                self.add_random_circle()

    def add_random_rectangle(self):
        start_x = random.randint(1, self.width - 10)
        start_y = random.randint(1, self.height - 10)
        width = random.randint(3, 10)
        height = random.randint(3, 10)
        for x in range(start_x, min(start_x + width, self.width)):
            for y in range(start_y, min(start_y + height, self.height)):
                self.grid[y][x] = 1
                self.obstacles.append((x, y))
        self.obstacle_lines.extend([
            [(start_x, start_y), (start_x + width - 1, start_y)],
            [(start_x, start_y), (start_x, start_y + height - 1)],
            [(start_x + width - 1, start_y), (start_x + width - 1, start_y + height - 1)],
            [(start_x, start_y + height - 1), (start_x + width - 1, start_y + height - 1)],
        ])


    def add_random_circle(self):
        center_x = random.randint(10, self.width - 10)
        center_y = random.randint(10, self.height - 10)
        radius = random.randint(3, 7)

        # 원의 좌표를 저장
        self.circular_obstacles.append((center_x, center_y, radius))

        # 그리드 맵에 원의 내부를 장애물로 추가
        for x in range(center_x - radius, center_x + radius + 1):
            for y in range(center_y - radius, center_y + radius + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance <= radius:
                        self.grid[y][x] = 1
                        self.obstacles.append((x, y))

if __name__ == "__main__":
    # 맵 크기를 지정하여 GridMap 생성 (예: 100x75)
    map_instance = RandomGridMap(width=100, height=80)
    map_instance.plot_map(title="Random Grid Map")

    plt.savefig("results/map_random_grid_map.png")
    plt.show()
