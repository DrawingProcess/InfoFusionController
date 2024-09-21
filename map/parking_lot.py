import matplotlib.pyplot as plt
import random
import math

from map.grid_map import GridMap

from route_planner.geometry import Pose

class ParkingLot(GridMap):
    def __init__(self, width=100, height=80, space_width=6, space_height=11):
        super().__init__(width, height)
        # 주차장 크기 및 주차 공간 크기 설정
        self.width = width
        self.height = height
        self.space_width = space_width
        self.space_height = space_height

        self.obstacles = []
        self.obstacle_lines = []

        self.create_outer_walls()
        self.create_horizontal_lines()
        self.create_vertical_lines()

    def create_horizontal_lines(self):
        # 주차장 내부의 가로선을 생성
        line_y_positions = list(range(self.space_height, self.height, self.space_height + 10))
        for y in line_y_positions:
            for x in range(11, self.width - 10):
                self.obstacles.append((x, y))
            self.obstacle_lines.append([(11, y), (self.width - 10, y)])

    def create_vertical_lines(self):
        # 가로선 사이에 주차 공간을 만드는 세로선 생성
        line_y_positions = list(range(self.space_height, self.height, self.space_height + 10))
        num_spaces_across = (self.width - 22) // self.space_width + 1

        for x in range(num_spaces_across):
            for y in line_y_positions:
                for h in range(self.space_height):
                    self.obstacles.append((11 + x * self.space_width, y + h - self.space_height//2))
                self.obstacle_lines.append([(11 + x * self.space_width, y - self.space_height//2), (11 + x * self.space_width, y + self.space_height//2)])

if __name__ == "__main__":
    # 주차장의 크기를 설정 (예: width=120, height=100 등으로 설정 가능)
    map_instance = ParkingLot(width=120, height=100)
    map_instance.plot_map(title="Parking Lot")

    plt.savefig("results/map_parkinglot.png")
    plt.show()
