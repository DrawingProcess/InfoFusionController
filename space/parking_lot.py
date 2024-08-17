import matplotlib.pyplot as plt
import random
import math

from space.grid_map import GridMap

from route_planner.geometry import Pose

class ParkingLot(GridMap):
    def __init__(self, lot_width=100, lot_height=80, space_width=6, space_height=11):
        super().__init__(lot_width, lot_height)
        # 주차장 크기 및 주차 공간 크기 설정
        self.lot_width = lot_width
        self.lot_height = lot_height
        self.space_width = space_width
        self.space_height = space_height

        self.obstacles = []
        self.obstacle_lines = []

        self.create_outer_walls()
        self.create_horizontal_lines()
        self.create_vertical_lines()

    def create_horizontal_lines(self):
        # 주차장 내부의 가로선을 생성
        line_y_positions = list(range(self.space_height, self.lot_height, self.space_height + 10))
        for y in line_y_positions:
            for x in range(11, self.lot_width - 10):
                self.obstacles.append((x, y))
            self.obstacle_lines.append([(11, y), (self.lot_width - 10, y)])

    def create_vertical_lines(self):
        # 가로선 사이에 주차 공간을 만드는 세로선 생성
        line_y_positions = list(range(self.space_height, self.lot_height, self.space_height + 10))
        num_spaces_across = (self.lot_width - 22) // self.space_width + 1

        for x in range(num_spaces_across):
            for y in line_y_positions:
                for h in range(self.space_height):
                    self.obstacles.append((11 + x * self.space_width, y + h - self.space_height//2))
                self.obstacle_lines.append([(11 + x * self.space_width, y - self.space_height//2), (11 + x * self.space_width, y + self.space_height//2)])

if __name__ == "__main__":
    # 주차장의 크기를 설정 (예: lot_width=120, lot_height=100 등으로 설정 가능)
    parking_lot = ParkingLot(lot_width=120, lot_height=100)
    parking_lot.plot_map()
    plt.xlim(-1, parking_lot.lot_width + 1)
    plt.ylim(-1, parking_lot.lot_height + 1)
    plt.title("Parking Lot")
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.show()
