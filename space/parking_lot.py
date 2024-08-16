import matplotlib.pyplot as plt
import random
import math

from route_planner.geometry import Pose

class ParkingLot:
    def __init__(self, lot_width=100, lot_height=80, space_width=6, space_height=11):
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

    def create_outer_walls(self):
        # 주차장의 외벽 생성
        for x in range(self.lot_width + 1):
            self.obstacles.append((x, 0))
            self.obstacles.append((x, self.lot_height))
        for y in range(1, self.lot_height):
            self.obstacles.append((0, y))
            self.obstacles.append((self.lot_width, y))
        self.obstacle_lines.extend([
            [(0, 0), (0, self.lot_height)],
            [(0, 0), (self.lot_width, 0)],
            [(self.lot_width, 0), (self.lot_width, self.lot_height)],
            [(0, self.lot_height), (self.lot_width, self.lot_height)],
        ])

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

    def get_grid_index(self, x, y):
        return x + y * self.lot_width

    def is_valid_position(self, x, y):
        # 주어진 좌표가 장애물이 아니고 맵 범위 내에 있는지 확인
        return 0 <= x < self.lot_width and self.space_height <= y < self.lot_height - self.space_height

    def get_random_valid_position(self):
        while True:
            x = random.randint(0, self.lot_width - 1)
            y = random.randint(self.space_height, self.lot_height - 1 - self.space_height)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(0))
            
    def get_random_valid_start_position(self):
        while True:
            x = random.randint(0, self.lot_width // 4)
            y = random.randint(self.space_height, self.lot_height // 4)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(0))
            
    def get_random_valid_goal_position(self):
        while True:
            x = random.randint(self.lot_width * 3 // 4, self.lot_width - 1)
            y = random.randint(self.lot_height * 3 // 4, self.lot_height - 1 - self.space_height)
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

if __name__ == "__main__":
    # 주차장의 크기를 설정 (예: lot_width=120, lot_height=100 등으로 설정 가능)
    parking_lot = ParkingLot(lot_width=120, lot_height=100)
    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")
    plt.xlim(-1, parking_lot.lot_width + 1)
    plt.ylim(-1, parking_lot.lot_height + 1)
    plt.title("Parking Lot")
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.show()
