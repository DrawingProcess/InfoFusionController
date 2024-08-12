"""
ParkingLot for Free Space Route Planner
"""

import matplotlib.pyplot as plt


class ParkingLot:
    def __init__(self):
        self.lot_width: int = 82
        self.lot_height: int = 63

        self.obstacles = []
        self.obstacle_lines = []

        # 주차장 외각
        for x in range(self.lot_width + 1):
            self.obstacles.append((x, 0))
            self.obstacles.append((x, self.lot_height))
        for y in range(1, self.lot_height):
            self.obstacles.append((0, y))
            self.obstacles.append((self.lot_width, y))
        self.obstacle_lines = [
            [(0, 0), (0, self.lot_height)],
            [(0, 0), (self.lot_width, 0)],
            [(self.lot_width, 0), (self.lot_width, self.lot_height)],
            [(0, self.lot_height), (self.lot_width, self.lot_height)],
        ]

        # 주차장 가로선
        for x in range(11, self.lot_width - 10):
            self.obstacles.append((x, 17))
            self.obstacles.append((x, 40))
        self.obstacle_lines.append([(11, 17), (self.lot_width - 10, 17)])
        self.obstacle_lines.append([(11, 40), (self.lot_width - 10, 40)])

        # 주차장 세로선
        for x in range(16):
            for y in range(6):
                self.obstacles.append((x * 4 + 11, y + 11))
                self.obstacles.append((x * 4 + 11, y + 18))
                self.obstacles.append((x * 4 + 11, y + 34))
                self.obstacles.append((x * 4 + 11, y + 41))
                self.obstacles.append((x * 4 + 11, y + 57))
            self.obstacle_lines.append([(x * 4 + 11, 11), (x * 4 + 11, 24)])
            self.obstacle_lines.append([(x * 4 + 11, 34), (x * 4 + 11, 47)])
            self.obstacle_lines.append([(x * 4 + 11, 57), (x * 4 + 11, 63)])

    def get_grid_index(self, x, y):
        return x + y * self.lot_width

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

    # Imitation Code: https://stackoverflow.com/a/9997374
    # Return true if line segments AB and CD intersect
    def intersect(self, line1, line2):
        A = line1[0]
        B = line1[1]
        C = line2[0]
        D = line2[1]
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


if __name__ == "__main__":
    parking_lot = ParkingLot()
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
