import matplotlib.pyplot as plt

class HousePlan:
    def __init__(self):
        self.house_width = 100
        self.house_height = 80

        self.obstacles = []
        self.obstacle_lines = []

        self.create_outer_walls()
        self.create_rooms()
        self.create_doors()

    def create_outer_walls(self):
        # Create the boundary walls of the house
        for x in range(self.house_width + 1):
            self.obstacles.append((x, 0))
            self.obstacles.append((x, self.house_height))
        for y in range(1, self.house_height):
            self.obstacles.append((0, y))
            self.obstacles.append((self.house_width, y))
        self.obstacle_lines.extend([
            [(0, 0), (0, self.house_height)],
            [(0, 0), (self.house_width, 0)],
            [(self.house_width, 0), (self.house_width, self.house_height)],
            [(0, self.house_height), (self.house_width, self.house_height)],
        ])

    def create_rooms(self):
        # Define the rooms with their boundaries
        rooms = [
            ((10, 10), (30, 30)),  # Room 1
            ((40, 10), (60, 30)),  # Room 2
            ((10, 40), (30, 60)),  # Room 3
            ((40, 40), (60, 60)),  # Room 4
            ((70, 10), (90, 30)),  # Room 5
            ((70, 40), (90, 60)),  # Room 6
        ]

        for room in rooms:
            (x1, y1), (x2, y2) = room
            # Add room boundaries as obstacles
            self.obstacles.extend([
                (x, y1) for x in range(x1, x2 + 1)
            ])
            self.obstacles.extend([
                (x, y2) for x in range(x1, x2 + 1)
            ])
            self.obstacles.extend([
                (x1, y) for y in range(y1, y2 + 1)
            ])
            self.obstacles.extend([
                (x2, y) for y in range(y1, y2 + 1)
            ])
            self.obstacle_lines.extend([
                [(x1, y1), (x2, y1)],
                [(x2, y1), (x2, y2)],
                [(x2, y2), (x1, y2)],
                [(x1, y2), (x1, y1)],
            ])

    def create_doors(self):
        # Define door locations between rooms
        doors = [
            ((30, 20), (40, 20)),  # Door between Room 1 and Room 2
            ((20, 30), (20, 40)),  # Door between Room 1 and Room 3
            ((50, 20), (50, 30)),  # Door between Room 2 and Room 4
            ((30, 50), (40, 50)),  # Door between Room 3 and Room 4
            ((60, 20), (70, 20)),  # Door between Room 2 and Room 5
            ((60, 50), (70, 50)),  # Door between Room 4 and Room 6
            ((70, 30), (70, 40))   # Door between Room 5 and Room 6
        ]
        
        for door in doors:
            (x1, y1), (x2, y2) = door
            # Add door locations as obstacles (points)
            self.obstacles.extend([
                (x1, y1),
                (x2, y2)
            ])
            self.obstacle_lines.append(door)

    def plot_plan(self):
        obstacle_x = [obstacle[0] for obstacle in self.obstacles]
        obstacle_y = [obstacle[1] for obstacle in self.obstacles]
        plt.plot(obstacle_x, obstacle_y, ".k")
        for line in self.obstacle_lines:
            (x1, y1), (x2, y2) = line
            plt.plot([x1, x2], [y1, y2], "k-")
        plt.xlim(-1, self.house_width + 1)
        plt.ylim(-1, self.house_height + 1)
        plt.title("House Floor Plan")
        plt.grid(True)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.show()

if __name__ == "__main__":
    house_plan = HousePlan()
    house_plan.plot_plan()
