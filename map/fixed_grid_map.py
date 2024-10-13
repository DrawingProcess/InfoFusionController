import matplotlib.pyplot as plt
import math
import argparse
import json
import random
import numpy as np

from utils import transform_trajectory_with_angles

from map.grid_map import GridMap

class FixedGridMap(GridMap):
    def __init__(self, width=20, height=20, obstacles=None):
        super().__init__(width=width, height=height)

        self.obstacles_dynamic = []
        self.obstacle_lines_dynamic = []

        if obstacles:
            self.add_config_obstacles(obstacles)
        else:
            # Default obstacles if no configuration is provided
            self.add_fixed_rectangle(13, 16, 24, 27)
            self.add_fixed_rectangle(30, 5, 42, 15)
            self.add_fixed_rectangle(42, 15, 48, 20)
            self.add_fixed_circle(33, 30, 4)

    def add_config_obstacles(self, obstacles):
        for obstacle in obstacles:
            if obstacle['type'] == 'rectangle':
                # obstacle['coordinates'] is expected to be a list of four numbers [min_x, min_y, max_x, max_y]
                coords = obstacle['coordinates']
                self.add_fixed_rectangle(coords[0], coords[1], coords[2], coords[3])
            elif obstacle['type'] == 'circle':
                # obstacle['parameters'] is expected to be a list [center_x, center_y, radius]
                params = obstacle['parameters']
                self.add_fixed_circle(params[0], params[1], params[2])
            else:
                print(f"Unknown obstacle type: {obstacle['type']}")

    def add_fixed_rectangle(self, min_x, min_y, max_x, max_y, is_dynamic=False):
        # Add obstacle coordinates
        fixed_obstacles = [
            [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        ]
        valid_points = []
        
        # Add obstacle positions to grid and obstacles list
        for group in fixed_obstacles:
            for x, y in group:
                if self.is_valid_position(x, y):
                    valid_points.append((x, y))
                    self.grid[y][x] = 1
                    if is_dynamic:
                        self.obstacles_dynamic.append((x, y))
                    else:
                        self.obstacles.append((x, y))
        
        if not valid_points:
            return  # 유효한 좌표가 없으면 종료

        # Add obstacle boundary lines
        box_lines = [
            ((min_x, min_y), (max_x, min_y)),  # Top line
            ((min_x, max_y), (max_x, max_y)),  # Bottom line
            ((min_x, min_y), (min_x, max_y)),  # Left line
            ((max_x, min_y), (max_x, max_y)),  # Right line
        ]
        if is_dynamic:
            self.obstacle_lines_dynamic.extend(box_lines)
        else:
            self.obstacle_lines.extend(box_lines)
        
    def add_fixed_circle(self, center_x, center_y, radius):
        # Ensure the list exists to store circular obstacles
        if not hasattr(self, 'circular_obstacles'):
            self.circular_obstacles = []
        
        # Store the circle's parameters
        self.circular_obstacles.append((center_x, center_y, radius))
        
        # Add the circle to the grid
        for x in range(int(center_x - radius), int(center_x + radius + 1)):
            for y in range(int(center_y - radius), int(center_y + radius + 1)):
                if 0 <= x < self.width and 0 <= y < self.height:
                    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance <= radius:
                        self.grid[y][x] = 1
                        self.obstacles.append((x, y))
                        
        # Optional: Add circle perimeter to obstacle_lines (approximation)
        points = []
        num_points = 100  # Number of points to approximate the circle
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        # Connect the points to form a circle outline
        for i in range(len(points)):
            self.obstacle_lines.append((points[i], points[(i + 1) % len(points)]))

    # map_instance.create_random_obstacles_in_path(ref_trajectory, n=3, box_size=(5, 5))
    def create_random_obstacles_in_path(self, ref_trajectory, n=3, box_size=(5, 5)):
        ref_trajectory = transform_trajectory_with_angles(ref_trajectory)
        for _ in range(n):
            idx = random.randint(len(ref_trajectory)//4, len(ref_trajectory)*3//4 - 1)
            x, y, _, _ = ref_trajectory[idx]
            if self.is_valid_position(x, y):
                pos = random.randint(0, box_size[0] // 2)
                self.add_fixed_rectangle(int(x) - pos, int(y) - pos, int(x) - pos + box_size[0], int(y) - pos + box_size[1], is_dynamic=True)


    def plot_map(self, title, map_path=None, path=None):
        if map_path:
            # Read and display background image
            bg_image = plt.imread(map_path)
            plt.imshow(bg_image, extent=[0, self.width, 0, self.height], origin='lower', cmap=plt.cm.gray)
        
        obstacle_x = [x for x, y in self.obstacles]
        obstacle_y = [y for x, y in self.obstacles]
        plt.plot(obstacle_x, obstacle_y, ".k")  # 장애물은 검은색 점으로 표시

        # 사각형 장애물의 경계선 표시
        for line in self.obstacle_lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, "k-")  # 장애물 라인은 검은 실선으로 표시
        
        if self.obstacles_dynamic is not None:
            obstacle_x = [x for x, y in self.obstacles_dynamic]
            obstacle_y = [y for x, y in self.obstacles_dynamic]
            plt.plot(obstacle_x, obstacle_y, ".g")  # 동적 장애물은 빨간색 점으로 표시

            for line in self.obstacle_lines_dynamic:
                x_values = [line[0][0], line[1][0]]
                y_values = [line[0][1], line[1][1]]
                plt.plot(x_values, y_values, "g-")

        # 원형 장애물 표시
        for center_x, center_y, radius in self.circular_obstacles:
            circle = plt.Circle((center_x, center_y), radius, color='black', fill=False)
            plt.gca().add_patch(circle)

        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            plt.plot(path_x, path_y, "-or")  # 경로는 빨간색 원으로 연결된 선으로 표시

        plt.xlim(-1, self.width + 1)
        plt.ylim(-1, self.height + 1)
        plt.title(title)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis("equal")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Fixed Grid Map with obstacles.')
    parser.add_argument('--conf', help='Path to configuration JSON file', default=None)
    args = parser.parse_args()

    if args.conf:
        # Read the JSON file and extract parameters
        with open(args.conf, 'r') as f:
            config = json.load(f)

        width = config.get('width', 50)
        height = config.get('height', 50)
        obstacles = config.get('obstacles', [])
    else:
        # Use default parameters
        width = 50
        height = 50
        obstacles = None  # Will trigger default obstacles in the class

    map_instance = FixedGridMap(width=width, height=height, obstacles=obstacles)
    map_instance.plot_map(title="Fixed Grid Map")
    plt.savefig("results/map_fixedgrid.png")
    plt.show()
