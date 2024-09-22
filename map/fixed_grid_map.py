import matplotlib.pyplot as plt
import math
import argparse
import json

from map.grid_map import GridMap

class FixedGridMap(GridMap):
    def __init__(self, width=20, height=20, obstacles=None):
        super().__init__(width=width, height=height)

        if obstacles:
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
        else:
            # Default obstacles if no configuration is provided
            self.add_fixed_rectangle(13, 16, 24, 27)
            self.add_fixed_rectangle(30, 5, 42, 15)
            self.add_fixed_rectangle(42, 15, 48, 20)
            self.add_fixed_circle(33, 30, 4)

    def add_fixed_rectangle(self, min_x, min_y, max_x, max_y):
        # Add obstacle coordinates
        fixed_obstacles = [
            [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        ]
        
        # Add obstacle positions to grid and obstacles list
        for group in fixed_obstacles:
            for x, y in group:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.obstacles.append((x, y))
                    self.grid[y][x] = 1
        
        # Add obstacle boundary lines
        box_lines = [
            ((min_x, min_y), (max_x, min_y)),  # Top line
            ((min_x, max_y), (max_x, max_y)),  # Bottom line
            ((min_x, min_y), (min_x, max_y)),  # Left line
            ((max_x, min_y), (max_x, max_y)),  # Right line
        ]
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
