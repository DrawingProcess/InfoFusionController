import matplotlib.pyplot as plt
import math  # Import math for square root calculations

from map.grid_map import GridMap

class FixedGridMap(GridMap):
    def __init__(self, width=20, height=20):
        super().__init__(width=width, height=height)

    def add_fixed_rectangle(self, min_x, min_y, max_x, max_y):
        # 장애물 좌표 추가
        fixed_obstacles = [
            [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        ]
        
        # 장애물 위치를 grid와 obstacles 리스트에 추가
        for group in fixed_obstacles:
            for x, y in group:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.obstacles.append((x, y))
                    self.grid[y][x] = 1
        
        # 장애물 경계선 추가
        box_lines = [
            ((min_x, min_y), (max_x, min_y)),  # 상단 라인
            ((min_x, max_y), (max_x, max_y)),  # 하단 라인
            ((min_x, min_y), (min_x, max_y)),  # 좌측 라인
            ((max_x, min_y), (max_x, max_y)),  # 우측 라인
        ]
        self.obstacle_lines.extend(box_lines)
        
    def add_fixed_circle(self, center_x, center_y, radius):
        # Ensure the list exists to store circular obstacles
        if not hasattr(self, 'circular_obstacles'):
            self.circular_obstacles = []
        
        # Store the circle's parameters
        self.circular_obstacles.append((center_x, center_y, radius))
        
        # Add the circle to the grid
        for x in range(center_x - radius, center_x + radius + 1):
            for y in range(center_y - radius, center_y + radius + 1):
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
    map_instance = FixedGridMap(width=50, height=50)
    # Add rectangular obstacles
    map_instance.add_fixed_rectangle(13, 16, 24, 27)
    map_instance.add_fixed_rectangle(30, 5, 42, 15)
    map_instance.add_fixed_rectangle(42, 15, 48, 20)
    
    # Add circular obstacles
    map_instance.add_fixed_circle(center_x=33, center_y=30, radius=4)

    map_instance.plot_map(title="Fixed Grid Map")
    plt.savefig("results/map_fixed_grid_map.png")
    plt.show()
