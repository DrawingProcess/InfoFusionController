import matplotlib.pyplot as plt

from map.grid_map import GridMap

class FixedGridMap(GridMap):
    def __init__(self, lot_width=20, lot_height=20):
        super().__init__(lot_width=lot_width, lot_height=lot_height)
        self.create_outer_walls()
        self.add_fixed_obstacles(1, 7, 5, 10)
        self.add_fixed_obstacles(5, 2, 10, 8)

    def add_fixed_obstacles(self, min_x, min_y, max_x, max_y):
        # 장애물 좌표 추가
        fixed_obstacles = [
            [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        ]
    
        # 장애물 위치를 grid와 obstacles 리스트에 추가
        for group in fixed_obstacles:
            for x, y in group:
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

if __name__ == "__main__":
    map_instance = FixedGridMap()
    map_instance.plot_map()

    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Pure Pursuit Route Planner")
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()
