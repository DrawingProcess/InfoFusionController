import matplotlib.pyplot as plt
import random
import math

from route_planner.geometry import Pose

class GridMap:
    def __init__(self, width=82, height=63):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(self.width + 1)] for _ in range(self.height + 1)]

        self.obstacles = []
        self.obstacle_lines = []
        self.circular_obstacles = []

        # 외벽 생성
        self.create_outer_walls()

    def create_outer_walls(self):
        # 주차장의 외벽을 생성하는 함수
        for x in range(self.width):
            self.obstacles.append((x, 0))  # 아래쪽 외벽
            self.obstacles.append((x, self.height))  # 위쪽 외벽
        for y in range(1, self.height):
            self.obstacles.append((0, y))  # 왼쪽 외벽
            self.obstacles.append((self.width, y))  # 오른쪽 외벽
        self.obstacle_lines.extend([
            [(0, 0), (0, self.height)],
            [(0, 0), (self.width, 0)],
            [(self.width, 0), (self.width, self.height)],
            [(0, self.height), (self.width, self.height)],
        ])

    def get_grid_index(self, x, y):
        return x + y * self.width
    
    def get_circle_obstacles(self):
        # 원형 장애물들의 (중심 좌표, 반지름)을 반환
        return self.circular_obstacles

    def is_obstacle(self, x, y):
        return (x, y) in self.obstacles

    def is_valid_position(self, x, y):
        # 주어진 좌표가 장애물이 아니고 맵 범위 내에 있는지 확인
        return 0 <= x < self.width and 0 <= y < self.height and not self.is_obstacle(x, y)

    def get_random_valid_position(self):
        while True:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(0))
            
    def get_random_valid_start_position(self):
        while True:
            x = random.randint(0, self.width // 4)
            y = random.randint(0, self.height // 4)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(0))
            
    def get_random_valid_goal_position(self):
        while True:
            x = random.randint(self.width * 3 // 4, self.width)
            y = random.randint(self.height * 3 // 4, self.height)
            if self.is_valid_position(x, y):
                return Pose(x, y, math.radians(90))

    def is_not_crossed_obstacle(self, previous_node, current_node):
        is_cross_line = any(
            self.intersect(obstacle_line, [previous_node, current_node])
            for obstacle_line in self.obstacle_lines
        )
        is_cross_circle = any(
            self.intersect_circle(center_x, center_y, radius, previous_node, current_node)
            for center_x, center_y, radius in self.get_circle_obstacles()
        )
        return (
            current_node not in set(self.obstacles)
            and 0 < current_node[0] < self.width
            and 0 < current_node[1] < self.height
            and not (is_cross_line or is_cross_circle)
        )

    # intersection check between rectangle obstacle and planning line
    def intersect(self, line1, line2):
        A = line1[0]
        B = line1[1]
        C = line2[0]
        D = line2[1]
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)
    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # intersection check between circle obstacle and planning line
    def intersect_circle(self, center_x, center_y, radius, node1, node2):
        cx, cy = center_x, center_y
        px, py = node1
        qx, qy = node2
        dx, dy = qx - px, qy - py
        fx, fy = px - cx, py - cy

        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - radius * radius

        # Handle the case where the segment length is near-zero
        if a == 0:
            # Check if the point (node1) is inside the circle
            return math.hypot(px - cx, py - cy) <= radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False  # No intersection

        discriminant = math.sqrt(discriminant)

        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if the intersection points are within the segment bounds
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)


    def plot_map(self, title, path=None):
        obstacle_x = [x for x, y in self.obstacles]
        obstacle_y = [y for x, y in self.obstacles]
        plt.plot(obstacle_x, obstacle_y, ".k")  # 장애물은 검은색 점으로 표시

        # 사각형 장애물의 경계선 표시
        for line in self.obstacle_lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, "k-")  # 장애물 라인은 검은 실선으로 표시

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

    def create_random_obstacles_in_path(self, ref_trajectory, n=3, box_size=(5, 5)):
        for _ in range(n):
            idx = random.randint(len(ref_trajectory)//4, len(ref_trajectory)*3//4 - 1)
            x, y, _, _ = ref_trajectory[idx]
            if self.is_valid_position(x, y):
                pos = random.randint(0, box_size[0] // 2)
                self.add_obstacle_box(x - pos, y - pos, box_size[0], box_size[1])

    def add_obstacle_box(self, x, y, width, height):
        """주어진 위치에 장애물 박스 추가"""
        valid_points = []

        """주어진 위치에 장애물 박스 추가"""
        for i in range(width):
            for j in range(height):
                if self.is_valid_position(x + i, y + j):
                    valid_points.append((x + i, y + j))
                    self.obstacles.append((x + i, y + j))

        if not valid_points:
            return  # 유효한 좌표가 없으면 종료

        # 유효한 영역의 최소 및 최대 좌표 계산
        min_x = min(p[0] for p in valid_points)
        max_x = max(p[0] for p in valid_points)
        min_y = min(p[1] for p in valid_points)
        max_y = max(p[1] for p in valid_points)

        # 유효한 영역의 경계를 obstacle_lines에 추가
        self.obstacle_lines.extend([
            [(min_x, min_y), (max_x, min_y)],  # 상단 라인
            [(min_x, min_y), (min_x, max_y)],  # 좌측 라인
            [(max_x, min_y), (max_x, max_y)],  # 우측 라인
            [(min_x, max_y), (max_x, max_y)]   # 하단 라인
        ])
        print(f"Added obstacle box at: ({min_x}, {min_y}) with size {width}x{height}")

if __name__ == "__main__":
    # 맵 크기를 지정하여 GridMap 생성 (예: 100x75)
    map_instance = GridMap(width=100, height=80)
    map_instance.plot_map(title="Base Grid Map")

    plt.savefig("results/map_grid_map.png")
    plt.show()
