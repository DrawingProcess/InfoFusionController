import numpy as np
import math
import matplotlib.pyplot as plt

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar
from utils import calculate_angle, transform_arrays_with_angles

class BaseController:
    def __init__(self, dt, wheelbase, map_instance):
        self.dt = dt
        self.wheelbase = wheelbase  # Wheelbase of the vehicle
        self.map_instance = map_instance  # Map instance for collision checking

    def find_target_index(self, state, ref_trajectory):
        x, y, theta = state[:3]
        min_distance = float('inf')
        target_index = 0

        # Find the closest point in the trajectory
        for i in range(len(ref_trajectory)):
            distance = np.hypot(ref_trajectory[i, 0] - x, ref_trajectory[i, 1] - y)
            if distance < min_distance:
                min_distance = distance
                target_index = i

        # If the target point is inside an obstacle, skip it and find the next collision-free point
        while target_index < len(ref_trajectory) and not self.is_collision_free(state, ref_trajectory[target_index]):
            target_index += 1

        return target_index
    
    def compute_control(self, current_state, target_point):
        # 현재 상태와 목표 지점의 좌표를 추출
        x, y, theta = current_state[:3]
        target_x, target_y = target_point[:2]

        # 현재 위치와 목표 지점 간의 상대 위치 계산
        dx = target_point[0] - x
        dy = target_point[1] - y
        target_angle = math.atan2(dy, dx)

        # 차량의 조향각 계산 (Pure Pursuit 공식)
        alpha = target_angle - theta
        # 조향각은 (2 * 목표 지점까지의 y 거리) / (lookahead 거리) 로 계산
        lookahead_distance = np.hypot(dx, dy)  # 목표 지점까지의 거리
        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead_distance)

        return steering_angle


    def apply_control(self, current_state, steering_angle, velocity):
        x, y, theta, v = current_state

        # Update the state using a kinematic bicycle model
        x += v * math.cos(theta) * self.dt  # Assume fixed time step of 0.1 seconds
        y += v * math.sin(theta) * self.dt
        theta += v / self.wheelbase * math.tan(steering_angle) * self.dt
        v = velocity  # Assume constant velocity

        return np.array([x, y, theta, v])

    def is_collision_free(self, current_position, target_position, num_checks=10):
        x1, y1 = current_position[:2]
        x2, y2 = target_position[:2]

        if not self.map_instance.is_not_crossed_obstacle((round(x1), round(y1)), (round(x2), round(y2))):
            return False
        return True

    # def avoid_obstacle(self, current_position, target_position):
    #     # 간단한 장애물 회피 로직: 목표 위치에서 벗어나지 않는 방향으로 한 단계 이동
    #     x1, y1 = current_position[:2]
    #     x2, y2 = target_position[:2]

    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             if dx == 0 and dy == 0:
    #                 continue
    #             new_x, new_y = x2 + dx, y2 + dy
    #             if self.map_instance.is_valid_position(new_x, new_y) and self.is_collision_free((x1, y1), (new_x, new_y)):
    #                 return [new_x, new_y, math.atan2(new_y - y1, new_x - x1)]
        
    #     # 회피 실패 시 원래 경로로 복귀
    #     return target_position
    
    def avoid_obstacle(self, current_state, target_state):
        # 현재 위치와 목표 지점의 좌표를 추출
        current_xy = np.array(current_state[:2])
        target_xy = np.array(target_state[:2])

        # 목표 지점으로 향하는 방향 벡터 계산
        direction_vector = target_xy - current_xy
        direction_vector = direction_vector / np.linalg.norm(direction_vector)  # 단위 벡터로 정규화

        # 방향 벡터의 좌우 수직 벡터 계산 (장애물 회피를 위한 후보 방향)
        perpendicular_vector1 = np.array([-direction_vector[1], direction_vector[0]])  # 90도 반시계 방향 회전
        perpendicular_vector2 = np.array([direction_vector[1], -direction_vector[0]])  # 90도 시계 방향 회전

        # 장애물 회피를 위한 거리 설정
        adjustment_distance = 2.0  # 이 값을 조정할 수 있습니다.

        # 두 가지 회피 경로 생성 (목표 지점과 가까운 쪽을 우선 탐색)
        adjusted_target1_xy = target_xy + perpendicular_vector1 * adjustment_distance
        adjusted_target2_xy = target_xy + perpendicular_vector2 * adjustment_distance

        # 각 회피 경로가 목표 지점과의 거리를 얼마나 줄이는지 계산
        distance1 = np.linalg.norm(adjusted_target1_xy - target_xy)
        distance2 = np.linalg.norm(adjusted_target2_xy - target_xy)

        # 목표 지점과 더 가까워지는 경로를 우선적으로 확인
        if distance1 < distance2:
            if self.is_collision_free(current_state, np.append(adjusted_target1_xy, target_state[2:])):
                return np.append(adjusted_target1_xy, target_state[2:])
            elif self.is_collision_free(current_state, np.append(adjusted_target2_xy, target_state[2:])):
                return np.append(adjusted_target2_xy, target_state[2:])
        else:
            if self.is_collision_free(current_state, np.append(adjusted_target2_xy, target_state[2:])):
                return np.append(adjusted_target2_xy, target_state[2:])
            elif self.is_collision_free(current_state, np.append(adjusted_target1_xy, target_state[2:])):
                return np.append(adjusted_target1_xy, target_state[2:])

        # 모든 회피 경로가 막혀 있을 경우, 원래 경로로 약간 전진
        adjusted_target_xy = target_xy + direction_vector * adjustment_distance
        return np.append(adjusted_target_xy, target_state[2:])


    def is_goal_reached(self, current_state, goal_position, tolerance=1.0):
        # Check if the current position is close enough to the goal position
        distance_to_goal = np.hypot(current_state[0] - goal_position[0], current_state[1] - goal_position[1])
        return distance_to_goal < tolerance

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.5])  # Start with a small velocity
        trajectory = [current_state.copy()]

        while not self.is_goal_reached(current_state, goal_position):
            target_index = self.find_target_index(current_state, ref_trajectory)
            if target_index >= len(ref_trajectory):
                target_point = ref_trajectory[-1]
            else:
                target_point = ref_trajectory[target_index]
            print(target_point)
            # Check if the path to the target point is collision-free
            if not self.is_collision_free(current_state, target_point):
                target_point = self.avoid_obstacle(current_state, target_point)

            steering_angle = self.compute_control(current_state, target_point)
            current_state = self.apply_control(current_state, steering_angle, velocity=0.5)  # Constant velocity
            trajectory.append(current_state)

            # Plot current state
            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        # If the goal is still not reached, adjust the final position
        if not self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        print("Trajectory following completed.")
        return np.array(trajectory)

def main(map_type="ComplexGridMap"):
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Pure Pursuit Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map()
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Pure Pursuit Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    # Create Informed TRRT* planner
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)

    try:
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

    plt.plot(rx, ry, "g--", label="Theta* Path")  # Green dashed line
    plt.plot(rx_opt, ry_opt, "-r", label="Informed TRRT* Path")  # Red solid line

    dt = 0.1  # Assume fixed time step of 0.1 seconds
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = BaseController(dt, wheelbase, map_instance)

    goal_position = [goal_pose.x, goal_pose.y]
    trajectory = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Pure Pursuit Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()