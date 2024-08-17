import numpy as np
import math
import matplotlib.pyplot as plt

from space.parking_lot import ParkingLot
from space.complex_grid_map import ComplexGridMap

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar
from utils import calculate_angle, transform_arrays_with_angles

class PurePursuitController:
    def __init__(self, lookahead_distance, wheelbase, map_instance):
        self.lookahead_distance = lookahead_distance  # Lookahead distance for Pure Pursuit
        self.wheelbase = wheelbase  # Wheelbase of the vehicle
        self.map_instance = map_instance  # Map instance for collision checking

    # find_target_index 함수: 주어진 상태에서 경로 상의 목표점을 찾습니다.
    def find_target_index(self, state, ref_trajectory): # 주어진 상태에서 경로 상의 목표점을 찾습니다.
        x, y, theta = state[:3]
        min_distance = float('inf')
        target_index = 0

        # Find the closest point in the trajectory
        for i in range(len(ref_trajectory)):
            distance = np.hypot(ref_trajectory[i, 0] - x, ref_trajectory[i, 1] - y)
            if distance < min_distance:
                min_distance = distance
                target_index = i

        # Move forward to find the target point based on lookahead distance
        while target_index < len(ref_trajectory) and np.hypot(ref_trajectory[target_index, 0] - x, ref_trajectory[target_index, 1] - y) < self.lookahead_distance:
            target_index += 1

        return target_index

    # compute_control 함수: 목표점에 도달하기 위해 조향 각도를 계산합니다.
    def compute_control(self, state, target_point):
        x, y, theta = state[:3]

        # Calculate the angle to the target point
        dx = target_point[0] - x
        dy = target_point[1] - y
        target_angle = math.atan2(dy, dx)

        # Calculate the steering angle using the Pure Pursuit formula
        alpha = target_angle - theta
        steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), self.lookahead_distance)

        return steering_angle
    # apply_control 함수: 계산된 조향 각도와 속도를 바탕으로 차량의 다음 상태를 계산합니다.
    def apply_control(self, current_state, steering_angle, velocity):
        x, y, theta, v = current_state

        # Update the state using a kinematic bicycle model
        x += v * math.cos(theta) * 0.1  # Assume fixed time step of 0.1 seconds
        y += v * math.sin(theta) * 0.1
        theta += v / self.wheelbase * math.tan(steering_angle) * 0.1
        v = velocity  # Assume constant velocity

        return np.array([x, y, theta, v])

    # follow_trajectory 함수: 경로를 따라가는 전체 과정입니다.
    def follow_trajectory(self, start_pose, ref_trajectory):
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.5])  # Start with a small velocity
        trajectory = [current_state.copy()]

        for _ in range(len(ref_trajectory)):
            target_index = self.find_target_index(current_state, ref_trajectory)
            if target_index >= len(ref_trajectory):
                break  # Reached the end of the path

            target_point = ref_trajectory[target_index]
            steering_angle = self.compute_control(current_state, target_point)
            current_state = self.apply_control(current_state, steering_angle, velocity=0.5)  # Constant velocity
            trajectory.append(current_state)

            # Plot current state
            plt.plot(current_state[0], current_state[1], "xr")
            plt.pause(0.001)

        return np.array(trajectory)

def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    obstacle_x = [obstacle[0] for obstacle in map_instance.obstacles]
    obstacle_y = [obstacle[1] for obstacle in map_instance.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Pure Pursuit Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

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

    # Ensure the route generation is completed
    try:
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

    # Plot Theta* Path
    plt.plot(rx, ry, "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path
    plt.plot(rx_opt, ry_opt, "-r", label="Informed TRRT* Path")  # Red solid line

    # Pure Pursuit Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    lookahead_distance = 5.0  # Example lookahead distance in meters
    pure_pursuit_controller = PurePursuitController(lookahead_distance, wheelbase, map_instance)

    # Follow the trajectory using the Pure Pursuit controller
    trajectory = pure_pursuit_controller.follow_trajectory(start_pose, ref_trajectory)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="Pure Pursuit Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
