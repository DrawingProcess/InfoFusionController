import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, transform_arrays_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class BaseController:
    def __init__(self, dt, wheelbase, map_instance):
        self.dt = dt
        self.wheelbase = wheelbase  # Wheelbase of the vehicle
        self.map_instance = map_instance  # Map instance for collision checking
    
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

    def is_collision_free(self, current_position, target_position):
        x1, y1 = current_position[:2]
        x2, y2 = target_position[:2]

        if not self.map_instance.is_not_crossed_obstacle((x1, y1), (x2, y2)):
            return False
        return True
    
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
        adjustment_distances = [0.5, 1.0, 2.0, 4.0]  # 두 개의 다른 거리로 조정

        # 네 가지 회피 경로 생성
        adjusted_targets = []
        for dist in adjustment_distances:
            adjusted_targets.append(target_xy + perpendicular_vector1 * dist)
            adjusted_targets.append(target_xy + perpendicular_vector2 * dist)

        return adjusted_targets

    def select_best_path(self, current_state, adjusted_targets, goal_position):
        best_target = None
        min_distance = float('inf')

        for i, adjusted_target in enumerate(adjusted_targets):
            if self.is_collision_free(current_state, adjusted_target):
                print(f"Adjusted Target {i} is collision-free: {adjusted_target}")
                distance = np.linalg.norm(adjusted_target[:2] - goal_position)
                if distance < min_distance:
                    min_distance = distance
                    best_target = adjusted_target
            else:
                print(f"Adjusted Target {i} is in collision: {adjusted_target}")

        if best_target is not None:
            return True, best_target
        else:
            print("No collision-free path found, staying in place.")
            return False, None

    def is_goal_reached(self, current_state, goal_position, tolerance=0.5):
        # Check if the current position is close enough to the goal position
        distance_to_goal = np.hypot(current_state[0] - goal_position[0], current_state[1] - goal_position[1])
        return distance_to_goal < tolerance

    def find_target_point(self, state, ref_trajectory):
        x, y, theta = state[:3]
        min_distance = float('inf')
        target_index = 0

        # Find the closest point in the trajectory
        for i in range(len(ref_trajectory)):
            distance = np.hypot(ref_trajectory[i, 0] - x, ref_trajectory[i, 1] - y)
            if distance < min_distance:
                min_distance = distance
                target_index = i
        
        if target_index >= len(ref_trajectory):
            target_point = ref_trajectory[-1]
        else:
            target_point = ref_trajectory[target_index]

        return target_point

    def predict_trajectory(self, current_state, target_point, n_steps=10, velocity=0.5):
        # Predict future trajectory using the kinematic bicycle model
        predicted_trajectory = []
        state = current_state.copy()
        for _ in range(n_steps):
            steering_angle = self.compute_control(state, target_point)
            state = self.apply_control(state, steering_angle, velocity)
            predicted_trajectory.append(state.copy())
        return np.array(predicted_trajectory)

    def calculate_trajectory_distance(self, trajectory):
        # 각 점 사이의 유클리드 거리를 계산하여 총 거리를 산출
        distances = np.sqrt(np.diff(trajectory[:, 0])**2 + np.diff(trajectory[:, 1])**2)
        total_distance = np.sum(distances)
        return total_distance

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.5])  # Start with a small velocity
        trajectory = [current_state.copy()]

        predicted_lines = []

        is_reached = False

        while not self.is_goal_reached(current_state, goal_position):
            # Find the target index
            target_point = self.find_target_point(current_state, ref_trajectory)

            # Generate possible adjusted targets to avoid obstacles
            adjusted_targets = self.avoid_obstacle(current_state, target_point)

            if show_process:
                # Clear the old prediction lines if they exist
                for line in predicted_lines:
                    line.remove()
                predicted_lines = []

                # Plot the new prediction lines
                predicted_trajectory = self.predict_trajectory(current_state, target_point)
                predicted_lines.append(plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "b-", label="Predicted Path")[0])
                colors = ["g--", "r--"]
                # labels = ["Left Avoid Path", "Right Avoid Path"]
                for i, adjusted_target in enumerate(adjusted_targets):
                    predicted_traj = self.predict_trajectory(current_state, adjusted_target)
                    predicted_lines.append(plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], colors[i % 2])[0])
                    # predicted_lines.append(plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], colors[i % 2], label=labels[i % 2])[0])
                    # Mark adjusted target points with a red 'X' and add to predicted_lines for clearing later
                    marker = plt.plot(adjusted_target[0], adjusted_target[1], 'rx')[0]
                    predicted_lines.append(marker)
                plt.legend()
                plt.pause(0.001)
            
            if not self.is_collision_free(current_state, target_point):
                is_reached, target_point = self.select_best_path(current_state, adjusted_targets, goal_position)
            
            # Apply control
            steering_angle = self.compute_control(current_state, target_point)
            current_state = self.apply_control(current_state, steering_angle, velocity=0.5)  # Constant velocity
            trajectory.append(current_state)

        # If the goal is still not reached, adjust the final position
        if not self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        total_distance = self.calculate_trajectory_distance(np.array(trajectory))

        print("Trajectory following completed.")
        return is_reached, total_distance, np.array(trajectory)

def main(map_type="ComplexGridMap"):
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Pure Pursuit Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map(title="Pure Pursuit Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

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
    is_reached, trajectory_distance, trajectory = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()