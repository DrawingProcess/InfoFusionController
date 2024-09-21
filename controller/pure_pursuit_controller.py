import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, transform_trajectory_with_angles
from map.parking_lot import ParkingLot
from map.random_grid_map import RandomGridMap

from controller.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class PurePursuitController(BaseController):
    def __init__(self, lookahead_distance, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.lookahead_distance = lookahead_distance  # Lookahead distance for Pure Pursuit

    def find_target_state(self, state, ref_trajectory):
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

        if target_index >= len(ref_trajectory):
            target_point = ref_trajectory[-1]
        else:
            target_point = ref_trajectory[target_index]

        # target_point를 target_state로 변환 (theta와 velocity 추가)
        velocity = state[3] if len(state) > 3 else 0.0  # 속도가 없을 경우 기본값 0.0 사용
        target_state = [target_point[0], target_point[1], theta, velocity]

        return target_state

    def compute_control(self, state, target_state):
        x, y, theta = state[:3]
        target_x, target_y = target_state[:2]

        # 각도 오차 계산
        dx = target_x - x
        dy = target_y - y
        target_angle = math.atan2(dy, dx)

        # Calculate the steering angle using the Pure Pursuit formula
        alpha = target_angle - theta
        steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), self.lookahead_distance)

        return steering_angle

def main(map_type="RandomGridMap"):
    if map_type == "ParkingLot":
        map_instance = ParkingLot(width=20, height=20)
    else:  # Default to RandomGridMap
        map_instance = RandomGridMap(width=20, height=20)

    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Pure Pursuit Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map(title="Pure Pursuit Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    # Create Informed TRRT* planner
    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)

    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)

    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    lookahead_distance = 5.0  # Example lookahead distance in meters
    dt = 0.1  # Assume a fixed time step of 0.1 seconds
    controller = PurePursuitController(lookahead_distance, dt, wheelbase, map_instance)

    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
