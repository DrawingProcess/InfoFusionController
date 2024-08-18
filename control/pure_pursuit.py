import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, transform_arrays_with_angles
from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from control.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class PurePursuitController(BaseController):
    def __init__(self, lookahead_distance, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.dt = dt
        self.lookahead_distance = lookahead_distance  # Lookahead distance for Pure Pursuit
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

        # Move forward to find the target point based on lookahead distance
        while target_index < len(ref_trajectory) and np.hypot(ref_trajectory[target_index, 0] - x, ref_trajectory[target_index, 1] - y) < self.lookahead_distance:
            target_index += 1

        # If the target point is inside an obstacle, skip it and find the next collision-free point
        while target_index < len(ref_trajectory) and not self.is_collision_free(state, ref_trajectory[target_index]):
            target_index += 1

        return target_index

    def compute_control(self, state, target_point):
        x, y, theta = state[:3]
        target_x, target_y = target_point[:2]

        # 각도 오차 계산
        dx = target_x - x
        dy = target_y - y
        target_angle = math.atan2(dy, dx)

        # Calculate the steering angle using the Pure Pursuit formula
        alpha = target_angle - theta
        steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), self.lookahead_distance)

        return steering_angle

def main(map_type="ComplexGridMap"):
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=20, lot_height=20)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=20, lot_height=20)

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

    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    lookahead_distance = 5.0  # Example lookahead distance in meters
    dt = 0.1  # Assume a fixed time step of 0.1 seconds
    pure_pursuit_controller = PurePursuitController(lookahead_distance, dt, wheelbase, map_instance)

    goal_position = [goal_pose.x, goal_pose.y]
    trajectory = pure_pursuit_controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Pure Pursuit Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
