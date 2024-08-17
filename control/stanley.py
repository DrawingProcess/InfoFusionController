import numpy as np
import math
import matplotlib.pyplot as plt

from space.parking_lot import ParkingLot
from space.complex_grid_map import ComplexGridMap

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar
from utils import calculate_angle, transform_arrays_with_angles

class StanleyController:
    def __init__(self, k, wheelbase, map_instance):
        self.k = k  # Control gain for the cross-track error
        self.wheelbase = wheelbase  # Wheelbase of the vehicle
        self.map_instance = map_instance  # Map instance for collision checking

    def find_nearest_point(self, state, ref_trajectory):
        x, y, theta = state[:3]
        min_distance = float('inf')
        nearest_index = 0

        for i in range(len(ref_trajectory)):
            distance = np.hypot(ref_trajectory[i, 0] - x, ref_trajectory[i, 1] - y)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        return nearest_index

    def compute_control(self, state, ref_trajectory, nearest_index):
        x, y, theta = state[:3]

        # Calculate cross-track error (CTE)
        target_x = ref_trajectory[nearest_index, 0]
        target_y = ref_trajectory[nearest_index, 1]
        dx = target_x - x
        dy = target_y - y

        # Calculate heading error
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # Normalize the angle

        # Calculate cross-track error (CTE) using the sign based on vehicle heading
        cross_track_error = np.hypot(dx, dy)
        cross_track_error = cross_track_error * np.sign(math.sin(target_heading - theta))

        # Stanley control law for steering angle
        steering_angle = heading_error + math.atan2(self.k * cross_track_error, state[3])  # state[3] is the velocity

        return steering_angle

    def apply_control(self, current_state, steering_angle, velocity):
        x, y, theta, v = current_state

        # Update the state using a kinematic bicycle model
        x += v * math.cos(theta) * 0.1  # Assume fixed time step of 0.1 seconds
        y += v * math.sin(theta) * 0.1
        theta += v / self.wheelbase * math.tan(steering_angle) * 0.1
        v = velocity  # Assume constant velocity

        return np.array([x, y, theta, v])

    def follow_trajectory(self, start_pose, ref_trajectory):
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.5])  # Start with a small velocity
        trajectory = [current_state.copy()]

        for _ in range(len(ref_trajectory)):
            nearest_index = self.find_nearest_point(current_state, ref_trajectory)
            if nearest_index >= len(ref_trajectory):
                break  # Reached the end of the path

            steering_angle = self.compute_control(current_state, ref_trajectory, nearest_index)
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
    print(f"Start Stanley Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Stanley Route Planner")
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

    # Stanley Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    k = 1.0  # Example gain for the Stanley controller
    stanley_controller = StanleyController(k, wheelbase, map_instance)

    # Follow the trajectory using the Stanley controller
    trajectory = stanley_controller.follow_trajectory(start_pose, ref_trajectory)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="Stanley Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
