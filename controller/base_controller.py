import numpy as np
import math
import matplotlib.pyplot as plt
import json
import argparse

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class BaseController:
    def __init__(self, dt, wheelbase, map_instance):
        self.dt = dt
        self.wheelbase = wheelbase  # Wheelbase of the vehicle
        self.map_instance = map_instance  # Map instance for collision checking
    
    def compute_control(self, current_state, next_state):
        x, y, theta, v = current_state
        x_next, y_next, theta_next, v_next = next_state

        a_ref = (v_next - v) / self.dt

        dx = x_next - x
        dy = y_next - y
        distance = np.hypot(dx, dy)

        desired_theta = np.arctan2(dy, dx)

        theta_error = desired_theta - theta
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))  # [-pi, pi] range normalization

        if distance > 0.001:  # Avoid division by zero
            delta_ref = np.arctan2(2 * self.wheelbase * np.sin(theta_error), distance)
        else:
            delta_ref = 0.0

        # Clip the steering angle to the maximum steering angle
        max_steering_angle = np.radians(30) 
        delta_ref = np.clip(delta_ref, -max_steering_angle, max_steering_angle)

        return a_ref, delta_ref

    def apply_control(self, state, control_input):
        x, y, theta, v = state
        a_ref, delta_ref = control_input

        # Update the state using the kinematic bicycle model
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += v / self.wheelbase * np.tan(delta_ref) * self.dt
        v += a_ref * self.dt

        return np.array([x, y, theta, v])

    def is_collision_free(self, current_position, target_position):
        x1, y1 = current_position[:2]
        x2, y2 = target_position[:2]

        if not self.map_instance.is_not_crossed_obstacle((x1, y1), (x2, y2)):
            return False
        return True
    
    def avoid_obstacle(self, current_state, target_state):
        # extract x, y coordinates from current_state and target_state
        current_xy = np.array(current_state[:2])
        target_xy = np.array(target_state[:2])

        # calculate the direction vector from current_state to target_state
        direction_vector = target_xy - current_xy
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # calculate two perpendicular vectors to the direction vector
        perpendicular_vector1 = np.array([-direction_vector[1], direction_vector[0]])
        perpendicular_vector2 = np.array([direction_vector[1], -direction_vector[0]])


        # generate adjusted_targets by adding the perpendicular vectors to the target_state
        adjustment_distances = [0.5, 1.0, 2.0, 4.0, 6.0]
        adjusted_targets = []
        for dist in adjustment_distances:
            adjusted_targets.append(target_xy + perpendicular_vector1 * dist)
            adjusted_targets.append(target_xy + perpendicular_vector2 * dist)

        # convert adjusted_targets to adjusted_states by adding theta and velocity
        adjusted_states = []
        for adjusted_target in adjusted_targets:
            x, y = adjusted_target
            theta = target_state[2]  
            velocity = target_state[3]
            new_state = [x, y, theta, velocity]
            adjusted_states.append(new_state)

        return adjusted_states

    def select_best_path(self, current_state, adjusted_states, goal_position):
        best_target = None
        min_distance = float('inf')

        for i, adjusted_state in enumerate(adjusted_states):
            if self.is_collision_free(current_state, adjusted_state):
                # print(f"Adjusted Target {i} is collision-free: {adjusted_state}")
                distance = np.linalg.norm(np.array(adjusted_state[:2]) - np.array(goal_position))
                if distance < min_distance:
                    min_distance = distance
                    best_target = adjusted_state
            # else:
            #     print(f"Adjusted Target {i} is in collision: {adjusted_state}")

        if best_target is not None:
            return True, best_target
        else:
            print("No collision-free path found, staying in place.")
            return False, None

    def is_goal_reached(self, current_state, goal_position, tolerance=0.5):
        # Check if the current position is close enough to the goal position
        distance_to_goal = np.hypot(current_state[0] - goal_position[0], current_state[1] - goal_position[1])
        return distance_to_goal < tolerance

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

        if target_index >= len(ref_trajectory):
            target_point = ref_trajectory[-1]
        else:
            target_point = ref_trajectory[target_index]

        # Extract the target state from the target point
        velocity = state[3] if len(state) > 3 else 0.0 
        target_state = [target_point[0], target_point[1], theta, velocity]

        return target_state

    def predict_trajectory(self, current_state, target_point, n_steps=10):
        # Predict future trajectory using the kinematic bicycle model
        predicted_trajectory = []
        state = current_state.copy()
        for _ in range(n_steps):
            control_input = self.compute_control(state, target_point)
            state = self.apply_control(state, control_input)
            predicted_trajectory.append(state.copy())
        return np.array(predicted_trajectory)

    def get_ref_segment(self, current_state, ref_trajectory, ref_index):
        max_ref_index = len(ref_trajectory) - 1

        # Limit the search window to ±window_size around ref_index
        window_size = 10  # Adjust this parameter as needed
        search_start = max(ref_index - window_size, 0)
        search_end = min(ref_index + window_size, max_ref_index)

        # Compute distances in the search window
        search_indices = np.arange(search_start, search_end + 1)
        ref_points = ref_trajectory[search_indices, :2]
        distances = np.linalg.norm(ref_points - current_state[:2], axis=1)

        # Find the index of the closest point in the search window
        min_distance_index = np.argmin(distances)
        ref_index = search_indices[min_distance_index]

        # Extract ref_segment from ref_index to ref_index + horizon
        ref_segment_end = min(ref_index + self.horizon, max_ref_index + 1)
        ref_segment = ref_trajectory[ref_index:ref_segment_end]

        # If ref_segment is shorter than horizon, pad it with the last point
        if len(ref_segment) < self.horizon:
            last_point = ref_segment[-1]
            num_padding = self.horizon - len(ref_segment)
            padding = np.tile(last_point, (num_padding, 1))
            ref_segment = np.vstack((ref_segment, padding))

        return ref_segment, ref_index

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.5])  # Start with a small velocity
        trajectory = [current_state.copy()]

        steering_angles = []
        accelations = []
        predicted_lines = []

        is_reached = True

        while not self.is_goal_reached(current_state, goal_position):
            # Find the target index
            target_state = self.find_target_state(current_state, ref_trajectory)

            # Generate possible adjusted targets to avoid obstacles
            adjusted_states = self.avoid_obstacle(current_state, target_state)

            if show_process:
                # Clear the old prediction lines if they exist
                for line in predicted_lines:
                    line.remove()
                predicted_lines = []    

                # Plot the new prediction lines
                predicted_trajectory = self.predict_trajectory(current_state, target_state)
                # print(f"predicted_trajectory: {predicted_trajectory}")
                predicted_lines.append(plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "b-", label="Predicted Path")[0])
                colors = ["g--", "r--"]
                # labels = ["Left Avoid Path", "Right Avoid Path"]
                for i, adjusted_state in enumerate(adjusted_states):
                    predicted_traj = self.predict_trajectory(current_state, adjusted_state)
                    predicted_lines.append(plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], colors[i % 2])[0])
                    # predicted_lines.append(plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], colors[i % 2], label=labels[i % 2])[0])
                    # Mark adjusted target points with a red 'X' and add to predicted_lines for clearing later
                    marker = plt.plot(adjusted_state[0], adjusted_state[1], 'rx')[0]
                    predicted_lines.append(marker)
                plt.pause(0.001)
            
            if not self.is_collision_free(current_state, target_state):
                is_reached, target_state = self.select_best_path(current_state, adjusted_states, goal_position)
                if not is_reached:
                    print("Goal not reachable.")
                    return is_reached, 0, np.array(trajectory), np.array(steering_angles), np.array(accelations)
            
            # Apply control
            control_input = self.compute_control(current_state, target_state)
            next_state = self.apply_control(current_state, control_input)  # Constant velocity
            
            current_state = next_state
            trajectory.append(current_state)

            accelations.append(control_input[0])
            steering_angles.append(control_input[1])

        # If the goal is still not reached, adjust the final position
        if self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        total_distance = calculate_trajectory_distance(trajectory)

        print("Trajectory following completed.")
        return is_reached, total_distance, np.array(trajectory), np.array(steering_angles), np.array(accelations)

def main():
    parser = argparse.ArgumentParser(description="Adaptive MPC Route Planner with configurable map, route planner, and controller.")
    parser.add_argument('--map', type=str, default='fixed_grid', choices=['parking_lot', 'fixed_grid', 'random_grid'], help='Choose the map type.')
    parser.add_argument('--conf', help='Path to configuration JSON file', default=None)
    args = parser.parse_args()

    if args.conf:
        # Read the JSON file and extract parameters
        with open(args.conf, 'r') as f:
            config = json.load(f)

        start_pose = Pose(config['start_pose'][0], config['start_pose'][1], config['start_pose'][2])
        goal_pose = Pose(config['goal_pose'][0], config['goal_pose'][1], config['goal_pose'][2])
        width = config.get('width', 50)
        height = config.get('height', 50)
        obstacles = config.get('obstacles', [])
    else:
        # Use default parameters
        width = 50
        height = 50
        start_pose = Pose(2, 2, 0)
        goal_pose = Pose(width - 5, height - 5, 0)
        obstacles = None  # Will trigger default obstacles in the class

    # Map selection using dictionary
    map_options = {
        'parking_lot': ParkingLot,
        'fixed_grid': FixedGridMap,
        'random_grid': RandomGridMap
    }
    map_instance = map_options[args.map](width, height, obstacles)

    if args.map == "random_grid":
        start_pose = map_instance.get_random_valid_start_position()
        goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

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

    dt = 0.1  # Assume fixed time step of 0.1 seconds
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = BaseController(dt, wheelbase, map_instance)

    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory, steering_angles, accelations = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Base Controller Path")
        plt.legend(fontsize=14)
        plt.show()

if __name__ == "__main__":
    main()