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

from controller.mpc_controller import MPCController

class AdaptiveMPCController(MPCController):
    def __init__(self, horizon, dt, wheelbase, map_instance):
        super().__init__(horizon, dt, wheelbase, map_instance)
        self.previous_control = None

    def update_horizon(self, current_state, ref_trajectory):
        # Update horizon dynamically based on deviation
        deviation = np.linalg.norm(current_state[:2] - ref_trajectory[0][:2])
        if deviation > 5:
            self.horizon = min(self.horizon + 1, 20)
        else:
            self.horizon = max(self.horizon - 1, 5)

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        steering_angles = []
        accelations = []

        is_reached = True
        ref_index = 0  # Start from the beginning of the trajectory

        # Follow the reference trajectory
        while True:
            if self.is_goal_reached(current_state, goal_position):
                print("Final adjustment to reach the goal.")
                current_state[0], current_state[1] = goal_position
                current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
                trajectory.append(current_state)
                break
            
            ref_segment, ref_index = self.get_ref_segment(current_state, ref_trajectory, ref_index)

            self.update_horizon(current_state, ref_segment)

            control_input, predicted_states = self.optimize_control(current_state, ref_segment)

            if control_input is None:
                print("Cannot find valid control input. Vehicle cannot move.")
                is_reached = False
                break

            next_state = self.apply_control(current_state, control_input)

            accelations.append(control_input[0])
            steering_angles.append(control_input[1])

            current_state = next_state
            trajectory.append(current_state)

            # Plot current state and predicted path
            if show_process:
                plt.plot(predicted_states[:, 0], predicted_states[:, 1], "b--")
                plt.plot(ref_segment[:, 0], ref_segment[:, 1], "g--")
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        # If the goal almost reached, adjust the final position
        if self.is_goal_reached(current_state, goal_position, tolerance=5):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        total_distance = calculate_trajectory_distance(trajectory)

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

    # Plot the map and start/goal positions
    map_instance.plot_map(title="Adaptive MPC Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
 
    # Create Informed TRRT* planner
    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    
    # Ensure the route generation is completed
    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
        if not isReached:
            print("TRRT* was unable to generate a valid path.")
            return

    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)
    
    # Check if the transformed trajectory is valid
    if ref_trajectory.ndim != 2 or ref_trajectory.shape[0] < 2:
        print("Invalid reference trajectory generated.")
        return

    # Plot Theta* Path
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    # Adaptive MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = AdaptiveMPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Follow the trajectory using the Adaptive MPC controller
    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory, steering_angles, accelations  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
