import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
import argparse

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from controller.base_controller import BaseController
from controller.mpc_controller import MPCController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

def mutual_information(state1, state2):
    """Calculate Mutual Information between two states."""
    mi_list = []

    min_len = min(len(state1), len(state2))
    state1 = state1[:min_len]
    state2 = state2[:min_len]

    for i in range(3):  # [x, y, theta]
        # Compute histograms with density=False
        hist1, _ = np.histogram(state1[:, i], bins=5, density=False)
        hist2, _ = np.histogram(state2[:, i], bins=5, density=False)
        
        # Convert to float and add epsilon to avoid zeros
        hist1 = hist1.astype(float) + 1e-12
        hist2 = hist2.astype(float) + 1e-12
        
        # Normalize histograms
        hist1 /= np.sum(hist1)
        hist2 /= np.sum(hist2)
        
        # Calculate entropy
        entropy1 = entropy(hist1)
        entropy2 = entropy(hist2)
        
        # Compute joint histogram and normalize
        joint_hist, _, _ = np.histogram2d(state1[:, i], state2[:, i], bins=5, density=False)
        joint_hist = joint_hist.astype(float) + 1e-12
        joint_hist /= np.sum(joint_hist)
        joint_entropy = entropy(joint_hist.flatten())
        
        # Calculate Mutual Information
        mi = entropy1 + entropy2 - joint_entropy
        mi_list.append(mi)
    
    return np.array(mi_list)

class MPCMIController(MPCController):
    def __init__(self, horizons, dt, wheelbase, map_instance):
        super().__init__(horizon=max(horizons), dt=dt, wheelbase=wheelbase, map_instance=map_instance)
        self.mpc_controllers = []
        for h in horizons:
            mpc = MPCController(horizon=h, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
            self.mpc_controllers.append(mpc)

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize current state
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        steering_angles = []
        accelations = []

        # Initialize reference index
        ref_index = 0  # Start from the beginning of the trajectory

        # Maximum index in the reference trajectory
        max_ref_index = len(ref_trajectory) - 1

        # Ensure ref_trajectory is a numpy array
        ref_trajectory = np.array(ref_trajectory)

        while True:
            if self.is_goal_reached(current_state, goal_position):
                print("Goal reached successfully!")
                break

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

            # Extract reference segment for this step
            max_horizon = max([mpc.horizon for mpc in self.mpc_controllers])
            ref_segment_end = min(ref_index + max_horizon, max_ref_index + 1)
            ref_segment = ref_trajectory[ref_index:ref_segment_end]

            # If ref_segment is shorter than max_horizon, pad it with the last point
            if len(ref_segment) < max_horizon:
                last_point = ref_segment[-1]
                num_padding = max_horizon - len(ref_segment)
                padding = np.tile(last_point, (num_padding, 1))
                ref_segment = np.vstack((ref_segment, padding))

            mi_values = []
            predicted_states_list = []
            control_inputs = []

            # For each MPC controller
            for mpc in self.mpc_controllers:
                # Local planning: MPC
                control_input, predicted_states_mpc = mpc.optimize_control(current_state, ref_segment)

                if predicted_states_mpc is None or len(predicted_states_mpc) == 0:
                    mi = -np.inf  # Assign negative infinity if failed
                else:
                    # Adjust the length of predicted_states_mpc and ref_segment to be equal
                    min_len = min(len(predicted_states_mpc), len(ref_segment))
                    predicted_states_mpc = np.array(predicted_states_mpc)[:min_len]
                    ref_segment_trimmed = ref_segment[:min_len]
                    mi = mutual_information(predicted_states_mpc, ref_segment_trimmed)
                    mi = np.sum(mi)  # Sum the mutual information over the state variables

                    if show_process:
                        plt.plot(predicted_states_mpc[:, 0], predicted_states_mpc[:, 1], "b--")
                        plt.plot(ref_segment_trimmed[:, 0], ref_segment_trimmed[:, 1], "g--")
                mi_values.append(mi)
                predicted_states_list.append(predicted_states_mpc)
                control_inputs.append(control_input)

            # Select the MPC with highest mutual information
            best_mi_index = np.argmax(mi_values)
            best_mpc = self.mpc_controllers[best_mi_index]
            best_predicted_states = predicted_states_list[best_mi_index]
            best_control_input = control_inputs[best_mi_index]

            if best_predicted_states is None or len(best_predicted_states) == 0:
                print("Error: All MPC controllers failed to produce predicted states.")
                return False, 0, np.array(trajectory)
            else:
                # Apply control and handle obstacle avoidance
                next_state = self.apply_control(current_state, best_control_input)

                # 현재 상태에서 next_states[0]으로 가기 위한 제어 입력 계산
                control_input = self.compute_control(current_state, next_state)

                # 스티어링 각도와 속도 저장
                accelations.append(control_input[0])
                steering_angles.append(control_input[1])

                current_state = next_state
                trajectory.append(current_state)

            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        total_distance = calculate_trajectory_distance(trajectory)

        print("Trajectory following completed.")
        return True, total_distance, np.array(trajectory), np.array(steering_angles), np.array(accelations)

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

    map_instance.plot_map(title="Path Planning with MPC")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    route_planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    if not isReached:
        print("TRRT* was unable to generate a valid path.")
        return

    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)

    # Plot Optimized Path 
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    horizons = [5, 10, 15]  # Different MPC horizons
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = MPCMIController(horizons=horizons, dt=dt, wheelbase=wheelbase, map_instance=map_instance)

    goal_position = [goal_pose.x, goal_pose.y]
    is_reached, trajectory_distance, trajectory  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Selected MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
