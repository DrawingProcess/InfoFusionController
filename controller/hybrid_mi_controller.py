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
from controller.adaptive_mpc_controller import AdaptiveMPCController
from controller.pure_pursuit_controller import PurePursuitController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

def mutual_information(state1, state2):
    """Calculate Mutual Information between two states."""
    mi_list = []

    min_len = min(len(state1), len(state2))
    state1 = state1[:min_len]
    state2 = state2[:min_len]
    
    for i in range(4):  # [x, y, theta, speed]
        # Compute histograms with density=False
        hist1, _ = np.histogram(state1[:, i], bins=5, density=False)
        hist2, _ = np.histogram(state2[:, i], bins=5, density=False)
        
        # Convert to float and add epsilon to avoid zeros
        hist1 = hist1.astype(float) + 1e-12
        hist2 = hist2.astype(float) + 1e-12
        
        # Normalize histograms
        hist1 /= np.sum(hist1)
        hist2 /= np.sum(hist2)
        print(f"State {i} Histograms: {hist1}, {hist2}")
        
        # Calculate entropy
        entropy1 = entropy(hist1)
        entropy2 = entropy(hist2)
        # print(f"State {i} Entropy: {entropy1}, {entropy2}")
        
        # Compute joint histogram and normalize
        joint_hist, _, _ = np.histogram2d(state1[:, i], state2[:, i], bins=5, density=False)
        joint_hist = joint_hist.astype(float) + 1e-12
        joint_hist /= np.sum(joint_hist)
        joint_entropy = entropy(joint_hist.flatten())
        # print(f"Joint Entropy: {joint_entropy}")
        
        # Calculate Mutual Information
        mi = entropy1 + entropy2 - joint_entropy
        mi_list.append(mi)
    
    return np.array(mi_list)

# 두 알고리즘의 상태를 결합하는 함수
def combine_states(states_pursuit, states_mpc, mi):
    """Mutual Information 기반으로 상태를 결합"""
    combined_state = np.zeros_like(states_pursuit)
    # print(f"State 1: {states_pursuit}, State 2: {states_mpc}")
    
    # Mutual Information을 바탕으로 각 상태에 가중치를 부여하여 결합
    for i in range(4):  # [x, y, theta, velocity]
        weight2 = mi[i] / (mi[i] + 1)  # Mutual Information 비율에 따른 가중치
        weight1 = 1 - weight2
        combined_state[:, i] = weight1 * states_pursuit[:, i] + weight2 * states_mpc[:, i]
        # print(f"mi{i}: {mi[i]}, w1: {weight1}, w2: {weight2}")

    return np.array(combined_state)

class HybridMIController(BaseController):
    def __init__(self, horizon, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.horizon = horizon

        self.mpc_controller = AdaptiveMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
        self.pure_pursuit_controller = PurePursuitController(lookahead_distance=5.0, dt=dt, wheelbase=wheelbase, map_instance=map_instance)

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        print(ref_trajectory)
        # 초기 상태 설정
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        steering_angles = []
        accelations = []

        # Initialize reference index
        ref_index = 0  # Start from the beginning of the trajectory

        # Follow the reference trajectory
        while True:
            if self.is_goal_reached(current_state, goal_position, tolerance=5):
                print("Final adjustment to reach the goal.")
                current_state[0], current_state[1] = goal_position
                current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
                trajectory.append(current_state)
                break

            is_pursuit_reached = True

            # local planning: mpc
            ref_segment, ref_index = self.get_ref_segment(current_state, ref_trajectory, ref_index)

            # global planning
            ref_segment_interpole = transform_trajectory_with_angles(ref_segment[:3], num_points=4, last_segment_factor=1)
            # print(f"global planning: {ref_segment_interpole}")

            # local planning: mpc
            self.mpc_controller.update_horizon(current_state, ref_segment)
            control_input, predicted_states_mpc = self.mpc_controller.optimize_control(current_state, ref_segment)
            # print(f"local planning mpc: {predicted_states_mpc}")

            # local planning: pure_pursuit
            target_state = self.pure_pursuit_controller.find_target_state(current_state, ref_trajectory)

            if not self.is_collision_free(current_state, target_state):
                # Generate possible adjusted targets to avoid obstacles
                adjusted_states = self.avoid_obstacle(current_state, target_state)
                is_pursuit_reached, target_state = self.select_best_path(current_state, adjusted_states, goal_position)
            
            if is_pursuit_reached:
                predicted_states_pursuit = self.predict_trajectory(current_state, target_state, n_steps=self.mpc_controller.horizon)
            # print(f"local planning pure_pursuit: {predicted_states_pursuit}")
            # steering_angle_pure_pursuit = self.pure_pursuit_controller.compute_control(current_state, target_state)
            # next_state_pure_pursuit = self.pure_pursuit_controller.apply_control(current_state, steering_angle_pure_pursuit, velocity=0.5)
            
            if predicted_states_mpc is None and is_pursuit_reached == False:
                print("Error: failed to produce predicted states MPC & pure_pursuit Controller.")
                return False, 0, np.array(trajectory), np.array(steering_angles), np.array(accelations)
            elif predicted_states_mpc is None:
                print("Warning: failed to produce predicted states MPC Controller.")
                next_states = predicted_states_pursuit
            elif is_pursuit_reached == False:
                print("Warning: failed to produce predicted states Pure Pursuit Controller.")
                next_states = predicted_states_mpc
            elif len(predicted_states_mpc) == self.mpc_controller.horizon:
                # 현재 스텝에서 두 경로 간의 Mutual Information 계산
                mi = mutual_information(np.array(predicted_states_pursuit), np.array(predicted_states_mpc))
                next_states = combine_states(np.array(predicted_states_pursuit), np.array(predicted_states_mpc), mi)

            # 현재 상태에서 next_states[0]으로 가기 위한 제어 입력 계산
            control_input = self.compute_control(current_state, next_states[1])
            # next_state = self.apply_control(current_state, control_input)

            accelations.append(control_input[0])
            steering_angles.append(control_input[1])

            current_state = next_states[1]
            trajectory.append(current_state)

            # Plot predicted states and reference segment if desired
            if show_process:
                plt.plot(next_states[:, 0], next_states[:, 1], "b--")
                plt.plot(ref_segment[:, 0], ref_segment[:, 1], "g--")
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

    map_instance.plot_map(title="Path Planning with MPC and Pure Pursuit")
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

    horizon = 10  # MPC horizon
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = HybridMIController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)

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
