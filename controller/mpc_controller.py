import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from controller.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class MPCController(BaseController):
    def __init__(self, horizon, dt, wheelbase, map_instance):
        super().__init__(dt, wheelbase, map_instance)
        self.horizon = horizon

    def compute_cost(self, predicted_states, ref_trajectory):
        cost = 0
        obstacle_avoidance_weight = 200  # 장애물 회피 비용 가중치
        safe_distance = 1.0  # 장애물과의 최소 안전 거리(m)
        max_deviation = 1.0  # 허용되는 최대 이탈 거리(m), 더 작은 값으로 설정
        deviation_weight = 500  # 경로 이탈에 대한 가중치를 크게 설정

        ref_target_state = ref_trajectory[len(predicted_states)-1]
        for i in range(len(predicted_states)):
            if i >= len(ref_trajectory):
                break
            state = predicted_states[i]
            ref_state = ref_trajectory[i]
            cost += np.sum((state - ref_state) ** 2)

            # 경로 이탈 페널티 추가 (더 강력하게 설정)
            deviation = np.linalg.norm(state[:2] - ref_state[:2])
            if deviation > max_deviation:
                cost += deviation_weight * (deviation - max_deviation) ** 2

            # 장애물 회피 비용
            if not self.is_collision_free(state[:2], ref_target_state[:2]):
                obstacle_distance, obstacle_angle = self.map_instance.get_nearest_obstacle_info(state)

                if obstacle_distance is not None and obstacle_distance < safe_distance:
                    cost += obstacle_avoidance_weight * (1 / (obstacle_distance + 1e-6)) * obstacle_angle

        return cost

    def predict_trajectory(self, current_state, control_input, ref_trajectory):
        # Predict future trajectory using the kinematic bicycle model
        predicted_trajectory = []
        state = current_state.copy()
        for _ in range(self.horizon):
            state = self.apply_control(state, control_input)
            predicted_trajectory.append(list(state))

        # predicted states와 reference trajectory 길이를 맞춤
        if len(predicted_trajectory) > len(ref_trajectory):
            predicted_trajectory = predicted_trajectory[:len(ref_trajectory)]
        
        return np.array(predicted_trajectory)
    
    def optimize_control(self, current_state, ref_trajectory):
        best_control = None
        best_predicted_states = None
        min_cost = float('inf')

        predicted_lines = []

        for a_ref in np.linspace(-1, 1, 7):  # 가속도 범위를 세밀하게 설정
            for delta_ref in np.linspace(-np.pi/4, np.pi/4, 11):  # 조향 각도 범위 설정
                predict_trajectory = self.predict_trajectory(current_state, (a_ref, delta_ref), ref_trajectory)
                predicted_lines.append(plt.plot(predict_trajectory[:, 0], predict_trajectory[:, 1], "b-", label="Predicted Path")[0])

                # validate collision-free states
                if all(self.is_collision_free(current_state, s) for s in predict_trajectory):
                    cost = self.compute_cost(predict_trajectory, ref_trajectory)
                    if cost < min_cost:
                        min_cost = cost
                        best_control = (a_ref, delta_ref)
                        best_predicted_states = predict_trajectory

        for line in predicted_lines:
            line.remove()
        
        # best_control이 없을 때, 움직일 수 없음을 알림
        if best_control is None:
            return None, None
        
        # # best_control이 없을 때, 기본 움직임을 설정하고 최소한의 예측 상태 생성
        # if best_control is None:
        #     best_control = (0.1, 0.0)  # 기본값: 천천히 직진
        #     best_predicted_states = []
        #     state = current_state
        #     for _ in range(self.horizon):
        #         state = self.apply_control(state, best_control)
        #         best_predicted_states.append(list(state))

        return best_control, np.array(best_predicted_states)

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

            control_input, predicted_states = self.optimize_control(current_state, ref_segment)

            if control_input is None:
                print("Cannot find valid control input. Vehicle cannot move.")
                is_reached = False  # 움직일 수 없는 경우 False로 설정
                break

            next_state = self.apply_control(current_state, control_input)

            accelations.append(control_input[0])
            steering_angles.append(control_input[1])

            current_state = next_state
            trajectory.append(current_state)

            # Plot predicted states and reference segment if desired
            if show_process:
                plt.plot(predicted_states[:, 0], predicted_states[:, 1], "b--")
                plt.plot(ref_segment[:, 0], ref_segment[:, 1], "g--")
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        total_distance = calculate_trajectory_distance(trajectory)

        return is_reached, total_distance, np.array(trajectory), np.array(steering_angles), np.array(accelations)

def main():
    parser = argparse.ArgumentParser(description="Adaptive MPC Route Planner with configurable map, route planner, and controller.")
    parser.add_argument('--map', type=str, default='fixed_grid', choices=['parking_lot', 'fixed_grid', 'random_grid'], help='Choose the map type.')
    parser.add_argument('--conf', help='Path to configuration JSON file', default=None)
    parser.add_argument('--show_process', action='store_true', help='Show the process of the route planner')
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

    print(f"Start MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 맵과 장애물 및 시작/목표 지점을 표시
    map_instance.plot_map(title="MPC Route Planner")
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
    print(ref_trajectory)

    # Plot Theta* Path
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path 
    plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT* Path")  # Red solid line

    # MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = MPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Follow the trajectory using the MPC controller
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
