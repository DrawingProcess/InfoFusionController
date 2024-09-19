import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, calculate_trajectory_distance, transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from controller.mpc_controller import MPCController
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class AdaptiveMPCController(MPCController):
    def __init__(self, horizon, dt, wheelbase, map_instance):
        super().__init__(horizon, dt, wheelbase, map_instance)
        self.previous_control = None

    def update_horizon(self, current_state, ref_trajectory):
        # Update horizon dynamically based on current state or trajectory deviation
        if np.linalg.norm(current_state[:2] - ref_trajectory[0][:2]) > 5:
            self.horizon = min(self.horizon + 1, 20)
        else:
            self.horizon = max(self.horizon - 1, 5)

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        is_reached = True

        # Follow the reference trajectory
        for i in range(len(ref_trajectory)):
            if self.is_goal_reached(current_state, goal_position):
                print("Goal reached successfully!")
                break

            self.horizon = min(self.horizon, len(ref_trajectory) - i)

            ref_segment = ref_trajectory[i:i + self.horizon]
            self.update_horizon(current_state, ref_segment)
            control_input = self.optimize_control(current_state, ref_segment)
            next_state = self.apply_control(current_state, control_input)

            adjusted_states = self.avoid_obstacle(current_state, next_state)
            if not self.is_collision_free(current_state, next_state):
                print(f"Collision detected at step {i}. Attempting to avoid obstacle...")
                is_reached, next_state = self.select_best_path(current_state, adjusted_states, goal_position)
                if not is_reached:
                    print("Goal not reachable.")
                    return is_reached, 0, np.array(trajectory)

            current_state = next_state
            trajectory.append(current_state)

            # Plot current state
            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        # If the goal is still not reached, adjust the final position
        if not self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        total_distance = calculate_trajectory_distance(trajectory)

        print("Trajectory following completed.")
        return True, total_distance, np.array(trajectory)


def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Adaptive MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 맵과 장애물 및 시작/목표 지점을 표시
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
    is_reached, trajectory_distance, trajectory  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
