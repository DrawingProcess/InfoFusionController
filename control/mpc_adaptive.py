import numpy as np
import math
import matplotlib.pyplot as plt

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from control.mpc_basic import MPCController
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar
from utils import calculate_angle, transform_arrays_with_angles

class AdaptiveMPCController(MPCController):
    def __init__(self, horizon, dt, map_instance, wheelbase):
        super().__init__(horizon, dt, map_instance, wheelbase)
        self.previous_control = None

    def update_horizon(self, current_state, ref_trajectory):
        # Update horizon dynamically based on current state or trajectory deviation
        if np.linalg.norm(current_state[:2] - ref_trajectory[0][:2]) > 5:
            self.horizon = min(self.horizon + 1, 20)
        else:
            self.horizon = max(self.horizon - 1, 5)

    def follow_trajectory(self, start_pose, ref_trajectory):
        """
        Follow the reference trajectory using the Adaptive MPC controller.

        Parameters:
        - start_pose: The starting pose (Pose object).
        - ref_trajectory: The reference trajectory (numpy array).

        Returns:
        - trajectory: The trajectory followed by the Adaptive MPC controller (numpy array).
        """
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        # Follow the reference trajectory
        for i in range(len(ref_trajectory)):
            remaining_points = len(ref_trajectory) - i
            if remaining_points < self.horizon:
                # Dynamically reduce the horizon as we approach the end of the path
                self.horizon = remaining_points

            ref_segment = ref_trajectory[i:i + self.horizon]
            if len(ref_segment) < self.horizon:
                break
            self.update_horizon(current_state, ref_segment)
            control_input = self.optimize_control(current_state, ref_segment)
            current_state = self.apply_control(current_state, control_input)
            trajectory.append(current_state)

            # Stop if close enough to the goal
            if np.linalg.norm(current_state[:2] - ref_trajectory[-1][:2]) < 2.0:
                print("Reached near the goal")
                break

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

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Adaptive MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 맵과 장애물 및 시작/목표 지점을 표시
    map_instance.plot_map()
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Adaptive MPC Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    # Create Informed TRRT* planner
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    
    # Ensure the route generation is completed
    try:
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
        if len(rx_opt) == 0 or len(ry_opt) == 0:
            print("TRRT* was unable to generate a valid path.")
            return

    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)
    
    # Check if the transformed trajectory is valid
    if ref_trajectory.ndim != 2 or ref_trajectory.shape[0] < 2:
        print("Invalid reference trajectory generated.")
        return

    # Plot Theta* Path
    plt.plot(rx, ry, "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path
    plt.plot(rx_opt, ry_opt, "-r", label="Informed TRRT* Path")  # Red solid line

    # Adaptive MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    adaptive_mpc = AdaptiveMPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Follow the trajectory using the Adaptive MPC controller
    trajectory = adaptive_mpc.follow_trajectory(start_pose, ref_trajectory)

    # Plot the MPC Path
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Adaptive MPC Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
