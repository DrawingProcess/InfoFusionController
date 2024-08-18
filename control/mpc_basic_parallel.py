import numpy as np
import math
import matplotlib.pyplot as plt
import threading
import queue
import time

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar
from utils import calculate_angle, transform_arrays_with_angles

class MPCController:
    def __init__(self, horizon, dt, map_instance, wheelbase):
        self.horizon = horizon
        self.dt = dt
        self.map_instance = map_instance
        self.wheelbase = wheelbase
        self.current_trajectory = None
        self.trajectory_lock = threading.Lock()

    def predict(self, state, control_input):
        x, y, theta, v = state
        v_ref, delta_ref = control_input

        # Update the state using the kinematic bicycle model
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += v / self.wheelbase * np.tan(delta_ref) * self.dt
        v += v_ref * self.dt

        return np.array([x, y, theta, v])

    def compute_cost(self, predicted_states, ref_trajectory):
        cost = 0
        for i in range(len(predicted_states)):
            state = predicted_states[i]
            ref_state = ref_trajectory[i]
            cost += np.sum((state - ref_state)**2)
        return cost

    def is_collision_free(self, state):
        x, y, _, _ = state
        return self.map_instance.is_not_crossed_obstacle((round(x), round(y)), (round(x), round(y)))

    def optimize_control(self, current_state, ref_trajectory):
        best_control = None
        min_cost = float('inf')

        for v_ref in np.linspace(-1, 1, 5):
            for delta_ref in np.linspace(-np.pi/4, np.pi/4, 5):
                predicted_states = []
                state = current_state
                for i in range(self.horizon):
                    state = self.predict(state, (v_ref, delta_ref))
                    predicted_states.append(state)

                if all(self.is_collision_free(s) for s in predicted_states):
                    cost = self.compute_cost(predicted_states, ref_trajectory)
                    if cost < min_cost:
                        min_cost = cost
                        best_control = (v_ref, delta_ref)

        return best_control

    def apply_control(self, current_state, control_input):
        return self.predict(current_state, control_input)

    def follow_trajectory(self, start_pose, ref_trajectory, plot_queue):
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        for _ in range(len(ref_trajectory) - self.horizon):
            # Acquire the lock to check for an updated trajectory
            with self.trajectory_lock:
                if self.current_trajectory is not None:
                    ref_trajectory = self.current_trajectory
                    self.current_trajectory = None  # Reset after applying the update

            # Extract the reference segment for the current horizon
            ref_segment = ref_trajectory[:self.horizon]
            control_input = self.optimize_control(current_state, ref_segment)
            current_state = self.apply_control(current_state, control_input)
            trajectory.append(current_state)

            # Add the current state to the plot queue
            plot_queue.put(("mpc", current_state))

            time.sleep(0.1)  # Slow down to visualize the movement in real-time

        return np.array(trajectory)

    def update_trajectory(self, new_trajectory):
        # Update the trajectory in a thread-safe manner
        with self.trajectory_lock:
            self.current_trajectory = new_trajectory


def trrt_planning_thread(start_pose, goal_pose, map_instance, mpc_controller, plot_queue):
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)

    while True:
        # Re-plan the route
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)

        if rx_opt and ry_opt:
            # Transform reference trajectory
            ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

            # Update the MPC controller with the new trajectory
            mpc_controller.update_trajectory(ref_trajectory)

            # Plot the generated TRRT* path
            plot_queue.put(("trrt", rx, ry, rx_opt, ry_opt))

        # Sleep or wait for a condition before the next re-plan (e.g., every 5 seconds)
        time.sleep(5)


def plot_mpc_path(plot_queue, obstacle_x, obstacle_y, start_pose, goal_pose, lot_width, lot_height):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.plot(obstacle_x, obstacle_y, ".k")
    ax.plot(start_pose.x, start_pose.y, "og")
    ax.plot(goal_pose.x, goal_pose.y, "xb")
    ax.set_xlim(-1, lot_width + 1)
    ax.set_ylim(-1, lot_height + 1)
    ax.set_title("Adaptive MPC Route Planner")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.set_aspect('equal')

    trrt_path_plot, = ax.plot([], [], "g--", label="TRRT* Path")
    optimized_path_plot, = ax.plot([], [], "r-", label="Optimized Path")
    mpc_path_plot, = ax.plot([], [], "xr", label="MPC Path")
    ax.legend()

    trrt_path_data = ([], [])
    optimized_path_data = ([], [])
    mpc_path_data = ([], [])

    while True:
        # Get the next state or path to plot from the queue
        if not plot_queue.empty():
            data = plot_queue.get()

            if isinstance(data, tuple):
                if data[0] == "trrt":
                    # Update TRRT* path
                    trrt_path_data = (data[1], data[2])
                    optimized_path_data = (data[3], data[4])
                elif data[0] == "mpc":
                    # Update MPC path
                    mpc_path_data[0].append(data[1][0])
                    mpc_path_data[1].append(data[1][1])

            # Redraw all paths
            trrt_path_plot.set_data(trrt_path_data[0], trrt_path_data[1])
            optimized_path_plot.set_data(optimized_path_data[0], optimized_path_data[1])
            mpc_path_plot.set_data(mpc_path_data[0], mpc_path_data[1])

            plt.draw()
            plt.pause(0.001)


def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    obstacle_x = [obstacle[0] for obstacle in map_instance.obstacles]
    obstacle_y = [obstacle[1] for obstacle in map_instance.obstacles]

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Adaptive MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 초기 경로 생성
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)

    if rx_opt and ry_opt:
        ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)
    else:
        print("Initial route generation failed.")
        return

    # MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    mpc_controller = MPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Plotting queue for the main thread to update the plot
    plot_queue = queue.Queue()

    # Create and start the planning thread
    planning_thread = threading.Thread(target=trrt_planning_thread, args=(start_pose, goal_pose, map_instance, mpc_controller, plot_queue))
    planning_thread.daemon = True
    planning_thread.start()

    # Create and start the MPC control thread
    control_thread = threading.Thread(target=mpc_controller.follow_trajectory, args=(start_pose, ref_trajectory, plot_queue))
    control_thread.start()

    # Start the plotting in the main thread
    plot_mpc_path(plot_queue, obstacle_x, obstacle_y, start_pose, goal_pose, map_instance.lot_width, map_instance.lot_height)

    # Wait for the control thread to finish (the planning and plot threads run indefinitely)
    control_thread.join()


if __name__ == "__main__":
    main()
