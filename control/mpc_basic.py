import numpy as np
import math
import matplotlib.pyplot as plt
from space.parking_lot import ParkingLot
from adlab_planning.route_planner.informed_rrt_star_smooth_planner import Pose, InformedTRRTStar
from utils import calculate_angle, transform_arrays_with_angles
class MPCController:
    def __init__(self, horizon, dt, parking_lot, wheelbase):
        self.horizon = horizon
        self.dt = dt
        self.parking_lot = parking_lot
        self.wheelbase = wheelbase  # Wheelbase of the vehicle

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
        return self.parking_lot.is_not_crossed_obstacle((round(x), round(y)), (round(x), round(y)))

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
    
    def follow_trajectory(self, start_pose, ref_trajectory):
        """
        Follow the reference trajectory using the MPC controller.
        
        Parameters:
        - start_pose: The starting pose (Pose object).
        - ref_trajectory: The reference trajectory (numpy array).
        
        Returns:
        - trajectory: The trajectory followed by the MPC controller (numpy array).
        """
        # Initialize the state and trajectory
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        # Follow the reference trajectory
        for i in range(len(ref_trajectory) - self.horizon):
            ref_segment = ref_trajectory[i:i + self.horizon]
            control_input = self.optimize_control(current_state, ref_segment)
            current_state = self.apply_control(current_state, control_input)
            trajectory.append(current_state)

            # Plot current state
            plt.plot(current_state[0], current_state[1], "xr")
            plt.pause(0.001)

        return np.array(trajectory)


def main():
    parking_lot = ParkingLot()
    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")

    # Start and goal pose
    start_pose = Pose(14.0, 4.0, np.radians(0))
    goal_pose = Pose(50.0, 38.0, np.radians(90))
    print(f"Start Adaptive MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, parking_lot.lot_width + 1)
    plt.ylim(-1, parking_lot.lot_height + 1)
    plt.title("Adaptive MPC Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    # Create Informed TRRT* planner
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, parking_lot, show_eclipse=False)
    rx_rrt, ry_rrt, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)

    # Ensure the route generation is completed
    try:
        rx_rrt, ry_rrt, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

    # Plot Informed RRT* Path
    plt.plot(rx_rrt, ry_rrt, "g--", label="Informed RRT* Path")  # Green dashed line

    # Plot Optimized Path
    plt.plot(rx_opt, ry_opt, "-r", label="Optimized Path")  # Red solid line

    # MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    mpc_controller = MPCController(horizon=10, dt=0.1, parking_lot=parking_lot, wheelbase=wheelbase)

    # Follow the trajectory using the MPC controller
    trajectory = mpc_controller.follow_trajectory(start_pose, ref_trajectory)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="MPC Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
