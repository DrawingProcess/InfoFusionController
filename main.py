import numpy as np
import math
import matplotlib.pyplot as plt

from space.parking_lot import ParkingLot
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar
from control.mpc_adaptive import AdaptiveMPCController
from control.mpc_basic import MPCController

from utils import calculate_angle, transform_arrays_with_angles

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

    # # Generate a reference trajectory using Informed RRT* (simulated here)
    # ref_trajectory = np.array([[start_pose.x + (goal_pose.x - start_pose.x) * t, start_pose.y + (goal_pose.y - start_pose.y) * t, goal_pose.theta, 1.0] for t in np.linspace(0, 1, 50)])

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
    mpc_controller = MPCController(horizon=10, dt=0.1, parking_lot=parking_lot)

    # Follow the trajectory using the MPC controller
    trajectory = mpc_controller.follow_trajectory(start_pose, ref_trajectory)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="MPC Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()