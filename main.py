import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from route_planner.geometry import Pose
from route_planner.a_star_route_planner import AStarRoutePlanner
from route_planner.hybrid_a_star_route_planner import HybridAStarRoutePlanner
from route_planner.theta_star_planner import ThetaStar
from route_planner.rrt_star_planner import RRTStar
from route_planner.informed_rrt_star_planner import InformedRRTStar
from route_planner.informed_rrt_star_smooth_planner import InformedRRTSmoothStar
from route_planner.informed_trrt_star_planner import InformedTRRTStar

from controller.pure_pursuit_controller import PurePursuitController
from controller.stanley_controller import StanleyController
from controller.mpc_controller import MPCController
from controller.adaptive_mpc_controller import AdaptiveMPCController
from controller.multi_purpose_mpc_controller import MultiPurposeMPCController

from utils import calculate_angle, transform_trajectory_with_angles

def main():
    parser = argparse.ArgumentParser(description="Adaptive MPC Route Planner with configurable map, route planner, and controller.")
    parser.add_argument('--map', type=str, default='random_grid', choices=['parking_lot', 'fixed_grid', 'random_grid'], help='Choose the map type.')
    parser.add_argument('--route_planner', type=str, default='informed_trrt_star', choices=[
        'a_star', 'hybrid_a_star', 'theta_star', 'rrt_star', 'informed_rrt_star', 'informed_rrt_smooth_star', 'informed_trrt_star'
    ], help='Choose the route planner.')
    parser.add_argument('--controller', type=str, default='mpc_basic', choices=['pure_pursuit', 'stanley', 'mpc_basic', 'adaptive_mpc'], help='Choose the controller.')
    args = parser.parse_args()

    # Map selection using dictionary
    map_options = {
        'parking_lot': ParkingLot,
        'fixed_grid': FixedGridMap,
        'random_grid': RandomGridMap
    }
    map_instance = map_options[args.map]()

    if args.map == "parking_lot":
        start_pose = Pose(14.0, 4.0, math.radians(0))
        goal_pose = Pose(50.0, 38.0, math.radians(90))
    elif args.map == "fixed_grid":
        start_pose = Pose(3, 5, math.radians(0))
        goal_pose = Pose(5, 15, math.radians(0))
    else:
        start_pose = map_instance.get_random_valid_start_position()
        goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # Route planner selection using dictionary
    route_planner_options = {
        'a_star': AStarRoutePlanner,
        'hybrid_a_star': HybridAStarRoutePlanner,
        'theta_star': ThetaStar,
        'rrt_star': RRTStar,
        'informed_rrt_star': InformedRRTStar,
        'informed_rrt_smooth_star': InformedRRTSmoothStar,
        'informed_trrt_star': InformedTRRTStar
    }
    route_planner = route_planner_options[args.route_planner](start_pose, goal_pose, map_instance)

    # Controller selection using dictionary
    horizon = 10  # MPC horizon
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller_options = {
        'mpc_basic': MPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance),
        'adaptive_mpc': AdaptiveMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance),
        'multi_purpose_mpc': MultiPurposeMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance),
        'pure_pursuit': PurePursuitController(lookahead_distance=5.0, dt=dt, wheelbase=wheelbase, map_instance=map_instance),
        'stanley': StanleyController(k=0.1, dt=dt, wheelbase=wheelbase, map_instance=map_instance),
    }
    controller = controller_options[args.controller]

    map_instance.plot_map(f"{args.controller.capitalize()} Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

    # Generate the route
    try:
        isReached, total_distance, route_trajectory, route_trajectory_opt = route_planner.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_trajectory_with_angles(route_trajectory_opt)

    # Plot the route
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label=f"{args.route_planner.replace('_', ' ').title()} Path")
    plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "-r", label="Optimized Path")

    # Follow the trajectory using the selected controller
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
