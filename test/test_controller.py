import argparse
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from utils import transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap
from map.slam_grid_map import SlamGridMap

from route_planner.geometry import Pose
from route_planner.informed_trrt_star_planner import InformedTRRTStar

from controller.mpc_controller import MPCController
from controller.adaptive_mpc_controller import AdaptiveMPCController
from controller.pure_pursuit_controller import PurePursuitController
from controller.mpc_mi_controller import MPCMIController
from controller.weighted_fusion_controller import WeightedFusionController
from controller.info_fusion_controller import InfoFusionController
from controller.multi_purpose_mpc_controller import MultiPurposeMPCController
from controller.stanley_controller import StanleyController

def main():
    parser = argparse.ArgumentParser(description="Controller Speed Test with Informed TRRT* Route Planner.")
    parser.add_argument('--map', type=str, default='fixed_grid', choices=['parking_lot', 'fixed_grid', 'random_grid', 'image_grid'], help='Choose the map type.')
    parser.add_argument('--conf', help='Path to configuration JSON file', default=None)
    parser.add_argument('--show_process', action='store_true', help='Show the process of the route planner')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic obstacles in the map')
    parser.add_argument('--output_dir', type=str, default='results/test_controller', help='Directory to save outputs')
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.conf:
        # Read the JSON file and extract parameters
        with open(args.conf, 'r') as f:
            config = json.load(f)

        start_pose = Pose(config['start_pose'][0], config['start_pose'][1], config['start_pose'][2])
        goal_pose = Pose(config['goal_pose'][0], config['goal_pose'][1], config['goal_pose'][2])
        width = config.get('width', 50)
        height = config.get('height', 50)
        obstacles = config.get('obstacles', [])
        obstacles_dynamic = config.get('obstacles_dynamic', [])
    else:
        # Use default parameters
        width = 50
        height = 50
        start_pose = Pose(2, 2, 0)
        goal_pose = Pose(width - 5, height - 5, 0)
        obstacles = None  # Will trigger default obstacles in the class
        obstacles_dynamic = []
        config = {
            'start_pose': [start_pose.x, start_pose.y, start_pose.theta],
            'goal_pose': [goal_pose.x, goal_pose.y, goal_pose.theta],
            'width': width,
            'height': height,
            'obstacles': obstacles,
            'obstacles_dynamic': obstacles_dynamic
        }

    # Map selection using dictionary
    map_options = {
        'parking_lot': ParkingLot,
        'fixed_grid': FixedGridMap,
        'random_grid': RandomGridMap,
        'image_grid': SlamGridMap
    }
    if args.map == "image_grid":
        map_instance = SlamGridMap(image_path='./map/fig/map_slam.png', obstacles=obstacles)
    else:
        map_instance = map_options[args.map](width, height, obstacles)

    if args.map == "random_grid":
        start_pose = map_instance.get_random_valid_start_position()
        goal_pose = map_instance.get_random_valid_goal_position()
        config['start_pose'] = [start_pose.x, start_pose.y, start_pose.theta]
        config['goal_pose'] = [goal_pose.x, goal_pose.y, goal_pose.theta]
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # Control show_process and show_eclipse with the show_process variable
    show_process = args.show_process

    num_trajectories = 2

    # Initialize route trajectories
    if 'route_trajectory' in config:
        # Load route trajectories from config
        route_trajectory_opts = [np.array(opt) for opt in config['route_trajectory']]
        route_trajectorys = route_trajectory_opts  # Assuming route_trajectorys are similar
        num_trajectories = len(route_trajectory_opts)
        print("Loaded route_trajectory from config.")
    else:
        route_trajectorys, route_trajectory_opts = [], []
        count = 0

        while (count < num_trajectories):
            # Informed TRRT* Planner
            planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)

            isReached, total_distance, route_trajectory, route_trajectory_opt = planner.search_route(show_process=show_process)
            if not isReached:
                continue

            plt.clf()
            map_instance.plot_map(title=f"Informed TRRT* Route Planner [{count}]")
            plt.plot(start_pose.x, start_pose.y, "og")
            plt.plot(goal_pose.x, goal_pose.y, "xb")
            plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT Path")  # Red solid line
            plt.legend(loc="upper left", fontsize=14)
            plt.savefig(os.path.join(args.output_dir, f"route_{count}.png"))

            route_trajectorys.append(route_trajectory)
            route_trajectory_opts.append(route_trajectory_opt.tolist())  # Convert to list for JSON serialization
            count += 1

        config['route_trajectory'] = route_trajectory_opts  # Save to config

    # Handle dynamic obstacles
    if args.dynamic:
        if obstacles_dynamic:
            map_instance.add_config_obstacles(obstacles_dynamic, is_dynamic=True)
            print("Loaded obstacles_dynamic from config.")
        else:
            # Generate dynamic obstacles and save them to config
            obstacles_dynamic = map_instance.create_random_obstacles_in_path(
                np.array(route_trajectory_opts[0]), n=2, box_size=(4, 4))
            config['obstacles_dynamic'] = obstacles_dynamic
            print("Generated and saved obstacles_dynamic to config.")

    # Plot the map with obstacles and routes
    map_instance.plot_map(title=f"Map")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.plot(np.array(route_trajectory_opts[0])[:, 0], np.array(route_trajectory_opts[0])[:, 1], "-r", label="Informed TRRT Path")  # Red solid line
    plt.legend(loc="upper left", fontsize=14)
    plt.savefig(os.path.join(args.output_dir, "route_trajectory.png"))

    # Save updated configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Controller selection using dictionary
    horizon = 10  # MPC horizon
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    goal_position = [goal_pose.x, goal_pose.y]
    algorithms = {
        'adaptive_mpc': lambda: AdaptiveMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
            .follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'mpc_basic': lambda: MPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
            .follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'pure_pursuit': lambda: PurePursuitController(lookahead_distance=5.0, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
            .follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'weighted_fusion': lambda: WeightedFusionController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
            .follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'info_fusion': lambda: InfoFusionController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance)
            .follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        # Add or remove controllers as needed
    }

    # Performance measurement and failure counts
    performance_results = {}
    distance_results = {}
    fail_counts = {name: 0 for name in algorithms}
    trajectory_data = {name: {} for name in algorithms}
    steering_angles_data = {name: {} for name in algorithms}
    accelations_data = {name: {} for name in algorithms}

    for name, func in algorithms.items():
        count = 0
        total_time = 0
        total_dist = 0
        while (count < num_trajectories):
            plt.clf()
            if show_process:
                map_instance.plot_map(title=f"{name} Controller [{count}]")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")
                plt.plot(np.array(route_trajectory_opts[count])[:, 0], np.array(route_trajectory_opts[count])[:, 1], "-r", label="Informed TRRT Path")  # Red solid line

            # Use the precomputed route
            ref_trajectory = transform_trajectory_with_angles(np.array(route_trajectory_opts[count]))
            start_time = time.time()
            is_reached, trajectory_distance, trajectory, steering_angles, accelations = func()
            end_time = time.time()
            control_time = end_time - start_time

            if is_reached:
                total_time += control_time
                total_dist += trajectory_distance

                trajectory_data[name][count] = trajectory
                steering_angles_data[name][count] = steering_angles
                accelations_data[name][count] = accelations
            else:
                fail_counts[name] += 1

            plt.clf()
            map_instance.plot_map(title=f"{name} Controller [{count}]")
            plt.plot(start_pose.x, start_pose.y, "og")
            plt.plot(goal_pose.x, goal_pose.y, "xb")
            plt.plot(np.array(route_trajectory_opts[count])[:, 0], np.array(route_trajectory_opts[count])[:, 1], "-r", label="Informed TRRT Path")  # Red solid line
            plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label=name)
            plt.legend(loc="upper left", fontsize=14)
            plt.savefig(os.path.join(args.output_dir, f"controller_{name}_{count}.png"))
            
            count += 1

        successful_runs = num_trajectories - fail_counts[name]
        if successful_runs != 0:
            performance_results[name] = total_time / successful_runs  # Average execution time
            distance_results[name] = total_dist / successful_runs
            print(f"{name}: {performance_results[name]:.6f} seconds (average)")
            print(f"{name}: {distance_results[name]:.6f} meters (average)")

    for name, time_taken in performance_results.items():
        print(f"{name}: {time_taken:.6f} seconds (average)")
    for name, dist in distance_results.items():
        print(f"{name}: {dist:.6f} meters (average)")
    
    # Ensure performance directory exists
    performance_dir = os.path.join(args.output_dir, "performance")
    os.makedirs(performance_dir, exist_ok=True)

    # Plot the two charts side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns

    # Failure Counts Plot
    algorithm_names = list(fail_counts.keys())
    fail_values = list(fail_counts.values())
    ax1.barh(algorithm_names, fail_values, color='red')
    ax1.set_xlabel("Fail Count")
    ax1.set_ylabel("Algorithm")
    ax1.set_title(f"Algorithm Pathfinding Failure Counts ({num_trajectories} Runs)")
    ax1.grid(True)

    # performance results graph
    algorithm_names = list(performance_results.keys())
    times = list(performance_results.values())
    ax2.barh(algorithm_names, times, color='skyblue')
    ax2.set_xlabel("Average Execution Time (seconds)")
    ax2.set_title(f"Algorithm Performance Comparison ({num_trajectories} Runs)")
    ax2.grid(True)

    # average trajectory distance graph
    algorithm_names = list(distance_results.keys())
    dists = list(distance_results.values())
    ax3.barh(algorithm_names, dists, color='purple')
    ax3.set_xlabel("Average Trajectory Distance (m)")
    ax3.set_title(f"Algorithm Trajectory Distance Comparison ({num_trajectories} Runs)")
    ax3.grid(True)

    # Save combined results to txt file
    with open(os.path.join(args.output_dir, "combined_results.txt"), "w") as f:
        f.write("Algorithm\tFail Count\tAverage Execution Time (seconds)\tAverage Trajectory Distance (m)\n")
        for name in fail_counts.keys():
            fail_count = fail_counts.get(name, 0)
            time_taken = performance_results.get(name, 0)
            dist = distance_results.get(name, 0)
            f.write(f"{name}\t{fail_count}\t{time_taken:.6f}\t{dist:.6f}\n")
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(os.path.join(performance_dir, "performance_controller.png"))
    
    # Save Compare Controller (Trajectory)
    for i in range(num_trajectories):
        plt.figure()
        map_instance.plot_map(title=f"Compare Controller Trajectory")
        plt.plot(start_pose.x, start_pose.y, "og")
        plt.plot(goal_pose.x, goal_pose.y, "xb")
        plt.plot(np.array(route_trajectory_opts[i])[:, 0], np.array(route_trajectory_opts[i])[:, 1], "-r", label="Informed TRRT Path")
        colors = ["g--", "b--", "c--", "y--", "m-", "k-", "r-"]
        for j, name in enumerate(algorithms.keys()):
            if trajectory_data[name].get(i) is not None:
                plt.plot(trajectory_data[name][i][:, 0], trajectory_data[name][i][:, 1], colors[j], label=name)
                # Save trajectory data to txt files
                trajectory = trajectory_data[name][i]
                np.savetxt(os.path.join(args.output_dir, f"trajectory_{name}_{i}.txt"), trajectory, header="x\ty", comments='')
    
        plt.legend(loc="upper left", fontsize=14)
        plt.savefig(os.path.join(args.output_dir, f"compare_controller_trajectory_{i}.png"))
        plt.close()
    
    # Save Compare Controller (Steering Angle)
    for i in range(num_trajectories):
        plt.figure()
        colors = ["g--", "b--", "c--", "y--", "m-", "k-", "r-"]
        for j, name in enumerate(algorithms.keys()):
            if steering_angles_data[name].get(i) is not None:
                accelations = accelations_data[name][i]
                # Calculate distance
                velocity = 0
                dist = np.zeros(len(accelations))
                for t in range(1, len(accelations)):
                    velocity += accelations[t - 1] * dt
                    dist[t] = dist[t - 1] + velocity * dt
                dist = dist * 100 / dist[-1]  # Convert to percentage
    
                steering_angles = steering_angles_data[name][i]
                plt.plot(dist, np.degrees(steering_angles), colors[j], label=name)
    
                # Save steering angles to txt files
                data = np.column_stack((dist, np.degrees(steering_angles)))
                np.savetxt(os.path.join(performance_dir, f"steering_angles_{name}_{i}.txt"), data, header="Progress (%)\tSteering Angle (degrees)", comments='')
    
        plt.legend(loc="upper left", fontsize=14)
        plt.title(f"Compare Controller Steering Angle", fontsize=20)
        plt.xlabel("Progress (%)", fontsize=16)
        plt.ylabel("Steering Angle (degree)", fontsize=16)
        plt.savefig(os.path.join(performance_dir, f"compare_controller_steering_angle_{i}.png"))
        plt.close()
    
    # Save Compare Controller (Acceleration)
    for i in range(num_trajectories):
        plt.figure()
        colors = ["g--", "b--", "c--", "y--", "m-", "k-", "r-"]
        for j, name in enumerate(algorithms.keys()):
            if accelations_data[name].get(i) is not None:
                accelations = accelations_data[name][i]
                # Calculate distance
                velocity = 0
                dist = np.zeros(len(accelations))
                for t in range(1, len(accelations)):
                    velocity += accelations[t - 1] * dt
                    dist[t] = dist[t - 1] + velocity * dt
                dist = dist * 100 / dist[-1]  # Convert to percentage
    
                plt.plot(dist, accelations, colors[j], label=name)
    
                # Save accelerations to txt files
                data = np.column_stack((dist, accelations))
                np.savetxt(os.path.join(performance_dir, f"accelerations_{name}_{i}.txt"), data, header="Progress (%)\tAcceleration (m/s^2)", comments='')
    
        plt.legend(loc="upper left", fontsize=14)
        plt.title(f"Compare Controller Acceleration", fontsize=20)
        plt.xlabel("Progress (%)", fontsize=16)
        plt.ylabel("Acceleration (m/s^2)", fontsize=16)
        plt.savefig(os.path.join(performance_dir, f"compare_controller_acceleration_{i}.png"))
        plt.close()
    
    # Steering angle histograms
    for i in range(num_trajectories):
        for name in algorithms.keys():
            if steering_angles_data[name].get(i) is not None:
                plt.figure()
                angles = np.array(steering_angles_data[name][i]).astype(float)
                plt.hist(angles, bins=30, alpha=0.7, color='green')
                plt.title(f"Steering Angle Histogram: {name}", fontsize=20)
                plt.xlabel("Steering Angle (radians)", fontsize=16)
                plt.ylabel("Frequency", fontsize=16)
                plt.savefig(os.path.join(performance_dir, f"steering_histogram_{name}_{i}.png"))
                plt.close()
    
                # Save histogram data to txt file
                counts, bin_edges = np.histogram(angles, bins=30)
                histogram_data = np.column_stack((bin_edges[:-1], counts))
                np.savetxt(os.path.join(performance_dir, f"steering_histogram_{name}_{i}.txt"), histogram_data, header="Steering Angle (radians)\tFrequency", comments='')
    
    # Steering angle over time plots
    for i in range(num_trajectories):
        for name in algorithms.keys():
            if steering_angles_data[name].get(i) is not None:
                plt.figure()
                accelations = accelations_data[name][i]
                # Calculate time steps
                time_steps = np.arange(len(accelations)) * dt
    
                steering_angles = steering_angles_data[name][i]
                plt.plot(time_steps, np.degrees(steering_angles), label='Steering Angle')
    
                # Save steering angles over time to txt file
                data = np.column_stack((time_steps, np.degrees(steering_angles)))
                np.savetxt(os.path.join(performance_dir, f"steering_angle_over_time_{name}_{i}.txt"), data, header="Time (s)\tSteering Angle (degrees)", comments='')
    
                plt.title(f"Steering Angle over Time: {name}", fontsize=20)
                plt.xlabel("Time (s)", fontsize=16)
                plt.ylabel("Steering Angle (degree)", fontsize=16)
                plt.legend(fontsize=14)
                plt.savefig(os.path.join(performance_dir, f"steering_angle_over_time_{name}_{i}.png"))
                plt.close()
    
    # Acceleration over time plots
    for i in range(num_trajectories):
        for name in algorithms.keys():
            if accelations_data[name].get(i) is not None:
                plt.figure()
                accelations = accelations_data[name][i]
                time_steps = np.arange(len(accelations)) * dt
    
                plt.plot(time_steps, accelations, label='Acceleration')
    
                # Save acceleration over time to txt file
                data = np.column_stack((time_steps, accelations))
                np.savetxt(os.path.join(performance_dir, f"acceleration_over_time_{name}_{i}.txt"), data, header="Time (s)\tAcceleration (m/s^2)", comments='')
    
                plt.title(f"Acceleration over Time: {name}", fontsize=20)
                plt.xlabel("Time (s)", fontsize=16)
                plt.ylabel("Acceleration (m/s^2)", fontsize=16)
                plt.legend(fontsize=14)
                plt.savefig(os.path.join(performance_dir, f"acceleration_over_time_{name}_{i}.png"))
                plt.close()
    

if __name__ == "__main__":
    main()
