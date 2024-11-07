import argparse
import time
import json
import matplotlib.pyplot as plt
import os

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap
from map.slam_grid_map import SlamGridMap

from route_planner.geometry import Pose
from route_planner.a_star_route_planner import AStarRoutePlanner
from route_planner.hybrid_a_star_route_planner import HybridAStarRoutePlanner
from route_planner.theta_star_planner import ThetaStar
from route_planner.rrt_star_planner import RRTStar
from route_planner.informed_rrt_star_planner import InformedRRTStar
from route_planner.informed_rrt_star_smooth_planner import InformedRRTSmoothStar
from route_planner.informed_trrt_star_planner import InformedTRRTStar

def main():
    parser = argparse.ArgumentParser(description="Adaptive MPC Route Planner with configurable map, route planner, and controller.")
    parser.add_argument('--map', type=str, default='fixed_grid', choices=['parking_lot', 'fixed_grid', 'random_grid', 'image_grid'], help='Choose the map type.')
    parser.add_argument('--conf', help='Path to configuration JSON file', default=None)
    parser.add_argument('--show_process', action='store_true', help='Show the process of the route planner')
    parser.add_argument('--output_dir', type=str, default='results/test_route_planner', help='Directory to save outputs')
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
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # Control show_process and show_eclipse with the show_process variable
    show_process = args.show_process

    num_trajectories = 5

    # Route planner selection using dictionary
    algorithms = {
        "A*": lambda: AStarRoutePlanner(start_pose, goal_pose, map_instance).search_route(show_process),
        "Theta*": lambda: ThetaStar(start_pose, goal_pose, map_instance).search_route(show_process),
        "HybridA*": lambda: HybridAStarRoutePlanner(start_pose, goal_pose, map_instance).search_route(show_process),
        "RRT*": lambda: RRTStar(start_pose, goal_pose, map_instance).search_route(show_process),
        "InformedRRT*": lambda: InformedRRTStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
        "InformedSmoothingRRT*": lambda: InformedRRTSmoothStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
        "InformedTRRT*": lambda: InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
    }
    config["route_trajectory"] = {name: {} for name in algorithms}

    # performance results
    performance_results = {}
    distance_results = {}
    fail_counts = {name: 0 for name in algorithms}
    trajectory_data = {name: {} for name in algorithms}

    for name, func in algorithms.items():
        count = 0
        total_time = 0.0
        total_dist = 0.0
        while (count < num_trajectories):
            plt.clf()  # Clear the plot
            if show_process:
                map_instance.plot_map(title=f"{name} Route Planner")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")
            
            start_time = time.time()  # start time
            result = func()
            end_time = time.time()    # finish time
            time_taken = end_time - start_time  # Calculate time taken

            if not result[0]:  # if not isReached
                fail_counts[name] += 1
            else:
                total_time += time_taken
                total_dist += result[1]

            if result[0]:
                plt.clf()  
                map_instance.plot_map(title=f"{name} Route Planner")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")

                if len(result) == 3:
                    trajectory_data[name][count] = result[2]
                    config['route_trajectory'][name][count] = result[2].tolist()
                    plt.plot(result[2][:, 0], result[2][:, 1], "-r", label=name)  # Green dashed line
                else: # len(result) == 4:
                    trajectory_data[name][count] = result[3]
                    config['route_trajectory'][name][count] = result[3].tolist()
                    plt.plot(result[3][:, 0], result[3][:, 1], "-r", label=name)  # Red solid line
                plt.legend(loc="upper left", fontsize=14)
                plt.savefig(os.path.join(args.output_dir, f"route_{name}_{count}.png"))

            count += 1
    
        successful_runs = num_trajectories - fail_counts[name]
        if successful_runs != 0:
            performance_results[name] = total_time / successful_runs  # Average execution time
            distance_results[name] = total_dist / successful_runs
            print(f"{name}: {performance_results[name]:.6f} seconds (average)")
            print(f"{name}: {distance_results[name]:.6f} meters (average)")

    # Save updated configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Print the results
    for name, time_taken in performance_results.items():
        print(f"{name}: {time_taken:.6f} seconds (average)")
    for name, dist in distance_results.items():
        print(f"{name}: {dist:.6f} meters (average)")

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

    # Save the combined results to a text file
    with open(os.path.join(args.output_dir, "combined_results.txt"), "w") as f:
        f.write("Algorithm\tFail Count\tAverage Execution Time (seconds)\tAverage Trajectory Distance (m)\n")
        for name in fail_counts.keys():
            fail_count = fail_counts.get(name, 0)
            time_taken = performance_results.get(name, 0)
            dist = distance_results.get(name, 0)
            f.write(f"{name}\t{fail_count}\t{time_taken:.6f}\t{dist:.6f}\n")

    # Adjust layout and show plot
    plt.tight_layout()  # Ensure there's enough space between the plots
    plt.savefig(os.path.join(args.output_dir, "performance_route_planner.png"))

    # Save trajectory data
    for i in range(num_trajectories):
        plt.figure()
        map_instance.plot_map(title=f"Compare Route Trajectory")
        plt.plot(start_pose.x, start_pose.y, "og")
        plt.plot(goal_pose.x, goal_pose.y, "xb")
        colors = ["g--", "b--", "c--", "m--", "y--", "k--", "r-"]
        for j, name in enumerate(algorithms.keys()):
            if trajectory_data[name].get(i) is not None:
                plt.plot(trajectory_data[name][i][:, 0], trajectory_data[name][i][:, 1], colors[j], label=name)
        plt.legend(loc="upper left", fontsize=14)
        plt.savefig(os.path.join(args.output_dir, f"compare_route_trajectory_{i}.png"))
        plt.close()

if __name__ == "__main__":
    main()
