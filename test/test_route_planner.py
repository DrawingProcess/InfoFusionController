import argparse
import time
import json
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

# 메인 함수
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

    # show_process 변수로 show_process와 show_eclipse 제어
    show_process = args.show_process

    # 성능 테스트를 위한 알고리즘 함수들
    algorithms = {
        "A*": lambda: AStarRoutePlanner(start_pose, goal_pose, map_instance).search_route(show_process),
        "Theta*": lambda: ThetaStar(start_pose, goal_pose, map_instance).search_route(show_process),
        "Hybrid A*": lambda: HybridAStarRoutePlanner(start_pose, goal_pose, map_instance).search_route(show_process),
        "RRT*": lambda: RRTStar(start_pose, goal_pose, map_instance).search_route(show_process),
        "Informed RRT*": lambda: InformedRRTStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
        "Informed RRT*(smoothing)": lambda: InformedRRTSmoothStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
        "Informed TRRT*": lambda: InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
    }

    # 각 알고리즘의 성능 측정 및 실패 여부 확인
    performance_results = {}
    distance_results = {}
    fail_counts = {name: 0 for name in algorithms}
    MAX_ITER = 5

    for name, func in algorithms.items():
        count = 0
        total_time = 0.0
        total_dist = 0.0
        while (count < MAX_ITER):
            plt.clf()  # 각 알고리즘 실행 전 플롯 초기화
            if show_process:
                map_instance.plot_map(title=f"{name} Route Planner")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")
            
            start_time = time.time()  # 시작 시간
            result = func()
            end_time = time.time()    # 종료 시간
            time_taken = end_time - start_time  # 실행 시간 계산

            if not result[0]:  # if not isReached
                fail_counts[name] += 1
            else:
                total_time += time_taken
                total_dist += result[1]

            if result[0]:
                plt.clf()  # 각 알고리즘 실행 전 플롯 초기화
                map_instance.plot_map(title=f"{name} Route Planner")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")

                if len(result) == 3:
                    plt.plot(result[2][:, 0], result[2][:, 1], "-r", label=name)  # Green dashed line
                else: # len(result) == 4:
                    plt.plot(result[3][:, 0], result[3][:, 1], "-r", label=name)  # Red solid line
                plt.legend(loc="upper left")
                plt.savefig(f"results/test_route_planner/route_{name}_{count}.png")
            count += 1
    
        if MAX_ITER - fail_counts[name] != 0:
            performance_results[name] = total_time / (MAX_ITER - fail_counts[name])  # 평균 실행 시간 계산
            distance_results[name] = total_dist / (MAX_ITER - fail_counts[name])
            print(f"{name}: {performance_results[name]:.6f} 초 (평균)")
            print(f"{name}: {distance_results[name]:.6f} m (평균)")

    # 성능 결과 정렬 및 출력
    sorted_performs = sorted(performance_results.items(), key=lambda x: x[1])
    for name, time_taken in sorted_performs:
        print(f"{name}: {time_taken:.6f} 초 (평균)")
    sorted_dists = sorted(distance_results.items(), key=lambda x: x[1])
    for name, dist in sorted_dists:
        print(f"{name}: {dist:.6f} m (평균)")

    # Plot the two charts side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 2 columns

    # Failure Counts Plot
    algorithm_names = list(fail_counts.keys())
    fail_values = list(fail_counts.values())
    ax1.barh(algorithm_names, fail_values, color='red')
    ax1.set_xlabel("Fail Count")
    ax1.set_ylabel("Algorithm")
    ax1.set_title(f"Algorithm Pathfinding Failure Counts ({MAX_ITER} Runs)")
    ax1.grid(True)

    # Performance Results Plot
    algorithm_names = [result[0] for result in sorted_performs]
    times = [result[1] for result in sorted_performs]
    ax2.barh(algorithm_names, times, color='skyblue')
    ax2.set_xlabel("Average Execution Time (seconds)")
    ax2.set_title(f"Algorithm Performance Comparison ({MAX_ITER} Runs)")
    ax2.grid(True)

    # Performance Results Plot
    algorithm_names = [result[0] for result in sorted_dists]
    dists = [result[1] for result in sorted_dists]
    ax3.barh(algorithm_names, dists, color='purple')
    ax3.set_xlabel("Average Trajectory Distance (m)")
    ax3.set_title(f"Algorithm Performance Comparison ({MAX_ITER} Runs)")
    ax3.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()  # Ensure there's enough space between the plots
    plt.savefig("results/test_route_planner/performance_route_planner.png")

if __name__ == "__main__":
    main()
