import argparse
import time
import json
import matplotlib.pyplot as plt

from utils import transform_trajectory_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.random_grid_map import RandomGridMap

from route_planner.geometry import Pose
from route_planner.informed_trrt_star_planner import InformedTRRTStar

from controller.mpc_controller import MPCController
from controller.adaptive_mpc_controller import AdaptiveMPCController
from controller.pure_pursuit_controller import PurePursuitController
from controller.mpc_mi_controller import MPCMIController
from controller.hybrid_mi_controller import HybridMIController
from controller.multi_purpose_mpc_controller import MultiPurposeMPCController
from controller.stanley_controller import StanleyController

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="Controller Speed Test with Informed TRRT* Route Planner.")
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

    # Informed TRRT* Planner
    planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)

    # 초기 경로 생성
    route_trajectorys, route_trajectory_opts = [], []
    MAX_ITER = 5
    count = 0

    while (count < MAX_ITER):
        isReached, total_distance, route_trajectory, route_trajectory_opt = planner.search_route(show_process=show_process)
        if not isReached:
            continue

        plt.clf()
        map_instance.plot_map(title=f"Informed TRRT* Route Planner [{count}]")
        plt.plot(start_pose.x, start_pose.y, "og")
        plt.plot(goal_pose.x, goal_pose.y, "xb")
        # plt.plot(route_trajectory[:, 0], route_trajectory[:, 1], "g--", label="Theta* Path")  # Green dashed line
        plt.plot(route_trajectory_opt[:, 0], route_trajectory_opt[:, 1], "-r", label="Informed TRRT Path")  # Red solid line
        plt.legend(loc="upper left")
        plt.savefig(f"results/test_controller/route_{count}.png")

        route_trajectorys.append(route_trajectory)
        route_trajectory_opts.append(route_trajectory_opt)
        count += 1

    # Controller selection using dictionary
    horizon = 10  # MPC horizon
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    goal_position = [goal_pose.x, goal_pose.y]
    algorithms = {
        'pure_pursuit': lambda: PurePursuitController(lookahead_distance=5.0, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'mpc_basic': lambda: MPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'adaptive_mpc': lambda: AdaptiveMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'mpc_mi': lambda: MPCMIController(horizons=[5, 10, 15], dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        # 'multi_purpose_mpc': lambda: MultiPurposeMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        # 'stanley': lambda: StanleyController(k=0.1, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
    }

    # 각 알고리즘의 성능 측정 및 실패 여부 확인
    performance_results = {}
    distance_results = {}
    fail_counts = {name: 0 for name in algorithms}

    for name, func in algorithms.items():
        count = 0
        total_time = 0
        total_dist = 0
        while (count < MAX_ITER):
            plt.clf()
            if show_process:
                map_instance.plot_map(title=f"{name} Controller [{count}]")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")
                # plt.plot(route_trajectorys[count][:, 0], route_trajectorys[count][:, 1], "g--", label="Theta* Path")  # Green dashed line
                plt.plot(route_trajectory_opts[count][:, 0], route_trajectory_opts[count][:, 1], "-r", label="Informed TRRT Path")  # Red solid line

            # 경로가 유효한 경우 컨트롤러 실행
            ref_trajectory = transform_trajectory_with_angles(route_trajectory_opts[count])
            start_time = time.time()
            is_reached, trajectory_distance, trajectory = func()
            end_time = time.time()
            control_time = end_time - start_time

            if not is_reached:
                fail_counts[name] += 1
            else:
                total_time += control_time
                total_dist += trajectory_distance

            if is_reached:
                plt.clf()
                map_instance.plot_map(title=f"{name} Controller [{count}]")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")
                # plt.plot(route_trajectorys[count][:, 0], route_trajectorys[count][:, 1], "g--", label="Theta* Path")  # Green dashed line
                plt.plot(route_trajectory_opts[count][:, 0], route_trajectory_opts[count][:, 1], "-r", label="Informed TRRT Path")  # Red solid line

                plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label=name)
                plt.legend(loc="upper left")
                plt.savefig(f"results/test_controller/controller_{name}_{count}.png")
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
    ax1.set_title("Algorithm Pathfinding Failure Counts ({MAX_ITER} Runs)")
    ax1.grid(True)

    # Performance Results Plot
    algorithm_names = [result[0] for result in sorted_performs]
    times = [result[1] for result in sorted_performs]
    ax2.barh(algorithm_names, times, color='skyblue')
    ax2.set_xlabel("Average Execution Time (seconds)")
    ax2.set_title("Algorithm Performance Comparison ({MAX_ITER} Runs)")
    ax2.grid(True)

    # Performance Results Plot
    algorithm_names = [result[0] for result in sorted_dists]
    dists = [result[1] for result in sorted_dists]
    ax3.barh(algorithm_names, dists, color='purple')
    ax3.set_xlabel("Average Trajectory Distance (m)")
    ax3.set_title("Algorithm Performance Comparison ({MAX_ITER} Runs)")
    ax3.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()  # Ensure there's enough space between the plots
    plt.savefig("results/test_controller/performance_controller.png")

if __name__ == "__main__":
    main()
