import argparse
import time
import math
import matplotlib.pyplot as plt

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.complex_grid_map import ComplexGridMap

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
    parser.add_argument('--map', type=str, default='fixed_grid', choices=['parking_lot', 'fixed_grid', 'complex_grid'], help='Choose the map type.')
    args = parser.parse_args()

    # Map selection using dictionary
    map_options = {
        'parking_lot': ParkingLot,
        'fixed_grid': FixedGridMap,
        'complex_grid': ComplexGridMap
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

    # show_process 변수로 show_process와 show_eclipse 제어
    show_process = True

    # 성능 테스트를 위한 알고리즘 함수들
    algorithms = {
        "A*": lambda: AStarRoutePlanner(start_pose, goal_pose, map_instance).search_route(show_process),
        "Theta*": lambda: ThetaStar(start_pose, goal_pose, map_instance).find_path(),
        "Hybrid A*": lambda: HybridAStarRoutePlanner(start_pose, goal_pose, map_instance).search_route(show_process),
        "RRT*": lambda: RRTStar(start_pose, goal_pose, map_instance).search_route(show_process),
        "Informed RRT*": lambda: InformedRRTStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
        "Informed RRT* (with smoothing)": lambda: InformedRRTSmoothStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
        "Informed TRRT*": lambda: InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=show_process).search_route(show_process=show_process),
    }

    # 각 알고리즘의 성능 측정 및 실패 여부 확인
    performance_results = {}
    fail_counts = {name: 0 for name in algorithms}

    for name, func in algorithms.items():
        total_time = 0.0
        for _ in range(10):  # 10번 반복 실행
            # 경로 생성 실패 여부 확인 (빈 리스트로 반환되면 실패)
            plt.clf()  # 각 알고리즘 실행 전 플롯 초기화
            if show_process:
                map_instance.plot_map(title=f"{name} Route Planner")
                plt.plot(start_pose.x, start_pose.y, "og")
                plt.plot(goal_pose.x, goal_pose.y, "xb")
            
            start_time = time.time()  # 시작 시간
            result = func()
            end_time = time.time()    # 종료 시간
            time_taken = end_time - start_time  # 실행 시간 계산
            total_time += time_taken

            if isinstance(result, tuple) and (not result[0] or not result[1]):  # rx, ry가 빈 리스트일 때 실패로 간주
                fail_counts[name] += 1
            print(result)

        performance_results[name] = total_time / 10  # 평균 실행 시간 계산
        print(f"{name}: {performance_results[name]:.6f} 초 (평균)")

    # 성능 결과 정렬 및 출력
    sorted_results = sorted(performance_results.items(), key=lambda x: x[1])
    for name, time_taken in sorted_results:
        print(f"{name}: {time_taken:.6f} 초 (평균)")

    # Plot the two charts side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Failure Counts Plot
    algorithm_names = list(fail_counts.keys())
    fail_values = list(fail_counts.values())
    ax1.barh(algorithm_names, fail_values, color='red')
    ax1.set_xlabel("Fail Count")
    ax1.set_ylabel("Algorithm")
    ax1.set_title("Algorithm Pathfinding Failure Counts (10 Runs)")
    ax1.grid(True)

    # Performance Results Plot
    algorithm_names = [result[0] for result in sorted_results]
    times = [result[1] for result in sorted_results]
    ax2.barh(algorithm_names, times, color='skyblue')
    ax2.set_xlabel("Average Execution Time (seconds)")
    ax2.set_title("Algorithm Performance Comparison (10 Runs)")
    ax2.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()  # Ensure there's enough space between the plots
    plt.show()

if __name__ == "__main__":
    main()
