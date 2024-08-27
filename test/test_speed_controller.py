import argparse
import time
import matplotlib.pyplot as plt

from utils import transform_arrays_with_angles

from map.parking_lot import ParkingLot
from map.fixed_grid_map import FixedGridMap
from map.complex_grid_map import ComplexGridMap

from route_planner.geometry import Pose
from route_planner.informed_trrt_star_planner import InformedTRRTStar

from controller.mpc_controller import MPCController
from controller.adaptive_mpc_controller import AdaptiveMPCController
from controller.multi_purpose_mpc_controller import MultiPurposeMPCController
from controller.pure_pursuit_controller import PurePursuitController
from controller.stanley_controller import StanleyController

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="Controller Speed Test with Informed TRRT* Route Planner.")
    parser.add_argument('--map', type=str, default='parking_lot', choices=['parking_lot', 'fixed_grid', 'complex_grid'], help='Choose the map type.')
    args = parser.parse_args()

    # Map selection using dictionary
    map_options = {
        'parking_lot': ParkingLot,
        'fixed_grid': FixedGridMap,
        'complex_grid': ComplexGridMap
    }
    map_instance = map_options[args.map]()

    if args.map == "parking_lot":
        start_pose = Pose(14.0, 4.0, 0)
        goal_pose = Pose(50.0, 38.0, 1.57)
    elif args.map == "fixed_grid":
        start_pose = Pose(3, 5, 0)
        goal_pose = Pose(5, 15, 0)
    else:
        start_pose = map_instance.get_random_valid_start_position()
        goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # show_process 변수로 show_process와 show_eclipse 제어
    show_process = False

    if show_process:
        map_instance.plot_map()
        plt.plot(start_pose.x, start_pose.y, "og")
        plt.plot(goal_pose.x, goal_pose.y, "xb")
        plt.xlim(-1, map_instance.lot_width + 1)
        plt.ylim(-1, map_instance.lot_height + 1)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis("equal")

    # Informed TRRT* Planner
    planner = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)

    # Controller selection using dictionary
    horizon = 10  # MPC horizon
    dt = 0.1  # Time step
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    goal_position = [goal_pose.x, goal_pose.y]
    algorithms = {
        'mpc_basic': lambda: MPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'adaptive_mpc': lambda: AdaptiveMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'multi_purpose_mpc': lambda: MultiPurposeMPCController(horizon=horizon, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'pure_pursuit': lambda: PurePursuitController(lookahead_distance=5.0, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
        'stanley': lambda: StanleyController(k=0.1, dt=dt, wheelbase=wheelbase, map_instance=map_instance).follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=show_process),
    }

    # 각 알고리즘의 성능 측정 및 실패 여부 확인
    performance_results = {}
    fail_counts = {name: 0 for name in algorithms}

    for name, func in algorithms.items():
        total_time = 0.0
        count = 0
        fail_count = 0
        while(True):  # 10번 반복 실행
            if count >= 10:
                break

            start_time = time.time()
            rx, ry, rx_opt, ry_opt = planner.search_route(show_process=False)
            end_time = time.time()
            planning_time = end_time - start_time

            if len(rx_opt) == 0 or len(ry_opt) == 0:
                continue

            # 경로가 유효한 경우 컨트롤러 실행
            ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)
            start_time = time.time()
            result = func()
            end_time = time.time()

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

if __name__ == "__main__":
    main()
