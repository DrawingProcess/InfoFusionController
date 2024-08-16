import timeit
import math
import matplotlib.pyplot as plt

from route_planner.geometry import Pose
from space.parking_lot import ParkingLot

from route_planner.a_star_route_planner import AStarRoutePlanner
from route_planner.hybrid_a_star_route_planner import HybridAStarRoutePlanner
from route_planner.theta_star_planner import ThetaStar
from route_planner.rrt_star_planner import RRTStar
from route_planner.informed_rrt_star_planner import InformedRRTStar
from route_planner.informed_rrt_star_smooth_planner import InformedRRTSmoothStar
from route_planner.informed_trrt_star_planner import InformedTRRTStar

# 메인 함수
def main():
    # 주차장 환경 설정
    parking_lot = ParkingLot()
    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")

    # 시작 및 목표 지점 설정
    start_pose = Pose(14.0, 4.0, math.radians(0))
    goal_pose = Pose(50.0, 38.0, math.radians(90))
    print(f"Start planning (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, parking_lot.lot_width + 1)
    plt.ylim(-1, parking_lot.lot_height + 1)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    # show_process 변수로 show_process와 show_eclipse 제어
    show_process = False

    # 성능 테스트를 위한 알고리즘 함수들
    algorithms = {
        "A*": lambda: AStarRoutePlanner(parking_lot).search_route(start_pose, goal_pose, show_process),
        "Theta*": lambda: ThetaStar(start_pose, goal_pose, parking_lot).find_path(),
        "Hybrid A*": lambda: HybridAStarRoutePlanner(parking_lot).search_route(start_pose, goal_pose, show_process),
        "RRT*": lambda: RRTStar(start_pose, goal_pose, parking_lot).search_route(),
        "Informed RRT*": lambda: InformedRRTStar(start_pose, goal_pose, parking_lot, show_eclipse=show_process).search_route(show_process=show_process),
        "Informed RRT* (with smoothing)": lambda: InformedRRTSmoothStar(start_pose, goal_pose, parking_lot, show_eclipse=show_process).search_route(show_process=show_process),
        "Informed TRRT*": lambda: InformedTRRTStar(start_pose, goal_pose, parking_lot, show_eclipse=show_process).search_route(show_process=show_process),
    }

    # 각 알고리즘의 성능 측정 및 실패 여부 확인
    performance_results = {}
    fail_counts = {}
    for name, func in algorithms.items():
        time_taken = timeit.timeit(func, number=1)  # 한 번 실행하여 시간을 측정

        # 경로 생성 실패 여부 확인 (빈 리스트로 반환되면 실패)
        result = func()
        if isinstance(result, tuple) and (not result[0] or not result[1]):  # rx, ry가 빈 리스트일 때 실패로 간주
            fail_counts[name] = 1
        else:
            fail_counts[name] = 0

        performance_results[name] = time_taken
        print(f"{name}: {time_taken:.6f} 초")

    # 성능 결과 정렬 및 출력
    sorted_results = sorted(performance_results.items(), key=lambda x: x[1])
    for name, time_taken in sorted_results:
        print(f"{name}: {time_taken:.6f} 초")

    # 실패 카운트 시각화
    plt.figure()
    algorithm_names = list(fail_counts.keys())
    fail_values = list(fail_counts.values())
    plt.bar(algorithm_names, fail_values, color='red')
    plt.xlabel("Algorithm")
    plt.ylabel("Fail Count")
    plt.title("Algorithm Pathfinding Failure Counts")
    plt.grid(True)
    plt.show()

    # 성능 결과 시각화
    plt.figure()
    algorithm_names = [result[0] for result in sorted_results]
    times = [result[1] for result in sorted_results]
    plt.barh(algorithm_names, times, color='skyblue')
    plt.xlabel("Execution Time (seconds)")
    plt.title("Algorithm Performance Comparison")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
