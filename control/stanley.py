import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, transform_arrays_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from control.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class StanleyController(BaseController):
    def __init__(self, k, dt, wheelbase, map_instance):
        super().__init__(dt=dt, wheelbase=wheelbase, map_instance=map_instance)
        self.k = k  # Control gain for the cross-track error
        self.dt = dt  # Time step for the controller
        self.wheelbase = wheelbase  # Wheelbase of the vehicle
        self.map_instance = map_instance  # Map instance for collision checking
    
    def compute_control(self, current_state, target_point):
        # 현재 상태와 목표 지점의 좌표를 추출
        x, y, theta = current_state[:3]
        target_x, target_y = target_point[:2]
    
        # 각도 오차 계산
        dx = target_x - x
        dy = target_y - y
        path_angle = math.atan2(dy, dx)
        heading_error = path_angle - theta
    
        # 각도 오차를 [-pi, pi] 범위로 맞추기
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
    
        # 측면 오프셋 계산 (cross-track error)
        # 경로와 차량 간의 수직 거리 계산
        cross_track_error = np.hypot(dx, dy) * math.sin(heading_error)
    
        # 속도에 따라 동적으로 k 값을 조정하여 안정성 향상
        velocity = max(current_state[3], 0.1)  # 속도가 0에 가까울 때를 방지
        dynamic_k = self.k / (1 + velocity)  # 속도가 높을수록 k 값을 줄여줌
    
        # Stanley 조향각 계산
        steering_angle = heading_error + math.atan2(dynamic_k * cross_track_error, velocity)
    
        return steering_angle


def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start Stanley Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    map_instance.plot_map()
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("Stanley Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    # Create Informed TRRT* planner
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)
    rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)

    # Ensure the route generation is completed
    try:
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

    # Plot Theta* Path
    plt.plot(rx, ry, "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path
    plt.plot(rx_opt, ry_opt, "-r", label="Informed TRRT* Path")  # Red solid line

    # Stanley Controller
    k = 0.5  # Example gain for the Stanley controller
    dt = 0.1  # Time step for the controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    controller = StanleyController(k, dt, wheelbase, map_instance)

    # Follow the trajectory using the Stanley controller
    goal_position = [goal_pose.x, goal_pose.y]
    trajectory = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="Stanley Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()