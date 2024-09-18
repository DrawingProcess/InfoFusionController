import numpy as np
import math
import matplotlib.pyplot as plt

from utils import calculate_angle, transform_arrays_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from controller.base_controller import BaseController

from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar

class StanleyController(BaseController):
    def __init__(self, k, dt, wheelbase, map_instance):
        super().__init__(dt=dt, wheelbase=wheelbase, map_instance=map_instance)
        self.k = k  # Control gain for the cross-track error

    def compute_control(self, current_state, target_point):
        # 현재 상태와 목표 지점의 좌표를 추출
        x, y, theta = current_state[:3]
        target_x, target_y = target_point[:2]
    
        # 경로 각도 계산
        dx = target_x - x
        dy = target_y - y
        path_angle = math.atan2(dy, dx)
    
        # Heading error (차량의 방향과 경로의 방향 간의 차이)
        heading_error = path_angle - theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
    
        # Cross-track error 계산
        # 경로와 차량 위치 사이의 수직 거리를 계산합니다.
        cross_track_error = np.hypot(dx, dy) * math.sin(heading_error)
    
        # Stanley 조향각 계산
        velocity = max(current_state[3], 0.1)  # 속도가 0에 가까울 때를 방지
        steering_angle = heading_error + math.atan2(self.k * cross_track_error, velocity)
    
        # 차량이 올바른 방향으로 가고 있는지 확인하고, 뒤로 가는 경우를 방지
        if np.cos(heading_error) < 0:  # 만약 헤딩 에러가 90도 이상이면 차량이 뒤로 가려는 경우
            steering_angle += math.pi  # 조향각을 반대로 뒤집어 전진하도록 함
    
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

    map_instance.plot_map(title="Stanley Route Planner")
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")

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
    is_reached, trajectory_distance, trajectory  = controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    if is_reached:
        print("Plotting the final trajectory.")
        print(f"Total distance covered: {trajectory_distance}")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Stanley Path")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
