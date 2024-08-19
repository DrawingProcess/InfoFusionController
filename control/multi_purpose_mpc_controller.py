import numpy as np
import matplotlib.pyplot as plt

from utils import calculate_angle, transform_arrays_with_angles

from map.parking_lot import ParkingLot
from map.complex_grid_map import ComplexGridMap

from control.mpc_controller import MPCController
from route_planner.informed_trrt_star_planner import Pose, InformedTRRTStar


class MultiPurposeMPCController(MPCController):
    def __init__(self, horizon, dt, wheelbase, map_instance, mode="path_tracking"):
        super().__init__(horizon, dt, wheelbase, map_instance)
        self.mode = mode  # 모드 설정 (path_tracking, time_optimal, obstacle_avoidance)
        self.weight_matrices = self.get_weight_matrices()

    def get_weight_matrices(self):
        # 모드에 따른 가중치 행렬 설정
        if self.mode == "path_tracking":
            return {
                "state_deviation": 10.0,  # 경로 추적에서 위치 편차의 가중치
                "control_effort": 1.0     # 제어 입력의 가중치
            }
        elif self.mode == "time_optimal":
            return {
                "state_deviation": 1.0,   # 시간 최적화 주행에서 위치 편차의 가중치
                "control_effort": 0.1,    # 제어 입력의 가중치 (속도 최대화)
                "time_penalty": 10.0      # 시간 페널티 가중치
            }
        elif self.mode == "obstacle_avoidance":
            return {
                "state_deviation": 5.0,   # 장애물 회피에서 위치 편차의 가중치
                "control_effort": 1.0,    # 제어 입력의 가중치
                "obstacle_penalty": 10.0  # 장애물 페널티 가중치
            }
        else:
            raise ValueError("Invalid mode selected for MultiPurposeMPC.")

    def compute_cost(self, predicted_states, ref_trajectory):
        cost = 0
        for i in range(len(predicted_states)):
            if i >= len(ref_trajectory):
                break
            state = predicted_states[i]
            ref_state = ref_trajectory[i]
            # 경로 추적에서의 위치 편차 가중치
            state_deviation = np.sum((state - ref_state) ** 2) * self.weight_matrices["state_deviation"]
            control_effort = np.sum((state[3] - ref_state[3]) ** 2) * self.weight_matrices["control_effort"]

            cost += state_deviation + control_effort

            # 시간 최적화 모드의 경우
            if self.mode == "time_optimal":
                time_penalty = state[3] * self.weight_matrices["time_penalty"]  # 속도 페널티 추가
                cost += time_penalty

            # 장애물 회피 모드의 경우
            if self.mode == "obstacle_avoidance" and self.map_instance.is_obstacle_near(state):
                obstacle_penalty = self.weight_matrices["obstacle_penalty"]
                cost += obstacle_penalty

        return cost

    def follow_trajectory(self, start_pose, ref_trajectory, goal_position, show_process=False):
        # 초기 상태와 경로 초기화
        start_pose.theta = calculate_angle(start_pose.x, start_pose.y, ref_trajectory[1, 0], ref_trajectory[1, 1])
        current_state = np.array([start_pose.x, start_pose.y, start_pose.theta, 0.0])
        trajectory = [current_state.copy()]

        # 경로를 따르며 목표 지점에 도달
        for i in range(len(ref_trajectory) - self.horizon):
            ref_segment = ref_trajectory[i:i + self.horizon]
            control_input = self.optimize_control(current_state, ref_segment)
            next_state = self.apply_control(current_state, control_input)

            # 충돌 감지 및 회피
            if not self.is_collision_free(current_state, next_state):
                print(f"Collision detected at step {i}. Avoiding obstacle...")
                next_state = self.avoid_obstacle(current_state, next_state)

            current_state = next_state
            trajectory.append(current_state)

            # 현재 상태 시각화
            if show_process:
                plt.plot(current_state[0], current_state[1], "xr")
                plt.pause(0.001)

        # 목표 지점에 도달하지 않은 경우 위치 조정
        if not self.is_goal_reached(current_state, goal_position):
            print("Final adjustment to reach the goal.")
            current_state[0], current_state[1] = goal_position
            current_state[2] = calculate_angle(current_state[0], current_state[1], goal_position[0], goal_position[1])
            trajectory.append(current_state)

        return np.array(trajectory)



def main(map_type="ComplexGridMap"):
    # 사용자가 선택한 맵 클래스에 따라 인스턴스 생성
    if map_type == "ParkingLot":
        map_instance = ParkingLot(lot_width=100, lot_height=75)
    else:  # Default to ComplexGridMap
        map_instance = ComplexGridMap(lot_width=100, lot_height=75)

    # 유효한 시작과 목표 좌표 설정
    start_pose = map_instance.get_random_valid_start_position()
    goal_pose = map_instance.get_random_valid_goal_position()
    print(f"Start MPC Controller (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

    # 맵과 장애물 및 시작/목표 지점을 표시
    map_instance.plot_map()
    plt.plot(start_pose.x, start_pose.y, "og")
    plt.plot(goal_pose.x, goal_pose.y, "xb")
    plt.xlim(-1, map_instance.lot_width + 1)
    plt.ylim(-1, map_instance.lot_height + 1)
    plt.title("MPC Route Planner")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")

    # Create Informed TRRT* planner
    informed_rrt_star = InformedTRRTStar(start_pose, goal_pose, map_instance, show_eclipse=False)

    # Ensure the route generation is completed
    try:
        rx, ry, rx_opt, ry_opt = informed_rrt_star.search_route(show_process=False)
        if len(rx_opt) == 0 or len(ry_opt) == 0:
            print("TRRT* was unable to generate a valid path.")
            return

    except Exception as e:
        print(f"Error in route generation: {e}")
        return

    # Transform reference trajectory
    ref_trajectory = transform_arrays_with_angles(rx_opt, ry_opt)

    # Plot Theta* Path
    plt.plot(rx, ry, "g--", label="Theta* Path")  # Green dashed line

    # Plot Optimized Path 
    plt.plot(rx_opt, ry_opt, "-r", label="Informed TRRT* Path")  # Red solid line

    # MPC Controller
    wheelbase = 2.5  # Example wheelbase of the vehicle in meters
    mpc_controller = MultiPurposeMPCController(horizon=10, dt=0.1, map_instance=map_instance, wheelbase=wheelbase)

    # Follow the trajectory using the MPC controller
    goal_position = [goal_pose.x, goal_pose.y]
    trajectory = mpc_controller.follow_trajectory(start_pose, ref_trajectory, goal_position, show_process=True)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="MPC Path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()