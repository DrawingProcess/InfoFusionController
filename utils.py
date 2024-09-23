import math
import numpy as np

def calculate_angle(x_start, y_start, x_end, y_end):
    """Calculate the angle of the line segment from (x_start, y_start) to (x_end, y_end)."""
    dx = x_end - x_start
    dy = y_end - y_start
    return math.atan2(dy, dx)

def calculate_trajectory_distance(trajectory):
    trajectory = np.array(trajectory)
    # 각 점 사이의 유클리드 거리를 계산하여 총 거리를 산출
    distances = np.sqrt(np.diff(trajectory[:, 0])**2 + np.diff(trajectory[:, 1])**2)
    total_distance = np.sum(distances)
    return total_distance

def transform_trajectory(x_array, y_array):
    return np.array([list(pair) for pair in zip(x_array, y_array)])

def transform_trajectory_with_angles(trajectory, num_points=30, velocity=2.0, last_segment_factor=1):
    trajectory = np.array(trajectory)  # trajectory를 numpy 배열로 변환
    
    n = len(trajectory)
    transformed_list = []

    for i in range(n - 1):
        # Get segment start and end
        x_start, y_start = trajectory[i][:2]
        x_end, y_end = trajectory[i + 1][:2]
        
        # Calculate angle for current segment
        theta_current = calculate_angle(x_start, y_start, x_end, y_end)
        
        # Calculate angle for the next segment
        if i < n - 2:
            x_next_start, y_next_start = trajectory[i + 1][:2]
            x_next_end, y_next_end = trajectory[i + 2][:2]
            theta_next = calculate_angle(x_next_start, y_next_start, x_next_end, y_next_end)
        else:
            # Use the same angle for the last segment if there's no next segment
            theta_next = theta_current
        
        # Average angle between current and next segment
        theta_avg = (theta_current + theta_next) / 2.0
        
        # Adjust the number of points for the last segment
        if i == n - 2:
            # Increase the number of interpolation points for the last segment
            num_points_last = num_points * last_segment_factor
            x_interp = np.linspace(x_start, x_end, num=num_points_last, endpoint=False)
            y_interp = np.linspace(y_start, y_end, num=num_points_last, endpoint=False)
        else:
            # Regular interpolation for other segments
            x_interp = np.linspace(x_start, x_end, num=num_points, endpoint=False)
            y_interp = np.linspace(y_start, y_end, num=num_points, endpoint=False)
        
        # Add the last point of the segment explicitly
        x_interp = np.append(x_interp, x_end)
        y_interp = np.append(y_interp, y_end)
        
        # Add points with the average angle to the final list
        for x, y in zip(x_interp, y_interp):
            transformed_list.append([x, y, theta_avg, velocity])
    
    return np.array(transformed_list)

# def transform_trajectory_with_angles(x_array, y_array, num_points=20, velocity=2.0, last_segment_factor=5):
#     x_array = np.array(x_array)
#     y_array = np.array(y_array)
    
#     n = len(x_array)
#     transformed_list = []

#     for i in range(n - 1):
#         # Get segment start and end
#         x_start, x_end = x_array[i], x_array[i + 1]
#         y_start, y_end = y_array[i], y_array[i + 1]
        
#         # Calculate angle for current segment
#         theta_start = calculate_angle(x_start, y_start, x_end, y_end)
        
#         # Calculate angle for the next segment
#         if i < n - 2:
#             x_next_start, x_next_end = x_array[i + 1], x_array[i + 2]
#             y_next_start, y_next_end = y_array[i + 1], y_array[i + 2]
#             theta_end = calculate_angle(x_next_start, y_next_start, x_next_end, y_next_end)
#         else:
#             # Use the same angle for the last segment if there's no next segment
#             theta_end = theta_start

#         # Adjust the number of points for the last segment
#         if i == n - 2:
#             # Increase the number of interpolation points for the last segment
#             num_points_last = num_points * last_segment_factor
#             x_interp = np.linspace(x_start, x_end, num=num_points_last, endpoint=False)
#             y_interp = np.linspace(y_start, y_end, num=num_points_last, endpoint=False)
#             theta_interp = np.linspace(theta_start, theta_end, num=num_points_last, endpoint=False)
#         else:
#             # Regular interpolation for other segments
#             x_interp = np.linspace(x_start, x_end, num=num_points, endpoint=False)
#             y_interp = np.linspace(y_start, y_end, num=num_points, endpoint=False)
#             theta_interp = np.linspace(theta_start, theta_end, num=num_points, endpoint=False)
        
#         # Add the last point of the segment explicitly
#         x_interp = np.append(x_interp, x_end)
#         y_interp = np.append(y_interp, y_end)
#         theta_interp = np.append(theta_interp, theta_end)
        
#         # Add points with interpolated angles to the final list
#         for x, y, theta in zip(x_interp, y_interp, theta_interp):
#             transformed_list.append([x, y, theta, velocity])
    
#     return np.array(transformed_list)
