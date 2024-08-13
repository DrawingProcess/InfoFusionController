import math
import numpy as np

def calculate_angle(x_start, y_start, x_end, y_end):
    """Calculate the angle of the line segment from (x_start, y_start) to (x_end, y_end)."""
    dx = x_end - x_start
    dy = y_end - y_start
    return math.atan2(dy, dx)

def transform_arrays_with_angles(x_array, y_array, num_points=50, velocity=1.0):
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    
    n = len(x_array)
    transformed_list = []

    for i in range(n - 1):
        # Get segment start and end
        x_start, x_end = x_array[i], x_array[i + 1]
        y_start, y_end = y_array[i], y_array[i + 1]
        
        # Calculate angle for current segment
        theta_current = calculate_angle(x_start, y_start, x_end, y_end)
        
        # Calculate angle for the next segment
        if i < n - 2:
            x_next_start, x_next_end = x_array[i + 1], x_array[i + 2]
            y_next_start, y_next_end = y_array[i + 1], y_array[i + 2]
            theta_next = calculate_angle(x_next_start, y_next_start, x_next_end, y_next_end)
        else:
            # Use the same angle for the last segment if there's no next segment
            theta_next = theta_current
        
        # Average angle between current and next segment
        theta_avg = (theta_current + theta_next) / 2.0
        
        # Interpolation
        x_interp = np.linspace(x_start, x_end, num=num_points, endpoint=False)
        y_interp = np.linspace(y_start, y_end, num=num_points, endpoint=False)
        
        # Add the last point of the segment explicitly
        x_interp = np.append(x_interp, x_end)
        y_interp = np.append(y_interp, y_end)
        
        # Add points with the average angle to the final list
        for x, y in zip(x_interp, y_interp):
            transformed_list.append([x, y, theta_avg, velocity])
    
    return np.array(transformed_list)