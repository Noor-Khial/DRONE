import numpy as np

def is_point_in_obstacle(x, y, obstacles, margin=0.7):
    for obs in obstacles:
        if obs['type'] == 'rectangle':
            if (obs['x'] - margin) <= x <= (obs['x'] + obs['width'] + margin) and (obs['y'] - margin) <= y <= (obs['y'] + obs['height'] + margin):
                return True
        elif obs['type'] == 'circle':
            if np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2) <= (obs['radius'] + margin):
                return True
    return False

def get_around_obstacle_points(current, end, step_size, obstacles):
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = [(current[0] + step_size * np.cos(angle), current[1] + step_size * np.sin(angle)) for angle in angles]
    
    #valid_points = []
    #for point in points:
    #    if not is_point_in_obstacle(point[0], point[1], obstacles):
    #        valid_points.append(point)
    #
    #if not valid_points:
    #    return [current]  # If no valid points, return the current point
    
    points.sort(key=lambda point: np.linalg.norm(np.array(point) - np.array(end)))
    return points

import numpy as np

def sample_end_point(mean=(100, 100), std_dev=10):
    x = np.random.normal(mean[0], std_dev)
    y = np.random.normal(mean[1], std_dev)
    return (x, y)