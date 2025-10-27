import numpy as np
from utils import is_point_in_obstacle, get_around_obstacle_points

def find_path(start, end, obstacles, step_size=0.2):
    path = [start]
    current = start
    while np.linalg.norm(np.array(current) - np.array(end)) > step_size:
        direction = np.array(end) - np.array(current)
        direction = direction / np.linalg.norm(direction) * step_size
        next_point = tuple(np.array(current) + direction)
        if not is_point_in_obstacle(next_point[0], next_point[1], obstacles):
            path.append(next_point)
            current = next_point
        else:
            around_points = get_around_obstacle_points(current, end, step_size, obstacles)
            for point in around_points:
                path.append(point)
                current = point
                break
    path.append(end)
    return path

if __name__ == "__main__":
    start_point = (0, 0)
    end_point = (40, 40)
    obstacles = [
        {"type": "rectangle", "x": 20, "y": 20, "width": 10, "height": 10},
        {"type": "circle", "x": 50, "y": 50, "radius": 5}
    ]
    path = find_path(start_point, end_point, obstacles)
    print(path)
