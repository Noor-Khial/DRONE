import numpy as np

def read_trajectory(file_path, index):
    ## i have created a trajectory for 300 episode, each with 400 step
    index = index % (300 * 200)
    positions = []
    with open(file_path, 'r') as file:
        line = next((line for idx, line in enumerate(file) if idx == index), None)
        if line is not None:
            positions = line.strip().split('\t')
            positions = [eval(pos.strip()) for pos in positions if pos.strip() != '']
            positions = [list(pos) for pos in positions]  # Convert tuples to lists
    return np.array(positions)

def calculate_distance(position1, position2):
    return np.linalg.norm(position1 - position2)

def normalize_angle(value):
    value = max(-2, min(value, 2))
    normalized_value = value + 2
    normalized_angle = normalized_value * 90
    return normalized_angle * np.pi / 180
