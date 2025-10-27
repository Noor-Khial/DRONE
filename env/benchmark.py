import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import heapq
import sys
import os

sys.path.append('.')

from throughWallDetection.target_detection import detection_probability
from env.targets_movements.load_obstacles import load_obstacles

def is_collision(point, obstacles):
    for obstacle in obstacles:
        if obstacle['type'] == 'rectangle':
            if (obstacle['x'] <= point[0] <= obstacle['x'] + obstacle['width'] and
                obstacle['y'] <= point[1] <= obstacle['y'] + obstacle['height']):
                return True
        elif obstacle['type'] == 'circle':
            if euclidean(point, (obstacle['x'], obstacle['y'])) <= obstacle['radius']:
                return True
    return False

def get_neighbors(current, step_size_range, obstacles, grid_size):
    neighbors = []
    for step_size in step_size_range:
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            next_pos = (current[0] + dx * step_size, current[1] + dy * step_size)
            if 0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size:
                # Check for collision along the path
                collision = False
                for t in np.linspace(0, 1, num=10):  # Check 10 points along the path
                    intermediate_pos = (current[0] + t * dx * step_size, current[1] + t * dy * step_size)
                    if is_collision(intermediate_pos, obstacles):
                        collision = True
                        break
                if not collision:
                    neighbors.append(next_pos)
    return neighbors

def dijkstra(start, goal, obstacles, step_size_range, grid_size):
    start = tuple(start)
    goal = tuple(goal)
    
    distances = {start: 0}
    pq = [(0, start)]
    came_from = {}
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if euclidean(current, goal) < min(step_size_range):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for neighbor in get_neighbors(current, step_size_range, obstacles, grid_size):
            distance = current_distance + euclidean(current, neighbor)
            
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                came_from[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    return None

def calculate_benchmark(trajectory_file, obstacles_file, start_point, step_size_range, min_distance, grid_size):
    with open(trajectory_file, 'r') as file:
        trajectories = [line.strip().split('\t') for line in file]

    targets = [np.array([[float(coord) for coord in point.split(',')] for point in traj]) for traj in zip(*trajectories)]
    obstacles = load_obstacles(obstacles_file)

    agent_pos = np.array(start_point)
    visited = [False] * len(targets)
    visited_positions = [None] * len(targets)
    total_distance = 0
    steps = 0
    rewards = []
    path = [agent_pos]

    for step in range(len(trajectories)):
        if all(visited):
            break

        step_reward = 0
        for target_idx, target_position in enumerate(targets):
            dist = euclidean(agent_pos, target_position[step])
            if not visited[target_idx]:
                detect = detection_probability(dist, target_position[step])
                step_reward += -(35 - detect)
                
                if dist < 3:
                    visited[target_idx] = True
                    visited_positions[target_idx] = agent_pos.copy()

        rewards.append(step_reward)

        distances = [euclidean(agent_pos, targets[i][step]) for i in range(3)]
        nearest_target = np.argmin(distances)

        if distances[nearest_target] > min_distance:
            dijkstra_path = dijkstra(agent_pos, targets[nearest_target][step], obstacles, step_size_range, grid_size)
            if dijkstra_path:
                new_pos = np.array(dijkstra_path[1])  # Take the next step in the Dijkstra path
                total_distance += euclidean(agent_pos, new_pos)
                agent_pos = new_pos
                path.append(agent_pos)

        steps += 1

    return total_distance, steps, all(visited), rewards, path, targets, visited_positions

def plot_environment(obstacles, targets, path, start_point, grid_size, visited_positions):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Plot obstacles
    for obs in obstacles:
        if obs['type'] == 'rectangle':
            rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], 
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        elif obs['type'] == 'circle':
            circ = patches.Circle((obs['x'], obs['y']), obs['radius'], 
                                  linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(circ)

    # Plot targets
    for idx, target in enumerate(targets):
        if visited_positions[idx] is not None:
            ax.plot(visited_positions[idx][0], visited_positions[idx][1], 'go', markersize=10, label=f'Visited {idx + 1}')

    # Plot agent's path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'b-')
    ax.plot(start_point[0], start_point[1], 'bs', markersize=10, label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'b*', markersize=10, label='End')

    ax.legend()
    plt.title("Agent's Path, Targets, and Obstacles")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.show()

def benchmark(start_point = (22.5, 22.5), step_size_range = range(1, 6), min_distance = 2.0, grid_size = 50,
              trajectory_file = 'target_policies/trajectory(3t)-50.txt',
              obstacles_file = 'experiments/exp-a/env1.json'):
    
    total_distance, steps, all_visited, rewards, path, targets, visited_positions = calculate_benchmark(
        trajectory_file, obstacles_file, start_point, step_size_range, min_distance, grid_size)

    cumulative_reward = sum(rewards)
    print(f"Total distance traveled: {total_distance:.2f}")
    print(f"Number of steps: {steps}")
    print(f"All targets visited: {all_visited}")
    print(f"Final step reward: {rewards[-1]:.2f}")
    print(f"Cumulative reward: {cumulative_reward:.2f}")

    # Plot the reward at each time step
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards)
    plt.title('Reward at Each Time Step')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()
    
    # Plot the environment, targets, and agent's path
    obstacles = load_obstacles(obstacles_file)
    plot_environment(obstacles, targets, path, start_point, grid_size, visited_positions)
    return rewards, cumulative_reward, steps, total_distance

benchmark()