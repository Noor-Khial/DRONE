import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from load_obstacles import load_obstacles
from bezier import bezier_curve
from pathfinding import find_path
from utils import sample_end_point

plt.rcParams["figure.dpi"] = 400

def plot_path(obstacles, control_points, xvals, yvals, episode, file_name):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Plot obstacles
    for obs in obstacles:
        if obs['type'] == 'rectangle':
            rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        elif obs['type'] == 'circle':
            circ = patches.Circle((obs['x'], obs['y']), obs['radius'], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(circ)

    # Plot Bezier curve
    ax.plot(xvals, yvals, 'b-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Episode {episode} - {file_name}")

if __name__ == "__main__":
    start_points = [(0, 0), (10, 0), (0, 10)]  # Define starting points for different targets
    mean_end_point = (45, 49)
    std_dev_end_point = 0.5
    n_episodes = 10

    obstacle_dir = 'env/targets_movements/shifted_obstacles'
    trajectory_dir = 'target_policies'

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    for obstacle_file in os.listdir(obstacle_dir):
        if obstacle_file.endswith('.json'):
            file_path = os.path.join(obstacle_dir, obstacle_file)
            obstacles = load_obstacles(file_path)
            trajectory_file = os.path.join(trajectory_dir, f'trajectory_{os.path.splitext(obstacle_file)[0]}.txt')

            with open(trajectory_file, 'w') as file:
                for episode in range(1, n_episodes + 1):
                    all_trajectories = []
                    for start_point in start_points:
                        end_point = sample_end_point(mean=mean_end_point, std_dev=std_dev_end_point)
                        path = find_path(start_point, end_point, obstacles)
                        control_points = [start_point] + path + [end_point]
                        xvals, yvals = bezier_curve(control_points)
                        trajectory = ', '.join([f"{x:05.2f}, {y:05.2f}" for x, y in zip(xvals, yvals)])
                        all_trajectories.append(trajectory)
                    file.write('\t'.join(all_trajectories) + '\n')

                    # Plot the path for visualization
                    plot_path(obstacles, control_points, xvals, yvals, episode, obstacle_file)
plt.show()