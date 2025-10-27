import matplotlib.pyplot as plt
import matplotlib.patches as patches
from load_obstacles import load_obstacles
from bezier import bezier_curve
from pathfinding import find_path
from utils import sample_end_point

plt.rcParams["figure.dpi"] = 400

def plot_path(obstacles, control_points, xvals, yvals, episode):
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
    plt.title(f"Episode {episode}")
    plt.show()

if __name__ == "__main__":
    start_points = [(2, 2), (4, 4), (6, 6), (1, 1), (3, 3),(1.5, 1), (1, 2), (1.5, 3),
                    (2.5, 2), (4.5, 4), (5.5, 6), (1.5, 4), (4, 4.5), (5, 5.5), (1.5, 2.5)]  # Define starting points for different targets
    #start_point = (2,2)
    n_targets = 1
    mean_end_point = (55, 55)
    std_dev_end_point = 0.8
    n_episodes = 300
    trajectory_file = f'target_visitation-dynamic_env/target_policies/trajectory(15t-50-expb).txt'
    obstacles = load_obstacles('target_visitation-dynamic_env/experiments/exp-b/obstacles.json')

    with open(trajectory_file, 'w') as file:
        for episode in range(1, n_episodes + 1):
            all_trajectories = []
            for start_point  in start_points:
                end_point = sample_end_point(mean=mean_end_point, std_dev=std_dev_end_point)
                path = find_path(start_point, end_point, obstacles)
                control_points = [start_point] + path + [end_point]
                xvals, yvals = bezier_curve(control_points)
                trajectories = [f"{x:05.2f}, {y:05.2f}" for x, y in zip(xvals, yvals)]
                all_trajectories.append(trajectories)
            # Ensure all trajectories have the same length by padding the shorter ones
            max_len = max(len(traj) for traj in all_trajectories)
            for traj in all_trajectories:
                traj.extend(['0.00, 0.00'] * (max_len - len(traj)))

            for i in range(max_len):
                line = '\t'.join(traj[i] for traj in all_trajectories)
                file.write(line + '\n')
