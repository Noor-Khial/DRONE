from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
sys.path.append('.')
from env.targets_movements.load_obstacles import load_obstacles
from throughWallDetection.target_detection import detection_probability
from env.utilities import read_trajectory, calculate_distance, normalize_angle
from env.targets_movements.utils import is_point_in_obstacle

def generate_state_matrix(agent_location: Tuple[float, float], 
                          target_locations: List[Tuple[float, float]], 
                          obstacles_file: str) -> np.ndarray:
    obstacles = load_obstacles(obstacles_file)
    state_matrix = np.zeros((9, 9))

    for i in range(9):
        for j in range(9):
            cell_center = (agent_location[0] - 4 + i + 0.5, agent_location[1] - 4 + j + 0.5)
            
            if is_point_in_obstacle(cell_center[0], cell_center[1], obstacles):
                state_matrix[i, j] = 0
                continue

            max_prob = 0
            for target in target_locations:
                distance = calculate_distance(np.array(cell_center), np.array(target))
                prob = detection_probability(distance, target)
                max_prob = max(max_prob, prob)

            if is_point_in_obstacle(target[0], target[1], obstacles):
                max_prob = -max_prob

            state_matrix[i, j] = max_prob

    return state_matrix

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_heatmap(state_matrix: np.ndarray, agent_location: Tuple[float, float], 
                 target_locations: List[Tuple[float, float]]):
    plt.figure(figsize=(10, 8))
    
    # Define color limits
    vmin = -1
    vmax = 1

    # Create the heatmap
    ax = sns.heatmap(
        state_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0, 
        vmin=vmin, 
        vmax=vmax,
        annot_kws={"size": 12, "weight": 'bold'},  # Set font size and weight for annotations
        cbar_kws={"shrink": 0.8, "aspect": 10}  # Customize colorbar size and aspect ratio
    )
    
    # Access the colorbar
    cbar = ax.collections[0].colorbar
    
    # Set font size and weight for colorbar ticks
    cbar.ax.tick_params(labelsize=12, width=2)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=12, fontweight='bold')
    cbar.set_label(r'$D^{(t)}_i$', fontsize=24, fontweight='bold')
    
    # Mark agent location
    agent_x, agent_y = 4, 4  # Center of the matrix
    plt.plot(agent_y + 0.5, agent_x + 0.5, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    # Mark target locations
    for target in target_locations:
        target_x = int(target[0] - agent_location[0] + 4)
        target_y = int(target[1] - agent_location[1] + 4)
        if 0 <= target_x < state_matrix.shape[0] and 0 <= target_y < state_matrix.shape[1]:
            plt.plot(target_y + 0.5, target_x + 0.5)
    
    # Set title and labels with increased font size and bold text
    plt.xlabel('Y Coordinate (m)', fontsize=14, fontweight='bold')
    plt.ylabel('X Coordinate (m)', fontsize=14, fontweight='bold')
    
    # Set font size and weight for tick labels
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Invert y-axis to match typical matrix coordinates
    plt.gca().invert_yaxis()
    
    # Save the figure
    plt.savefig('Figures/stateTWTDmatrix1.png')
    
    # Display the plot
    plt.show()

# Example usage
#agent_location = (10, 8)
#obstacles_file = 'env/targets_movements/obstacles2.json'
#target_locations = read_trajectory('target_policies/trajectory(3t)-50.txt', 10)
#state_matrix = generate_state_matrix(agent_location, target_locations, obstacles_file)
#plot_heatmap(state_matrix, agent_location, target_locations)

def state(agent_location, target_locations , obstacles_file = 'experiments/exp-a/env1.json'):
    #plot_heatmap(state_matrix, agent_location, target_locations)
    return  generate_state_matrix(agent_location, target_locations, obstacles_file)