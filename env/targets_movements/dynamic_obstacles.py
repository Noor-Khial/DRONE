import json
import random
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Parameters
num_files = 5
shift_units = 10
shift_percentages = [0.2, 0.4, 0.6, 0.8, 1.0]

# Create output directory
output_dir = 'env/targets_movements/shifted_obstacles'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Load the original obstacle file
with open('env/targets_movements/obstacles2.json', 'r') as f:
    original_obstacles = json.load(f)

def check_overlap(obstacle1, obstacle2):
    if obstacle1['type'] == 'rectangle' and obstacle2['type'] == 'rectangle':
        return not (obstacle1['x'] + obstacle1['width'] <= obstacle2['x'] or
                    obstacle2['x'] + obstacle2['width'] <= obstacle1['x'] or
                    obstacle1['y'] + obstacle1['height'] <= obstacle2['y'] or
                    obstacle2['y'] + obstacle2['height'] <= obstacle1['y'])
    elif obstacle1['type'] == 'circle' and obstacle2['type'] == 'circle':
        distance = ((obstacle1['x'] - obstacle2['x'])**2 + (obstacle1['y'] - obstacle2['y'])**2)**0.5
        return distance < (obstacle1['radius'] + obstacle2['radius'])
    else:
        # For simplicity, we'll treat circle-rectangle overlap as rectangle-rectangle
        circle = obstacle1 if obstacle1['type'] == 'circle' else obstacle2
        rect = obstacle2 if obstacle1['type'] == 'circle' else obstacle1
        closest_x = max(rect['x'], min(circle['x'], rect['x'] + rect['width']))
        closest_y = max(rect['y'], min(circle['y'], rect['y'] + rect['height']))
        distance = ((circle['x'] - closest_x)**2 + (circle['y'] - closest_y)**2)**0.5
        return distance < circle['radius']

def shift_obstacles(obstacles, shift_percentage, shift_units):
    num_obstacles_to_shift = int(len(obstacles) * shift_percentage)
    shifted_obstacles = obstacles[:]
    
    for _ in range(num_obstacles_to_shift):
        idx = random.randint(0, len(obstacles) - 1)
        original_position = shifted_obstacles[idx].copy()
        
        max_attempts = 10
        for _ in range(max_attempts):
            direction = random.choice(['x', 'y'])
            shift = random.choice([-shift_units, shift_units])
            
            if direction == 'x':
                shifted_obstacles[idx]['x'] += shift
            else:
                shifted_obstacles[idx]['y'] += shift
            
            # Check bounds
            if shifted_obstacles[idx]['type'] == 'rectangle':
                if (shifted_obstacles[idx]['x'] < 0 or 
                    shifted_obstacles[idx]['x'] + shifted_obstacles[idx]['width'] > 80 or
                    shifted_obstacles[idx]['y'] < 0 or 
                    shifted_obstacles[idx]['y'] + shifted_obstacles[idx]['height'] > 80):
                    shifted_obstacles[idx] = original_position.copy()
                    continue
            else:  # circle
                if (shifted_obstacles[idx]['x'] - shifted_obstacles[idx]['radius'] < 0 or 
                    shifted_obstacles[idx]['x'] + shifted_obstacles[idx]['radius'] > 80 or
                    shifted_obstacles[idx]['y'] - shifted_obstacles[idx]['radius'] < 0 or 
                    shifted_obstacles[idx]['y'] + shifted_obstacles[idx]['radius'] > 80):
                    shifted_obstacles[idx] = original_position.copy()
                    continue
            
            # Check overlap
            overlap = False
            for j, other_obstacle in enumerate(shifted_obstacles):
                if j != idx and check_overlap(shifted_obstacles[idx], other_obstacle):
                    overlap = True
                    break
            
            if not overlap:
                break  # Successfully shifted without overlap
            else:
                shifted_obstacles[idx] = original_position.copy()
        
    return shifted_obstacles

S = 50
# Generate and save the new obstacle files
for percentage in shift_percentages:
    new_obstacles = shift_obstacles(original_obstacles, percentage, shift_units)
    fig, ax = plt.subplots()
    ax.set_xlim(0, S)
    ax.set_ylim(0, S)

    for obs in new_obstacles:
        if obs['type'] == 'rectangle':
            rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        elif obs['type'] == 'circle':
            circ = patches.Circle((obs['x'], obs['y']), obs['radius'], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(circ)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    file_name = f'obstacles_{percentage}.json'
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(new_obstacles, f, indent=4)

print(f"Generated {num_files} obstacle files in the '{output_dir}' directory.")