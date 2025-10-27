import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams["figure.dpi"] = 400

# Define the environment size S
S = 50

# Number of obstacles
num_obstacles = 20

# Lists to store obstacle information
obstacles = []

def is_overlapping(obs1, obs2):
    if obs1['type'] == 'rectangle' and obs2['type'] == 'rectangle':
        return not (obs1['x'] + obs1['width'] < obs2['x'] or
                    obs1['x'] > obs2['x'] + obs2['width'] or
                    obs1['y'] + obs1['height'] < obs2['y'] or
                    obs1['y'] > obs2['y'] + obs2['height'])
    elif obs1['type'] == 'circle' and obs2['type'] == 'circle':
        distance = ((obs1['x'] - obs2['x']) ** 2 + (obs1['y'] - obs2['y']) ** 2) ** 0.5
        return distance < (obs1['radius'] + obs2['radius'])
    else:
        rect, circ = (obs1, obs2) if obs1['type'] == 'rectangle' else (obs2, obs1)
        nearest_x = max(rect['x'], min(circ['x'], rect['x'] + rect['width']))
        nearest_y = max(rect['y'], min(circ['y'], rect['y'] + rect['height']))
        distance = ((nearest_x - circ['x']) ** 2 + (nearest_y - circ['y']) ** 2) ** 0.5
        return distance < circ['radius']

# Generate obstacles with minimal overlap
while len(obstacles) < num_obstacles:
    shape_type = random.choice(['rectangle', 'circle'])
    #shape_type = 'circle'
    if shape_type == 'rectangle':
        width = random.uniform(1, 7)
        height = random.uniform(1, 7)
        x = random.uniform(0, S - width)
        y = random.uniform(0, S - height)
        new_obstacle = {
            'type': 'rectangle',
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
    else:
        radius = random.uniform(1, 5)
        x = random.uniform(radius, S - radius)
        y = random.uniform(radius, S - radius)
        new_obstacle = {
            'type': 'circle',
            'x': x,
            'y': y,
            'radius': radius
        }

    if all(not is_overlapping(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
        obstacles.append(new_obstacle)

# Save obstacles to a JSON file
with open('experiments/exp-a/env4.json', 'w') as f:
    json.dump(obstacles, f, indent=4)

# Plot the obstacles
fig, ax = plt.subplots()
ax.set_xlim(0, S)
ax.set_ylim(0, S)

for obs in obstacles:
    if obs['type'] == 'rectangle':
        rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    elif obs['type'] == 'circle':
        circ = patches.Circle((obs['x'], obs['y']), obs['radius'], linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(circ)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
