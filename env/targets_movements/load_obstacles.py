import json

def load_obstacles(file_path):
    with open(file_path, 'r') as f:
        obstacles = json.load(f)
    return obstacles

#if __name__ == "__main__":
#    obstacles = load_obstacles('envs/targets_movements/obstacles.json')
#    print(obstacles)
