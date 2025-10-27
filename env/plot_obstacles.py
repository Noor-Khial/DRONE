import pygame
import numpy as np
from targets_movements.load_obstacles import load_obstacles

class ObstaclePlotter:
    def __init__(self, window_size=1000, size=50, obstacles_file='experiments/exp-a/env1.json'):
        self.window_size = window_size
        self.size = size
        self.obstacles = load_obstacles(obstacles_file)
        pygame.init()
        # Ensure Pygame initializes the display window even if it's not shown
        self.window = pygame.display.set_mode((self.window_size, self.window_size))

    def render_obstacles(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Fill the background with white
        pix_square_size = self.window_size / self.size

        for obs in self.obstacles:
            if obs['type'] == 'rectangle':
                rect_x = int(obs['x'] * pix_square_size)
                rect_y = int(obs['y'] * pix_square_size)
                rect_width = int(obs['width'] * pix_square_size)
                rect_height = int(obs['height'] * pix_square_size)
                pygame.draw.rect(canvas, (128, 128, 128), pygame.Rect(rect_x, rect_y, rect_width, rect_height), 0)
            elif obs['type'] == 'circle':
                circle_x = int(obs['x'] * pix_square_size)
                circle_y = int(obs['y'] * pix_square_size)
                radius = int(obs['radius'] * pix_square_size)
                pygame.draw.circle(canvas, (128, 128, 128), (circle_x, circle_y), radius)

        return canvas

    def save_as_image(self, filename='obstacles.png'):
        canvas = self.render_obstacles()
        # Check if the canvas is valid and save the image
        if isinstance(canvas, pygame.Surface):
            pygame.image.save(canvas, filename)
            print(f"Obstacles image saved as {filename}")
        else:
            print("Error: canvas is not a valid Pygame surface")

    def close(self):
        pygame.quit()

# Usage:
plotter = ObstaclePlotter(window_size=1000, size=50, obstacles_file='env/targets_movements/shifted_obstacles/obstacles_1.0.json')
plotter.save_as_image('experiments/exp-a/obstacles_1.0.png')
plotter.close()
