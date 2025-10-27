import gym
from gym import spaces
import pygame
import numpy as np
import random
from env.utilities import read_trajectory, calculate_distance, normalize_angle
from env.targets_movements.load_obstacles import load_obstacles
from throughWallDetection.target_detection import detection_probability
from env.state import state


class GridWorldEnv:
    def __init__(self, render_mode=None, obstacles_file='env/targets_movements/shifted_obstacles/obstacles_1.0.json', 
                 size=5, trajectory_file='target_policies/trajectory(3t)-50.txt'):
        self.size = size
        self.window_size = 1000
        
        # Modification for Exp4 compatibility
        self.action_space = spaces.Box(low=np.array([-np.pi, -5.5]), high=np.array([np.pi, 5.5]), dtype=np.float32)
        
        self.trajectory_file = trajectory_file
        self.state = np.zeros(84)
        self.prev_action = np.array([self.size/3, self.size/3])
        self.target_observation = 0
        self._agent_location = np.array([self.size/3, self.size/3])
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.episode = 1
        self.found_targets = []
        self.t = 0
        self.obstacles = load_obstacles(obstacles_file)
        self.hit_obstacle = False

        self.total_dis = 0
        self.num_collision = 0
        self.cum_reward = 0
        
        # Ex p4 specific attributes
        self.current_state = None
        self.last_reward = 0
        self.envs = ['env/targets_movements/shifted_obstacles/obstacles_1.0.json', 'env/targets_movements/shifted_obstacles/obstacles_0.8.json', 'env/targets_movements/shifted_obstacles/obstacles_0.6.json']

    def _get_obs(self):
        observation = state(self._agent_location, self._target_location)
        observation = observation.reshape(-1)
        observation = np.append(observation, [self._agent_location[0], self._agent_location[1], self.t])
        self.state = observation
        self.current_state = observation
        return self.state

    def reset(self):
        inference = self.inference_me()
        self.total_dis = 0
        self.num_collision = 0
        self.cum_reward = 0

        self._agent_location = np.array([self.size/3, self.size/3])
        self.prev_action = np.array([self.size/3, self.size/3])
        self.t = 0
        trajectory_line = read_trajectory(self.trajectory_file, self.t + self.episode)
        self._target_location = trajectory_line
        self.episode += 26

        observation = self._get_obs()
        self.found_targets = []
        self.hit_obstacle = False

        return observation

    def get_exp4_action(self, action):
        """
        Method for Exp4 to provide an action to the environment
        
        Args:
            action (np.ndarray): Action from Exp4
        
        Returns:
            float: Reward for the action
        """
        # Perform step with the given action
        _, reward, terminated, _, _ = self.step(action)
        
        # Store the last reward for potential use
        self.last_reward = reward
        
        return reward

    def step(self, action, policy):
        self.obstacles = load_obstacles(self.envs[policy])
        # Clip and process action
        angle = action[0]
        action[1] += 0.1
        direction = np.array([
            (action[1] * np.sin(angle) + self.prev_action[0]) % (self.size-1),
            (action[1] * np.cos(angle) + self.prev_action[1]) % (self.size-1)
        ])
        self._agent_location = np.clip(direction.copy(), 0, self.size - 1)
        self.total_dis += np.abs(action[1])
        
        # Update trajectory
        trajectory_line = read_trajectory(self.trajectory_file, self.episode + self.t)
        self._target_location = trajectory_line
        
        # Calculate reward
        reward, found_targets_list = self.get_reward()
        self.found_targets.extend([target for target in found_targets_list if target not in self.found_targets])
        
        # Additional penalty for large movement
        if calculate_distance(self.prev_action, direction.copy()) > 30:
            reward = -40

        # Obstacle collision check
        for o in range(len(self.obstacles)):
            obstacle = self.obstacles[o]
            if obstacle['type'] == 'circle':
                center = [obstacle['x'], obstacle['y']]
                distance = calculate_distance(center, self._agent_location)
                if distance < obstacle['radius']:
                    reward = -10
                    self.hit_obstacle = True
                    self.num_collision += 1
            elif obstacle['type'] == 'rectangle':
                rect_x = obstacle['x']
                rect_y = obstacle['y']
                rect_width = obstacle['width']
                rect_height = obstacle['height']
                agent_x = self._agent_location[0]
                agent_y = self._agent_location[1]
                if rect_x <= agent_x <= rect_x + rect_width and rect_y <= agent_y <= rect_y + rect_height:
                    reward = -10
                    self.hit_obstacle = True
                    self.num_collision += 1
        
        # Termination conditions
        terminated = False
        self.cum_reward += reward
        if len(self.found_targets) == len(self._target_location) or self.hit_obstacle:
            terminated = True

        if terminated: 
            self.hit_obstacle = False

        self.prev_action = direction.copy()
        state = self._get_obs()
        self.t += 1

        return state, reward, terminated, False, self._target_location, self.inference_me()

    def get_reward(self, threshold_distance=0.001):
        found_targets_list = []
        reward = 0
        for target_idx, target_position in enumerate(self._target_location):
            dist = calculate_distance(self._agent_location, target_position)
            if target_idx not in self.found_targets:
                detect = detection_probability(dist, target_position)
                reward += -(35 - detect)
                if dist < 3: 
                    self.found_targets.append(target_idx)
                    found_targets_list.append(target_idx)

        return reward, found_targets_list

    def get_parameters(self):
        return self.action_space.shape, self.state.reshape(-1).shape, self.action_space.low, self.action_space.high

    def inference_me(self):
        return self.total_dis, len(self.found_targets), self.num_collision, self.t, self.cum_reward

    def render(self):
        # Rendering logic remains the same as in the original implementation
        pass

    def close(self):
        # Close rendering resources
        pass