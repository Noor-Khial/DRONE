import gym
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath('.'))
from env.main import GridWorldEnv

class DDPGPolicy:
    def __init__(self, model_path, obstacles_file):
        """Initialize the inference environment and model.
        
        Args:
            model_path (str): Path to the trained actor model.
            obstacles_file (str): Path to the obstacles configuration file.
        """
        self.env = GridWorldEnv(render_mode="human", size=55)
        self.obstacles_file = obstacles_file

        # Load environment parameters
        self.num_actions, self.num_states, self.lower_bound, self.upper_bound = self.env.get_parameters()
        self.num_actions = self.num_actions[0]
        self.num_states = self.num_states[0]

        # Load the trained actor model
        self.actor_model = tf.keras.models.load_model(model_path)

        # Metrics
        self.ep_rewards = []
        self.total_distance = 0
        self.targets_found = 0
        self.collisions = 0
        self.time_steps = 0
        self.cum_reward = 0

    def run(self, prev_state, policy, num_episodes=20):
        """Run the inference loop for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to run inference.
        """
        for episode in range(num_episodes):
            #prev_state, inference = self.env.reset()
            #self.total_distance += inference[0]
            #self.targets_found += inference[1]
            #self.collisions += inference[2]
            #self.time_steps += inference[3]
            #self.cum_reward += inference[4]

            episode_reward = 0
            done = False

            for t in range(26):
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = self.actor_model.predict(tf_prev_state)
                state, reward, done, truncated, _, metrics = self.env.step(action[0], policy)
                episode_reward += reward
                prev_state = state

                if done or truncated:
                    break

            #print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Time = {t}")
            self.ep_rewards.append(episode_reward)

            return state, episode_reward, t, metrics


        ## Print aggregated metrics
        #self._print_metrics(num_episodes)

    def _print_metrics(self, num_episodes):
        """Print aggregated performance metrics.
        
        Args:
            num_episodes (int): Number of episodes for averaging.
        """
        #print(f"Average Distance: {self.total_distance / num_episodes}")
        #print(f"Average Targets Found: {self.targets_found / num_episodes}")
        #print(f"Average Collisions: {self.collisions / num_episodes}")
        #print(f"Average Time Steps: {self.time_steps / num_episodes}")
        #print(f"Average Cumulative Reward: {self.cum_reward / num_episodes}")
        return self.total_distance / num_episodes, self.targets_found / num_episodes, self.collisions / num_episodes, self.time_steps / num_episodes, self.cum_reward / num_episodes

# Example usage:
# action = DDPGPolicy(model_path='DDPG/exp-a-2/env1', obstacles_file='experiments/exp-a/env1.json')
# action.run(num_episodes=1)
