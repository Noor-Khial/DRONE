

import gym
import numpy as np
import sys 
import os 
sys.path.append(os.path.abspath('.'))


from env.main import GridWorldEnv
import tensorflow as tf
# Load the trained DDPG model
# Example:
# from your_module import DDPGModel
# model = DDPGModel.load("path_to_trained_model")

# Run the inference loop

class Policy(object):
     
     def __init__(self, obstacle_file, actor_model,  env_size= 55):
          self.env = GridWorldEnv(obstacle_file, env_size)
          num_actions, num_states, lower_bound, upper_bound = self.env.get_parameters()
          self.num_actions = num_actions[0]
          self.num_states = num_states[0]
          self.actor_model =  tf.keras.models.load_model(actor_model)
          self.prev_state = self.env.reset()
        
    
     def step(self):
          tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_state), 0)
          action = self.actor_model.predict(tf_prev_state)
          state, reward, done, truncated, _ = self.env.step(action[0])
          cost = self.env.get_cost()
          self.prev_state = state
          if done: 
               cost = 0 
          return cost