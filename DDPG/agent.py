import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime

class RLAgent:
    def __init__(self, num_actions, num_states, lower_bound, upper_bound, log_dir=None):
        # Environment setup
        #self.env = env
        #num_actions, num_states, lower_bound, upper_bound = env.get_parameters()
        self.num_actions = num_actions[0]
        self.num_states = num_states[0]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Noise setup
        self.ou_noise = self._create_ou_noise()

        # Model initialization
        self.actor_model = self._get_actor()
        self.critic_model = self._get_critic()
        self.target_actor = self._get_actor()
        self.target_critic = self._get_critic()

        # Initialize target networks with same weights
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Optimizer setup
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.buffer = self._create_buffer()

        # Logging setup
        self.log_dir = log_dir
        self.summary_writer = self._setup_tensorboard() if log_dir else None

        # Training tracking
        self.ep_reward_list = []
        self.avg_reward_list = []

    def _get_actor(self):
        from models import get_actor
        return get_actor(self.num_states, self.upper_bound)

    def _get_critic(self):
        from models import get_critic
        return get_critic(self.num_states, self.num_actions)

    def _create_buffer(self):
        from buffer import Buffer
        return Buffer(self.num_states, self.num_actions, 20000, 64)

    def _setup_tensorboard(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir_tensorboard = os.path.join(self.log_dir, current_time)
        return tf.summary.create_file_writer(log_dir_tensorboard)

    def get_action(self, state):
        from utils import policy_saria
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = policy_saria(tf_state, 0, self.actor_model, self.lower_bound, self.upper_bound)
        return action[0]

    def update(self, prev_state, action, reward, next_state):
        # Record experience in tables, and charts) enhances comprehension and effectively supports the narrative.buffer
        self.buffer.record((prev_state, action, reward, next_state))

        # Learn from buffer
        critic_loss, actor_loss = self.buffer.learn(
            self.actor_model, 
            self.critic_model, 
            self.target_actor, 
            self.target_critic, 
            self.critic_optimizer, 
            self.actor_optimizer, 
            self.gamma, 
            self.summary_writer, 
            0  # step placeholder
        )

        # Update target networks
        from utils import update_target
        update_target(self.target_actor, self.actor_model, self.tau)
        update_target(self.target_critic, self.critic_model, self.tau)

        return critic_loss, actor_loss

    def receive_reward(self, reward):
        # This method can be used to track rewards from Exp4
        self.ep_reward_list.append(reward)
        return np.mean(self.ep_reward_list[-40:]) if self.ep_reward_list else reward

    def save_models(self, log_dir):
        self.actor_model.save(os.path.join(log_dir, 'actor_model'))
        self.critic_model.save(os.path.join(log_dir, 'critic_model'))