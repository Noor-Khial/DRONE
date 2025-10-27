import numpy as np
import tensorflow as tf

class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, 
           actor_model, critic_model, target_actor, target_critic, 
               critic_optimizer, actor_optimizer, gamma, summary_writer, step):

            # Critic update
            with tf.GradientTape() as tape:
                target_actions = target_actor(next_state_batch, training=True)
                y = reward_batch + gamma * target_critic(
                    [next_state_batch, target_actions], training=True
                )
                critic_value = critic_model([state_batch, action_batch], training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

            # Actor update
            with tf.GradientTape() as tape:
                actions = actor_model(state_batch, training=True)
                critic_value = critic_model([state_batch, actions], training=True)
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

            # Debugging metrics
            actions = actor_model(state_batch)
            q_values = critic_model([state_batch, action_batch])

            action_mean = tf.reduce_mean(actions)
            action_std = tf.math.reduce_std(actions)
            q_mean = tf.reduce_mean(q_values)
            q_min = tf.reduce_min(q_values)
            q_max = tf.reduce_max(q_values)
            actor_grad_norm = tf.linalg.global_norm(actor_grad)
            critic_grad_norm = tf.linalg.global_norm(critic_grad)

            # Print debugging information
            #print(f"Step {step}:")
            #print(f"  Action distribution - Mean: {action_mean.numpy():.4f}, Std: {action_std.numpy():.4f}")
            #print(f"  Q-value stats - Mean: {q_mean.numpy():.4f}, Min: {q_min.numpy():.4f}, Max: {q_max.numpy():.4f}")
            #print(f"  Gradient norms - Actor: {actor_grad_norm.numpy():.4f}, Critic: {critic_grad_norm.numpy():.4f}")
            #print(f"  Losses - Actor: {actor_loss.numpy():.4f}, Critic: {critic_loss.numpy():.4f}")

            # Log to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('critic_loss', critic_loss, step=step)
                tf.summary.scalar('actor_loss', actor_loss, step=step)
                tf.summary.scalar('action_mean', action_mean, step=step)
                tf.summary.scalar('action_std', action_std, step=step)
                tf.summary.scalar('q_value_mean', q_mean, step=step)
                tf.summary.scalar('q_value_min', q_min, step=step)
                tf.summary.scalar('q_value_max', q_max, step=step)
                tf.summary.scalar('actor_gradient_norm', actor_grad_norm, step=step)
                tf.summary.scalar('critic_gradient_norm', critic_grad_norm, step=step)

                # log histograms for more detailed distribution information
                tf.summary.histogram('actions', actions, step=step)
                tf.summary.histogram('q_values', q_values, step=step)

            return critic_loss, actor_loss
    
    def learn(self, actor_model, critic_model, target_actor, target_critic, 
              critic_optimizer, actor_optimizer, gamma, summary_writer, step):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        critic_loss, actor_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch, 
                                              actor_model, critic_model, target_actor, target_critic, 
                                              critic_optimizer, actor_optimizer, gamma, summary_writer, step)

        return critic_loss, actor_loss


#tensorboard --logdir=/path/to/your/logs
