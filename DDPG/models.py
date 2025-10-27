import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_actor(num_states, upper_bound):
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

    #outputs =  outputs * (upper_bound)
    #outputs = -1 + outputs * (upper_bound - -1)
    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model

def get_critic(num_states, num_actions):
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = keras.Model([state_input, action_input], outputs)
    return model