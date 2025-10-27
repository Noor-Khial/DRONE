import gym
import logging
import sys
import os
sys.path.append(os.path.abspath('.'))
from env.gym_env import GridWorldEnv


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def main():
    env = GridWorldEnv(render_mode='human', size=5)
    observation = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, done, _, _ = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    main()
