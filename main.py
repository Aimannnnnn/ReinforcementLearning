import os
import gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environtment_name = 'CartPole-v0'
env = gym.make(environtment_name, render_mode='human')

episodes = 11
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        nstate, reward, done, truncated, info = env.step(action)
        score += reward
    print('Episode: {} Score: {}'.format(episodes, score))
env.close()