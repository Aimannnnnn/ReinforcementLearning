import os
import gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#Creare l'environment
environtment_name = 'CartPole-v0'
env = gym.make(environtment_name, render_mode='human')


##Basi => creare il ciclo per allenare il modello + render dell'environment
# episodes = 5
# for episodes in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         nstate, reward, done, truncated, info = env.step(action)
#         score += reward
#     print('Episode: {} Score: {}'.format(episodes, score))

# Definire directory dove salvare log e il modello in sè
log_path = os.path.join('Training', 'Logs')
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')

# env = DummyVecEnv([lambda: env])
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# #Allenare il modello
# model.learn(total_timesteps=20000) 
# #Salvare il modello
# model.save(PPO_Path)

#Caricare il modello
model = PPO.load(PPO_Path, env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# episodes = 5
# for episodes in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         nstate, reward, done, truncated, info = env.step(action)
#         score += reward
#     print('Episode: {} Score: {}'.format(episodes, score))