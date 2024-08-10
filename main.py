import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
import cv2

env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', max_episode_steps=5000)
env = AtariWrapper(env)
env = Monitor(env)

# Creating and training the model
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=200000)

# Evaluatign the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Record the video of the gameplay
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
env = AtariWrapper(env)
env = Monitor(env)
obs, info = env.reset()
frames = []

for _ in range(5000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    frames.append(env.render())
    if dones or truncated:
        break

env.close()

# Saving the video
video_path = 'videos/atari_demo.mp4'
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()

