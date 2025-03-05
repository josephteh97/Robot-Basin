import gym
from stable_baselines3 import PPO
from environments.cleaning_env3 import CleaningEnv

# Create the environment
env = CleaningEnv()

# Define the PPO model
model = PPO('PPOPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("trained_models/ppo_cleaning_robot")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
