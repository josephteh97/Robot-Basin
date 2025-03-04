import gym
import isaacgym
import torch, torchvision
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from environments.cleaning_env16 import EC66CleaningEnv

# Create and normalize the environment
env = make_vec_env(EC66CleaningEnv, n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Define the PPO model with custom hyperparameters
model = PPO(
    'MlpPolicy', 
    env, 
    device='cpu',
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    n_epochs=10
)

# Callbacks for checkpointing and evaluation
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/')
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10000)

# Train the model with TensorBoard logging
try:
    model.learn(total_timesteps=100000, callback=[checkpoint_callback, eval_callback], tb_log_name="ppo_cleaning_robot")
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save("trained_models/ppo_cleaning_robot_interrupted")
finally:
    env.close()

# Save the final model
model.save("PPO_weights/ppo_cleaning_robot")

# Evaluate the model
eval_env = EC66CleaningEnv()
obs, _ = eval_env.reset()
print("size fo obs: ", type(obs), len(obs))
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()
    if done:
        obs, _ = eval_env.reset()
eval_env.close()
