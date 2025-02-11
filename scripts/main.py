import gym
from stable_baselines3 import PPO
from cleaning_env import CleaningEnv
import os

# Directory to save trained models and logs
MODEL_DIR = "trained_models"
LOG_DIR = "logs"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_cleaning_robot")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_model():
    # Create the environment
    env = CleaningEnv()

    # Define the PPO model
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save(MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")

def evaluate_model():
    # Load the trained model
    model = PPO.load(MODEL_PATH)

    # Create the environment
    env = CleaningEnv()

    # Evaluate the model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

if __name__ == "__main__":
    choice = input("Do you want to train (T) or evaluate (E) the model? ").strip().upper()
    if choice == 'T':
        train_model()
    elif choice == 'E':
        evaluate_model()
    else:
        print("Invalid choice. Please enter 'T' to train or 'E' to evaluate.")