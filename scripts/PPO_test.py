import mujoco_py
import numpy as np
from stable_baselines3 import PPO
import time
import argparse

class MujocoTester:
    def __init__(self, model_path, env):
        self.model_path = model_path
        self.env = env
        self.model = PPO.load(model_path)

    def test(self, num_episodes=10, render=True):
        total_rewards = []
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                if render:
                    self.env.render()
                    time.sleep(0.01)
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
        return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the performance of the trained PPO model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained PPO model')
    parser.add_argument('--env_path', type=str, required=True, help='Path to the MuJoCo model')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to test')
    parser.add_argument('--render', type=bool, default=True, help='Render the environment during testing')

    args = parser.parse_args()

    # Load the MuJoCo environment
    model_path = args.env_path
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    class CustomEnv:
        def __init__(self, sim, viewer):
            self.sim = sim
            self.viewer = viewer
        
        def reset(self):
            self.sim.reset()
            self.sim.step()
            return self.sim.get_state()
        
        def step(self, action):
            self.sim.data.ctrl[:] = action
            self.sim.step()
            obs = self.sim.get_state()
            reward = self.calculate_reward()
            done = self.check_done()
            info = {}
            return obs, reward, done, info
        
        def render(self):
            self.viewer.render()
        
        def calculate_reward(self):
            # Define your reward calculation logic here
            return 0
        
        def check_done(self):
            # Define your done condition here
            return False

    env = CustomEnv(sim, viewer)
    
    tester = MujocoTester(args.model_path, env)
    tester.test(num_episodes=args.num_episodes, render=args.render)