import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco.viewer

class CleaningEnv(gym.Env):
    """MuJoCo environment for a robot cleaning a face-washing bowl"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, xml_path="environments/ec63.xml"):
        super(CleaningEnv, self).__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Action Space (Joint control: 6-DOF arm)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation Space (Robot joint states + bowl cleanliness)
        obs_dim = self.model.nq + self.model.nv + 1  # Joint positions, velocities, cleanliness score
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Viewer (Optional for debugging)
        self.viewer = None

    def step(self, action):
        """Apply action and update the simulation"""
        # Apply action to control robot joints
        self.data.ctrl[:] = action  # Directly setting joint torques
        mujoco.mj_step(self.model, self.data)

        # Compute reward (cleanliness improvement)
        prev_cleanliness = self.data.qpos[-1]  # Assume cleanliness is the last state variable
        new_cleanliness = prev_cleanliness + np.random.uniform(0, 0.05)  # Simulate cleaning effect
        self.data.qpos[-1] = max(0, min(1, new_cleanliness))  # Normalize to [0,1]

        reward = new_cleanliness - prev_cleanliness  # Reward is improvement in cleanliness

        # Check if cleaning is done (threshold reached)
        done = new_cleanliness >= 0.95

        # Observation: Robot state + cleanliness score
        observation = np.concatenate([self.data.qpos, self.data.qvel, [self.data.qpos[-1]]])

        return observation, reward, done, {}

    def reset(self):
        """Reset the environment for a new episode"""
        mujoco.mj_resetData(self.model, self.data)  # Reset simulation state
        self.data.qpos[-1] = 0.0  # Reset cleanliness score to 0
        return np.concatenate([self.data.qpos, self.data.qvel, [self.data.qpos[-1]]])

    def render(self, mode="human"):
        """Render the MuJoCo simulation"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if mode == "rgb_array":
            width, height = 640, 480
            data = mujoco.mj_render(self.model, self.data, width, height)
            return data
        elif mode == "human":
            mujoco.viewer.render(self.viewer)

    def close(self):
        """Close the simulation viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Register the custom environment
gym.envs.registration.register(
    id='CleaningEnv-v0',
    entry_point='custom_cleaning_env:CleaningEnv',
)

# Create the custom environment
env = gym.make('CleaningEnv-v0')
env.reset()

# Render as image array
env.render(mode="rgb_array")