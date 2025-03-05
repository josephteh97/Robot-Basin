import gym
from gym import spaces
import numpy as np
import math
from math import cos, sin
import random
# import cv2
import mujoco_py

# Constants
FOV = 60
CAMERA_START_POSE = [0, 0, 1]
TARGET_START_POSE = [5, 0, 0.48]
HALF_VIEW_RANGE = abs(TARGET_START_POSE[0] - CAMERA_START_POSE[0]) * math.tan(math.pi / (180 / (FOV / 2)))
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
NUM_FRAMES = 1
NUM_CHANNELS_BINARY = 1
NUM_CHANNELS_RGB = 4
AMPLITUDE = 5
NUM_STEPS = 300
STEP = 2 * math.pi / NUM_STEPS
DELTA_ANGLE = 2 * np.pi / NUM_STEPS
NUM_ACTIONS = 3

class CleaningRobotEnv(gym.Env):
    def __init__(self):
        super(CleaningRobotEnv, self).__init__()
        
        # Define the action space: 0 - no change, 1 - clockwise, 2 - counterclockwise
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # Define the observation space as an image
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES), 
                                            dtype=np.uint8)
        
        # Load the Mujoco model
        self.model = mujoco_py.load_model_from_path("path_to_mujoco_model.xml")
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
        
        # Camera settings
        self.fov = FOV
        self.aspect = 1.0
        self.near = 0.01
        self.far = 10
        self.camera_pos = CAMERA_START_POSE.copy()
        self.target_pos = TARGET_START_POSE.copy()
        self.up_vec = [0, 0, 1]
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT
        self.num_frames = NUM_FRAMES
        self.delta_angle = DELTA_ANGLE
        
        # Robot settings
        self.robot_start_pos = np.array([0, 0, 0])
        self.robot_current_step = 0
        self.robot_current_pose = TARGET_START_POSE.copy()
        self.direction = 0
        self.is_binary = True
        self.out_of_frame = False
        
        # Initial reset
        self.reset()

    def get_image(self):
        self.viewer.render(width=self.width, height=self.height)
        rgb_img = self.viewer.read_pixels(self.width, self.height, depth=False)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        if self.is_binary:
            return binary_img
        else:
            return rgb_img

    def get_reward(self):
        current_robot_pos = self.sim.data.get_body_xpos('robot')
        frame = self.is_point_in_fov(self.camera_pos, self.target_pos, current_robot_pos)
        if frame == 0:
            return 40
        elif frame == 1:
            return 10
        elif frame == 2:
            return 0
        else:
            self.out_of_frame = True
            return -1000

    def get_observation(self):
        if self.is_binary:
            observation = np.zeros((self.height, self.width, self.num_frames), dtype=np.uint8)
            for i in range(self.num_frames):
                observation[:, :, i] = np.array(self.get_image())
            return observation
        else:
            observation = np.zeros((self.height, self.width, self.num_channels, self.num_frames), dtype=np.uint8)
            for i in range(self.num_frames):
                observation[:, :, :, i] = np.array(self.get_image())
            return observation

    def is_point_in_fov(self, camera_pos, camera_dir, point_pos, fov=FOV):
        rel_pos = (point_pos[0] - camera_pos[0], point_pos[1] - camera_pos[1], point_pos[2] - camera_pos[2])
        dot_prod = rel_pos[0] * camera_dir[0] + rel_pos[1] * camera_dir[1] + rel_pos[2] * camera_dir[2]
        mag_dir = math.sqrt(camera_dir[0]**2 + camera_dir[1]**2 + camera_dir[2]**2)
        mag_rel = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
        angle = math.acos(dot_prod / (mag_dir * mag_rel))
        if np.rad2deg(angle) <= fov / 10:
            return 0
        elif np.rad2deg(angle) <= fov / 4:
            return 1
        elif np.rad2deg(angle) <= fov / 2:
            return 2
        else:
            return 3

    def reset(self):
        self.sim.reset()
        self.total_reward = 0
        self.robot_current_step = 0
        self.robot_current_pose = TARGET_START_POSE.copy()
        self.camera_pos = CAMERA_START_POSE.copy()
        self.target_pos = self.robot_current_pose
        observation = self.get_observation()
        return np.array(observation)

    def move_robot(self):
        if self.direction % 50 == 0:
            if random.randint(0, 1):
                global STEP
                STEP = STEP * -1
        self.direction += 1
        self.robot_current_step += STEP
        self.robot_current_pose[0] = AMPLITUDE * math.cos(self.robot_current_step)
        self.robot_current_pose[1] = AMPLITUDE * math.sin(self.robot_current_step)
        self.sim.data.set_joint_qpos('robot', self.robot_current_pose)

    def step(self, action):
        x1 = self.target_pos[0]
        y1 = self.target_pos[1]
        if action == 0:
            self.change = 0
        elif action == 1:
            self.change = 1
        elif action == 2:
            self.change = -1
        theta = self.change * self.delta_angle
        x2 = cos(theta) * x1 - sin(theta) * y1
        y2 = sin(theta) * x1 + cos(theta) * y1
        self.target_pos = [x2, y2, 0.5]
        observation = self.get_observation()
        reward = self.get_reward()
        self.total_reward += reward
        done = self.total_reward < 0 or self.total_reward > 4000 or self.out_of_frame
        self.out_of_frame = False
        self.move_robot()
        return np.array(observation), reward, done

# Register the environment
gym.envs.registration.register(id='CleaningRobot-v0', entry_point='cleaning_robot_env:CleaningRobotEnv')