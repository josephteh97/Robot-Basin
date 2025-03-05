import math
import numpy as np
from isaacgym import gymapi, gymutil
import gym
from gym import spaces

class EC66CleaningEnv(gym.Env):
    def __init__(self):
        super(EC66CleaningEnv, self).__init__()

        # Initialize Isaac Gym components
        self.gym = gymapi.acquire_gym()
        self.args = gymutil.parse_arguments()
        self.sim = self._create_sim()
        self.env = self._create_environment()
        self.robot_handle, self.dof_props, self.num_dofs = self._load_robot()
        self.target_points = self._define_target_points()
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "brush")
        
        # RL setup
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_dofs,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(2*self.num_dofs + 6,),  # dof_pos + dof_vel + ee_pos + target_pos
            dtype=np.float32
        )
        
        self.current_target_idx = 0
        self.max_steps = 500
        self.current_step = 0
        self.viewer = None

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1/60.0
        sim_params.substeps = 2
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = self.args.use_gpu
        sim_params.use_gpu_pipeline = False
        
        sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )
        if not sim:
            raise RuntimeError("Failed to create simulation")
        return sim

    def _create_environment(self):
        self.gym.add_ground(self.sim, gymapi.PlaneParams())
        return self.gym.create_env(self.sim, 
                                 gymapi.Vec3(-1.5, 0.0, -1.5), 
                                 gymapi.Vec3(1.5, 1.5, 1.5), 
                                 1)

    def _load_robot(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        
        robot_asset = self.gym.load_asset(
            self.sim, 
            "./assets", 
            "urdf/ec66/ec66.urdf", 
            asset_options
        )
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.1, 0.3, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        
        robot_handle = self.gym.create_actor(
            self.env, 
            robot_asset, 
            pose, 
            "EC66", 
            0, 
            1
        )
        
        dof_props = self.gym.get_actor_dof_properties(self.env, robot_handle)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props['stiffness'].fill(100.0)
        dof_props['damping'].fill(1000.0)
        self.gym.set_actor_dof_properties(self.env, robot_handle, dof_props)
        
        return robot_handle, dof_props, self.gym.get_actor_dof_count(self.env, robot_handle)

    def _define_target_points(self):
        return [
            gymapi.Vec3(0.4, 0.6, 0.0),
            gymapi.Vec3(0.4, 0.5, 0.0),
            gymapi.Vec3(0.6, 0.5, -0.1),
            gymapi.Vec3(0.6, 0.5, -0.2),
            gymapi.Vec3(0.5, 0.5, -0.3),
            gymapi.Vec3(0.3, 0.5, -0.3),
            gymapi.Vec3(0.2, 0.5, -0.2),
            gymapi.Vec3(0.2, 0.5, 0.2),
            gymapi.Vec3(0.3, 0.5, 0.3),
            gymapi.Vec3(0.5, 0.5, 0.3),
            gymapi.Vec3(0.6, 0.5, 0.2),
            gymapi.Vec3(0.6, 0.5, 0.1)
        ]

    def reset(self, seed=None):
        self.current_target_idx = 0
        self.current_step = 0
        
        # Reset robot state
        self.gym.set_actor_dof_position_targets(
            self.env, 
            self.robot_handle, 
            [0.0]*self.num_dofs
        )
        
        # Reset simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Get joint states
        dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
        dof_pos = np.array([s['pos'] for s in dof_states], dtype=np.float32)
        dof_vel = np.array([s['vel'] for s in dof_states], dtype=np.float32)
        
        # Get end-effector position
        ee_pose = self.gym.get_rigid_transform(self.env, self.hand_handle)
        ee_pos = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z], dtype=np.float32)
        
        # Get current target
        target = self.target_points[self.current_target_idx]
        target_pos = np.array([target.x, target.y, target.z], dtype=np.float32)
        
        return np.concatenate([dof_pos, dof_vel, ee_pos, target_pos])

    def step(self, action):
        # Convert action to joint targets
        scaled_action = self._scale_action(action)
        self.gym.set_actor_dof_position_targets(
            self.env, 
            self.robot_handle, 
            scaled_action
        )
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.current_step += 1
        
        # Get observations
        obs = self._get_obs()
        
        # Calculate reward
        reward, done = self._compute_reward(obs[-6:-3], obs[-3:])
        
        # Check termination conditions
        truncated = self.current_step >= self.max_steps
        return obs, reward, done, truncated, {}

    def _scale_action(self, action):
        scaled = []
        for i in range(self.num_dofs):
            lower = self.dof_props['lower'][i]
            upper = self.dof_props['upper'][i]
            scaled.append(lower + (action[i] + 1) * (upper - lower) / 2)
        return scaled

    def _compute_reward(self, ee_pos, target_pos):
        distance = np.linalg.norm(ee_pos - target_pos)
        reward = -distance  # Base reward
        
        if distance < 0.05:
            reward += 10
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.target_points):
                return reward, True
        return reward, False

    def render(self):
        if not self.viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(-0.8, 2.0, -1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def close(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_env(self.env)
        self.gym.destroy_sim(self.sim)

# Register the environment
gym.envs.registration.register(
    id='EC66Cleaning-v0',
    entry_point=EC66CleaningEnv
)
