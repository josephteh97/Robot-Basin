import math
import numpy as np
import isaacgym
import torch
import torchvision
from isaacgym import gymapi, gymutil
import gym
from gym import spaces
import trimesh

class EC66CleaningEnv(gym.Env):
    def __init__(self):
        super(EC66CleaningEnv, self).__init__()
        
        # Initialize Isaac Gym components
        self.gym = gymapi.acquire_gym()
        self.args = gymutil.parse_arguments(description="EC66 Elite Robot and Table Cleaning Task")
        self.sim = self._create_sim()
        self.env = self._create_environment()
        self.robot_handle, self.dof_props, self.num_dofs = self._load_robot()
        self.washbasin_handle = self._load_washbasin()
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "brush")
        self.attractor_handle = self._create_attractor()
        self.mesh = trimesh.load("./assets/urdf/washbasin/washbasin.obj")
        self.total_area = self.mesh.area
        self.cleaned_area = 0.0

        
        # RL setup
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,),  # X, Y, Z attractor position deltas
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),     #shape=(3*3 + 1,),  # attractor_pos, ee_pos, target_pos, contact_status
            dtype=np.float32
        )
        
        # Washbasin parameters
        self.washbasin_center = np.array([0.4, 0.4, 0.0])
        self.valid_radius = 0.3
        self.contact_threshold = 0.05  # 5cm proximity considered as contact
        
        # Training parameters
        self.max_steps = 500
        self.current_step = 0
        self.last_attractor_pos = None
        self.viewer = None

    def _compute_cleaned_area(self, ee_pos):
        cleaned_area = 0.0
        return cleaned_area

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0/20.0
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

    def _load_washbasin(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        
        washbasin_asset = self.gym.create_box(
            self.sim, 
            0.8, 0.8, 0.2,  # Dimensions for washbasin
            asset_options
        )
        
        washbasin_pose = gymapi.Transform()
        washbasin_pose.p = gymapi.Vec3(0.4, 0.4, 0.0)
        return self.gym.create_actor(
            self.env, 
            washbasin_asset, 
            washbasin_pose, 
            "Washbasin", 
            0, 
            1
        )

    def _create_attractor(self):
        attractor_props = gymapi.AttractorProperties()
        attractor_props.stiffness = 5e5
        attractor_props.damping = 5e3
        attractor_props.axes = gymapi.AXIS_ALL
        attractor_props.rigid_handle = self.hand_handle
        
        # Initial target position
        initial_target = gymapi.Transform()
        initial_target.p = gymapi.Vec3(0.4, 0.6, 0.0)
        attractor_props.target = initial_target
        
        return self.gym.create_rigid_body_attractor(self.env, attractor_props)

    def reset(self, seed=None, options=None):       
        # Reset attractor position
        initial_target = gymapi.Transform()
        initial_target.p = gymapi.Vec3(0.4, 0.6, 0.0)
        self.gym.set_attractor_target(self.env, self.attractor_handle, initial_target)

        # Initialize position history with numpy arrays
        self.last_attractor_pos = np.array([initial_target.p.x, initial_target.p.y, initial_target.p.z]) # Convert initial position to numpy array
        self.second_last_attractor_pos = self.last_attractor_pos.copy()

        # Reset other states
        self.current_step = 0
        self.cleaned_area = 0.0
        
        # Reset simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Get attractor position
        attractor_props = self.gym.get_attractor_properties(self.env, self.attractor_handle)
        attractor_pos = np.array([attractor_props.target.p.x, 
                                 attractor_props.target.p.y, 
                                 attractor_props.target.p.z], dtype=np.float32)
        
        # Get end-effector position
        ee_pose = self.gym.get_rigid_transform(self.env, self.hand_handle)
        ee_pos = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z], dtype=np.float32)
        
        # Get contact status
        contact_status = 1.0 if self._check_contact() else -1.0

        obs = np.concatenate([attractor_pos, ee_pos, self.washbasin_center, [contact_status]])
        assert obs.shape == (10,), f"Invalid observation shape: {obs.shape}"    # Debug check
        # print("Observation shape: ", obs.shape)       # Debug line
        
        return obs

    def step(self, action):
        
        # Update attractor position
        current_attractor = self.gym.get_attractor_properties(self.env, self.attractor_handle).target
        new_pos = gymapi.Vec3(
            current_attractor.p.x + action[0]*0.05,
            current_attractor.p.y + action[1]*0.05,
            current_attractor.p.z + action[2]*0.05
        )
        self.gym.set_attractor_target(self.env, self.attractor_handle, gymapi.Transform(p=new_pos))

        # Convert new position to numpy array and update history
        new_pos_np = np.array([new_pos.x, new_pos.y, new_pos.z])
        self.second_last_attractor_pos = self.last_attractor_pos.copy()     # Store as numpy array
        self.last_attractor_pos = new_pos_np


        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.current_step += 1
        
        # Get observations and compute reward
        obs = self._get_obs()
        reward, done = self._compute_reward(obs)
        
        # Check termination
        truncated = self.current_step >= self.max_steps
        return obs, reward, done, truncated, {}

    def _compute_reward(self, obs):
        attractor_pos = obs[:3]   # numpy array (first 3 index selected)
        ee_pos = obs[3:6]
        contact_status = obs[-1]

        # Check if position history is initialized
        if self.last_attractor_pos is None or self.second_last_attractor_pos is None:
            return 0.0, False
        
        movement = np.linalg.norm(attractor_pos - self.last_attractor_pos)
        
        # Reward components
        basin_distance = np.linalg.norm(attractor_pos - self.washbasin_center)
        ee_basin_distance = np.linalg.norm(ee_pos - self.washbasin_center)
        
        # 1. Attractor proximity reward
        proximity_reward = max(0, (self.valid_radius - basin_distance)/self.valid_radius)
        
        # 2. Contact reward
        contact_reward = 2.0 if contact_status > 0 else -1.0
        
        # 3. Movement penalties
        smoothness_penalty = 0.0
        step_size_penalty = 0.0
        if self.last_attractor_pos is not None:
            movement = np.linalg.norm(attractor_pos - self.last_attractor_pos)
            
            # Quadratic penalty for sudden movements
            smoothness_penalty = 0.2 * movement**2
            
            # Linear penalty with minimum -1 for large steps
            step_size_penalty = -min(movement * 20, 1.0)  # 20=1/0.05 (5cm max allowed step)
            
            # Additional penalty for directional changes
            if self.current_step > 1:
                prev_vector = self.last_attractor_pos - self.second_last_attractor_pos
                current_vector = attractor_pos - self.last_attractor_pos
                direction_change = np.arccos(
                    np.dot(prev_vector, current_vector) / 
                    (np.linalg.norm(prev_vector) * np.linalg.norm(current_vector) + 1e-8))
                smoothness_penalty += 0.5 * direction_change
        
        # 4. Washbasin Cleaning Effectiveness
        if contact_status > 0: # Only update cleaned area if there is contact
            self.cleaned_area += self._compute_cleaned_area(ee_pos) # Update cleaned area
        cleaning_ratio = self.cleaned_area / self.total_area
        cleaning_reward = 5.0 * cleaning_ratio  # Scale the reward based on cleaning ratio
                
        
        
        # Total reward calculation
        total_reward = (
            5.0 * proximity_reward +
            3.0 * contact_reward +
            2.0 * cleaning_reward -
            smoothness_penalty +
            step_size_penalty -  # Already negative value
            0.1 * basin_distance
        )
        
        # Update movement history
        self.second_last_attractor_pos = self.last_attractor_pos.copy() if self.last_attractor_pos is not None else None
        self.last_attractor_pos = attractor_pos.copy()
        
        # Termination conditions
        done = (
            basin_distance > self.valid_radius * 1.5 or 
            self.current_step >= self.max_steps
        )
        
        return total_reward, done


    def _check_contact(self):
        # Get all contacts in the environment
        contacts = self.gym.get_env_rigid_contacts(self.env)
        
        # Check if any contact involves the brush and washbasin
        for contact in contacts:
            body0 = contact['body0']
            body1 = contact['body1']
            if (body0 == self.hand_handle and body1 == self.washbasin_handle) or \
               (body1 == self.hand_handle and body0 == self.washbasin_handle):
                return True
        return False

    def render(self, mode='human'):
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
    id='EC66Cleaning-v1',
    entry_point=EC66CleaningEnv
)
