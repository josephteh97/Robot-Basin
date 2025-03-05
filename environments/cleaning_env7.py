import isaacgym
from isaacgym import gymapi
import numpy as np

# Initialize the gym
gym = gymapi.acquire_gym()

# Create a simulation
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.physx.num_threads = 4
sim_params.physx.solver_type = 1
sim_params.use_gpu_pipeline = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create simulation")

# Load assets
asset_root = "./environments"
robot_asset_file = "washbasin.xml"
asset = gym.load_asset(sim, asset_root, robot_asset_file)

# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# Add an asset instance to the environment
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)
robot_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

# Create a viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)