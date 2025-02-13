import isaacgym
from isaacgym import gymapi
import matplotlib.pyplot as plt
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

# Load asset
asset_root = "./environments"
robot_asset_file = "ec63.xml"
basin_asset_file = "washbasin.xml"
robot_asset_options = gymapi.AssetOptions()
basin_asset_options = gymapi.AssetOptions()

print("Loading assets...")
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, robot_asset_options)
if robot_asset is None:
    raise Exception(f"Failed to load robot asset from {robot_asset_file}")

basin_asset = gym.load_asset(sim, asset_root, basin_asset_file, basin_asset_options)
if basin_asset is None:
    raise Exception(f"Failed to load washbasin asset from {basin_asset_file}")
print("Assets loaded!")

# Set up the environment
print("Setting up the environments...")
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 2)
robot_handle = gym.create_actor(env, robot_asset, gymapi.Transform(), "robot", 0, 1)
if robot_handle is None:
    raise Exception("Failed to create robot actor")
basin_handle = gym.create_actor(env, basin_asset, gymapi.Transform(), "washbasin", 0, 1)
if basin_handle is None:
    raise Exception("Failed to create washbasin actor")
print("Envionment is successfully setup!")


# Create a viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")
print("Viewer is created!")

# Simulation loop
for _ in range(10):  # Run for 100 timesteps
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step the graphics
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.draw_viewer(viewer, sim, True)
    
    
    # Render the viewer
    gym.sync_frame_time(sim)

    if gym.query_viewer_has_closed(viewer):
        break


# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)