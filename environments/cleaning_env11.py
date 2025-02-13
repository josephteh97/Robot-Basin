import pybullet as p
import pybullet_data
import time
import random

# Connect to PyBullet
p.connect(p.GUI)

# Set the gravity
p.setGravity(0, 0, -9.8)

# Load the plane
plane_id = p.loadURDF("plane.urdf")

# Load the URDF asset
asset_root = "./environments"
asset_file = "panda.urdf"  # Assuming the file is converted to URDF format
asset_path = asset_root + '/' + asset_file

# Define asset options
flags = p.URDF_USE_IMPLICIT_CYLINDER
asset_id = p.loadURDF(asset_path, [0, 0, 0], useFixedBase=True, flags=flags)
print("Assets Loaded!")

# Define environment creation parameters
spacing = 2.0
env_lower = [-spacing, 0.0, -spacing]
env_upper = [spacing, spacing, spacing]
num_envs = 1

# Create and populate the environments
envs = []
actor_ids = []

for i in range(num_envs):
    height = random.uniform(1.0, 2.5)
    env_position = [0.0, height, 0.0]
    envs.append(env_position)

    actor_id = p.loadURDF(asset_path, env_position, useFixedBase=True, flags=flags)
    actor_ids.append(actor_id)

# Configure camera properties
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])

# Simulation loop
while True:
    # Step the simulation
    p.stepSimulation()
    
    # Update the viewer (PyBullet does this automatically)
    time.sleep(1 / 60.0)  # Synchronize with the rendering rate

# Cleanup (PyBullet handles cleanup automatically on disconnect)
p.disconnect()