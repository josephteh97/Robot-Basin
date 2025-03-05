import numpy as np
import pybullet as p
from scripts.utils import *
import time
from tqdm import tqdm
from trac_ik_python.trac_ik import IK

# Color codes
YELLOW = "\033[93m"
RESET = "\033[0m"

# Option
np.random.seed(69)
useRealTimeSimulation = 0
useSequentialControl = 0

# Connect pybullet physics server
pc = PybulletClient(connection="GUI", real_time=useRealTimeSimulation)
# time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]
plane_id, robot_id = pc.load_assets(urdf_list=["plane.urdf", "models/ec66/urdf/ec66.urdf"], fixedBaseIdx=[1])
# Set robot init config
pc.set_init_config(objectId=robot_id, init_config="camera") # init_config="camera"/"tool"
# Set robot to control
rc = RobotControl(robot_id, useSequentialControl=useSequentialControl)

def update_random():
    if rc.check_ik_run_completion(print_result=True):
        success = rc.set_random_target() # Set next target
        if not success:
            return False
    else:
        rc.update_motion_control()
    return True

def update_trajectory(hard_follow=True):
    # Soft follow not supporting sequential motion
    if not hard_follow:
        rc.update_end_effector_config(plot_trajectory=True)
        success = rc.step_trajectory(plot_trajectory=True)
        if not success:
            return False
    elif rc.check_ik_run_completion():
        success = rc.step_trajectory(plot_trajectory=True)
        if not success:
            return False
    else:
        rc.update_motion_control()
    return True

print("\n--------------------------------------------------------------------\n")
print("Start simulation...")

# Set camera focus
pc.focus_object(robot_id)
# rc.set_random_target() # Set initial target

# Define circular motion parameters
center = [-0.5, 0, 0.4]  # Circle center in world frame
radius = 0.25
num_points = 100
# Generate circular trajectory (XZ-plane)
theta = np.linspace(0, 2*np.pi, num_points)
circle_points = [
    [center[0], center[1] + radius * np.cos(t), center[2] + radius * np.sin(t)]
    for t in theta
]
pos_list = circle_points*2
orn_list = [p.getQuaternionFromEuler([0, -np.pi, 0])]*200
rc.set_trajectory(pos_list, orn_list)
rc.step_trajectory()

print("Simulation steps:")
for step in tqdm(range(2400)):
    if not useRealTimeSimulation:
        p.stepSimulation()
    time.sleep(1/240) # 1/240 is equal as real-time running, lower sleep time speed up sim
    # End simulation if collision occur
    if pc.check_collision():
        break
    success = update_trajectory(hard_follow=False)
    # End if no valid solution
    if not success:
        break

print("\nSimulation ends.")

# p.disconnect()

# # For GUI Inspection, cross GUI or press Ctrl+C to terminate
while True:
    pass