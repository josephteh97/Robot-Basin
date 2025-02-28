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
np.random.seed(1)
useRealTimeSimulation = 0
useSequentialControl = 0

# Connect pybullet physics server
pc = PybulletClient(connection="GUI", real_time=useRealTimeSimulation)
plane_id, robot_id = pc.load_assets(urdf_list=["plane.urdf", "models/ec66/urdf/ec66.urdf"], fixedBaseIdx=[1])
# Set robot init config
pc.set_init_config(objectId=robot_id, init_config="camera") # init_config="camera"/"tool"
# Set robot to control
rc = RobotControl(robot_id, useSequentialControl=useSequentialControl)

def update():
    if rc.check_ik_run_completion(print_result=True):
        success = rc.set_random_target() # Set next target
        if not success:
            return False
    else:
        rc.update_motion_control()
    return True

print("\n--------------------------------------------------------------------\n")
print("Start simulation...")

# Set camera focus
pc.focus_object(robot_id)
rc.set_random_target() # Set initial target

print("Simulation steps:")
for step in tqdm(range(2400)):
    if not useRealTimeSimulation:
        p.stepSimulation()
    time.sleep(1/240) # This is equal as real-time running, lower sleep time speed up sim
    success = update()
    # End if no valid solution
    if not success:
        break
    # End simulation if collision occur
    if pc.check_collision(robot_id, plane_id):
        break

print("\nSimulation ends.")



p.disconnect()

# # For GUI Inspection, cross GUI or press Ctrl+C to terminate
# while True:
#     pass