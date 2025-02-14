"""
Kuka robot and table setup
-------------------------------
Test simulation performance and stability of the robotic arm with just a table.
"""

from __future__ import print_function, division, absolute_import

import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Kuka Robot and Table Test")

# configure sim
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 25
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = False
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "./environments"

table_dims = gymapi.Vec3(0.6, 0.4, 1.0)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.7, 0.5 * table_dims.y + 0.001, 0.0)

table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

kuka_asset_file = "kuka_allegro.urdf"

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True

print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)
print("Asset loaded!")

# set up the env grid
env_lower = gymapi.Vec3(-1.5, 0.0, -1.5)
env_upper = gymapi.Vec3(1.5, 1.5, 1.5)

# create env
env = gym.create_env(sim, env_lower, env_upper, 1)

# create table
table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

# add kuka robot
kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", 0, 1)

# get joint limits and ranges for kuka
kuka_dof_props = gym.get_actor_dof_properties(env, kuka_handle)
kuka_lower_limits = kuka_dof_props['lower']
kuka_upper_limits = kuka_dof_props['upper']
kuka_ranges = kuka_upper_limits - kuka_lower_limits
kuka_mids = 0.5 * (kuka_upper_limits + kuka_lower_limits)
kuka_num_dofs = len(kuka_dof_props)

# override default stiffness and damping values
kuka_dof_props['stiffness'].fill(100.0)
kuka_dof_props['damping'].fill(100.0)

# Set base to track pose zero to maintain posture
kuka_dof_props["driveMode"][0] = gymapi.DOF_MODE_POS

gym.set_actor_dof_properties(env, kuka_handle, kuka_dof_props)

# set camera to focus on the robot and the table
cam_pos = gymapi.Vec3(2.0, 2.0, 3.0)
cam_target = gymapi.Vec3(0.3, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# a helper function to initialize the env
def init():
    # set updated stiffness and damping properties
    gym.set_actor_dof_properties(env, kuka_handle, kuka_dof_props)

    kuka_dof_states = gym.get_actor_dof_states(env, kuka_handle, gymapi.STATE_NONE)
    for j in range(kuka_num_dofs):
        kuka_dof_states['pos'][j] = kuka_mids[j] - kuka_mids[j] * .5
    gym.set_actor_dof_states(env, kuka_handle, kuka_dof_states, gymapi.STATE_POS)

def update_kuka(t):
    gym.clear_lines(viewer)

init()

next_kuka_update_time = 0.1
frame = 0

while not gym.query_viewer_has_closed(viewer):
    # check if we should update
    t = gym.get_sim_time(sim)
    if t >= next_kuka_update_time:
        update_kuka(t)
        next_kuka_update_time += 0.01

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

    frame = frame + 1

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)