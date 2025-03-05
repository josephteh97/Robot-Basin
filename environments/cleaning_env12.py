"""
Kuka bin performance test
-------------------------------
Test simulation performance and stability of the robotic arm dealing with a set of complex objects in a bin.
"""

from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

axes_geom = gymutil.AxesGeometry(0.1)

sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]

tray_color = gymapi.Vec3(0.24, 0.35, 0.8)
banana_color = gymapi.Vec3(0.85, 0.88, 0.2)
brick_color = gymapi.Vec3(0.9, 0.5, 0.1)

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Kuka Bin Test",
    custom_parameters=[
        {"name": "--object_type", "type": int, "default": 0, "help": "Type of objects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

num_objects = 10
box_size = 0.05

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
asset_root = "./assets"

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

bin_pose = gymapi.Transform()
bin_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

object_pose = gymapi.Transform()

table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# load assets of objects in a bin
asset_options.fix_base_link = False

can_asset_file = "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf"
banana_asset_file = "urdf/ycb/011_banana/011_banana.urdf"
mug_asset_file = "urdf/ycb/025_mug/025_mug.urdf"
brick_asset_file = "urdf/ycb/061_foam_brick/061_foam_brick.urdf"

object_files = [can_asset_file, banana_asset_file, mug_asset_file, brick_asset_file]

object_assets = []
object_assets.append(gym.create_box(sim, box_size, box_size, box_size, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, can_asset_file, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, banana_asset_file, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, mug_asset_file, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, brick_asset_file, asset_options))

spawn_height = gymapi.Vec3(0.0, 0.3, 0.0)

# load bin asset
bin_asset_file = "urdf/tray/traybox.urdf"

print("Loading asset '%s' from '%s'" % (bin_asset_file, asset_root))
bin_asset = gym.load_asset(sim, asset_root, bin_asset_file, asset_options)

corner = table_pose.p - table_dims * 0.5

kuka_asset_file = "urdf/kuka_allegro_description/kuka_allegro.urdf"

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True

print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)

# set up the env grid
env_lower = gymapi.Vec3(-1.5, 0.0, -1.5)
env_upper = gymapi.Vec3(1.5, 1.5, 1.5)

# create env
env = gym.create_env(sim, env_lower, env_upper, 1)

table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

x = corner.x + table_dims.x * 0.5
y = table_dims.y + box_size + 0.01
z = corner.z + table_dims.z * 0.5

bin_pose.p = gymapi.Vec3(x, y, z)
tray_handle = gym.create_actor(env, bin_asset, bin_pose, "bin", 0, 0)
gym.set_rigid_body_color(env, tray_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

for j in range(num_objects):
    x = corner.x + table_dims.x * 0.5 + np.random.rand() * 0.35 - 0.2
    y = table_dims.y + box_size * 1.2 * j - 0.05
    z = corner.z + table_dims.z * 0.5 + np.random.rand() * 0.3 - 0.15

    object_pose.p = gymapi.Vec3(x, y, z) + spawn_height

    object_asset = object_assets[0]
    if args.object_type >= 5:
        object_asset = object_assets[np.random.randint(len(object_assets))]
    else:
        object_asset = object_assets[args.object_type]

    object_handle = gym.create_actor(env, object_asset, object_pose, "object" + str(j), 0, 0)
    gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, colors[j % len(colors)])

# add kuka
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