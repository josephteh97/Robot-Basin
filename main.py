import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import transformations
from scripts.utils import IKinBody, normalize_theta, quaternion_distance, plot_frame
import time
from tqdm import tqdm

np.random.seed(90)

physicsClientId = p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

print("\n--------------------------------------------------------------------\n")
connectionInfo = p.getConnectionInfo(physicsClientId)
print(f"Physics Client Id: {physicsClientId}")
print(f"Connected: {connectionInfo['isConnected']}, Connection: {connectionInfo['connectionMethod']}\n")

# load assets
print("Loading assets...")
p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
targid = p.loadURDF("models/ec66/urdf/ec66.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
print("Assets Loaded!")
obj_of_focus = targid

# Set end effector pose and orientation
num_joints = p.getNumJoints(targid)
# for i in range(num_joints):
#     p.setJointMotorControl2(targid, i, p.POSITION_CONTROL, force=0)
end_effector_link = num_joints - 1
# Generate random position within the given ranges
target_position = [
    np.random.choice([np.random.uniform(-0.6, -0.2), np.random.uniform(0.2, 0.6)]),
    np.random.choice([np.random.uniform(-0.6, -0.2), np.random.uniform(0.2, 0.6)]),
    np.random.uniform(0.2, 0.6)
]
# Generate random roll, pitch, yaw within -π to π
random_rpy = np.random.uniform(-np.pi, np.pi, 3)
# Convert RPY to quaternion
target_orientation = p.getQuaternionFromEuler(random_rpy)
plot_frame(target_position, target_orientation, axis_length=0.2)
thetalist = p.calculateInverseKinematics(targid, end_effector_link, target_position, target_orientation)

# Blist = np.array([[ 0.00000000e+00,1.00000000e+00 ,0.00000000e+00, -3.30000000e-02
#   , 0.00000000e+00, -8.16000000e-01],
#  [ 0.00000000e+00, -3.62462919e-06, -1.00000000e+00, -9.79998887e-02,
#   -8.16000000e-01,  2.95769742e-06],
#  [ 0.00000000e+00 ,-3.62462919e-06, -1.00000000e+00, -9.79998812e-02,
#   -3.98000017e-01,  1.44260248e-06],
#  [ 0.00000000e+00, -3.62462919e-06, -1.00000000e+00, -9.79998689e-02,
#   -1.52587890e-08,  5.53074521e-14],
#  [ 0.00000000e+00, -1.00000000e+00,  7.34641026e-06, -8.89996437e-02,
#    1.12097324e-13,  1.52587890e-08],
#  [ 0.00000000e+00,  1.09581808e-05,  1.00000000e+00,  2.04559220e-07,
#    1.52587890e-08, -1.67208569e-13]]).T

# M = np.array([[1, 0, 0, 0.816],
#             [0, 0, -1, 0.033],
#             [0, 1, 0, -0.002],
#             [0, 0, 0, 1]])

# T = transformations.euler_matrix(*random_rpy)
# T[:3, 3] = target_position

# current_joint_angles = [p.getJointState(targid, i)[0] for i in range(num_joints)]
# thetalist, success = IKinBody(Blist=Blist, M=M, T=T, thetalist0=current_joint_angles, max_iter=20)
# thetalist = normalize_theta(thetalist)
# print(f"thetalist: {thetalist}, success: {success}")
p.setJointMotorControlArray(targid, list(range(num_joints)), p.POSITION_CONTROL, targetPositions=thetalist)

print("Start simulation...")
focus_position, _ = p.getBasePositionAndOrientation(targid)
p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focus_position)
for step in tqdm(range(200)):
    p.stepSimulation()
    time.sleep(.01)

link_state = p.getLinkState(targid, end_effector_link)
end_effector_position = link_state[4]
end_effector_orientation = link_state[5]
plot_frame(end_effector_position, end_effector_orientation, axis_length=0.2)
time.sleep(3)
print(f"Target position: {target_position}")
print(f"End effector position (xyz, m): {end_effector_position}")
print(f"Normalized position error (m): {np.linalg.norm(np.array(end_effector_position) - np.array(target_position))}")
print(f"Target orientation: {target_orientation}")
print(f"End effector orientation (rpy, rad): {end_effector_orientation}")
print(f"Normalized orientation error (rad): {quaternion_distance(end_effector_orientation, target_orientation)}")
