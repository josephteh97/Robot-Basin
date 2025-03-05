import numpy as np
import pybullet as p
import pybullet_data
import time
from trac_ik_python.trac_ik import IK

# Color codes
YELLOW = "\033[93m"
RESET = "\033[0m"

def normalize_theta(theta_list):
    # Normalize thetalist into [-np.pi, np.pi]
    return [(theta + np.pi) % (2 * np.pi) - np.pi for theta in theta_list]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def quaternion_distance(q1, q2):
    # Calculate angle between two quaternions

    q1 = q1 / np.linalg.norm(q1) # Normalize
    q2 = q2 / np.linalg.norm(q2)
    # Compute the dot product between the quaternions
    dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)  # Clamp to avoid floating-point issues
    # Compute the angle between the quaternions
    angle = 2 * np.arccos(np.abs(dot_product))
    return angle

# __________________________________________________________________________________________________
# Below only for pybullet simulation
# __________________________________________________________________________________________________

class PybulletClient():
    # Class to control robot in pybullet
    def __init__(self, connection="DIRECT", real_time = False, verbose=True):
        VALID_CONNECTIONS = ["DIRECT", "GUI"]
        if connection.upper() in VALID_CONNECTIONS:
            self.connection = connection
        else:
            print(f"{YELLOW}[WARNING] '{connection}' is invalid pybullet physics server connection. \
                  Use 'DIRECT' or 'GUI'. This session will be connected using 'DIRECT' (headless).{RESET}")
            self.connection_mode = "DIRECT"
        self.real_time = real_time
        self.verbose = verbose

        self.connect_server()

        camera_config = [0.0, -150*np.pi/180, 108*np.pi/180, -50*np.pi/180, np.pi/2, np.pi/4]
        tool_config = [66.002*np.pi/180, -111.693*np.pi/180, 85.751*np.pi/180, -65.06*np.pi/180, 90.002*np.pi/180, 21*np.pi/180]
        self.EC66CONFIG = {"camera": camera_config, "tool": tool_config}

    def connect_server(self):
        # Connect to pybullet physics server
        if self.connection == "DIRECT":
            physicsClientId = p.connect(p.DIRECT)
        else:
            physicsClientId = p.connect(p.GUI)
        # Initialize
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        if self.real_time:
            p.setRealTimeSimulation(1)
        else:
            p.setRealTimeSimulation(0)
        # Opitional print
        if self.verbose:
            print("\n--------------------------------------------------------------------\n")
            connectionInfo = p.getConnectionInfo(physicsClientId)
            print(f"Physics Client Id: {physicsClientId}")
            print(f"Connected: {connectionInfo['isConnected']}, Connection: {connectionInfo['connectionMethod']}\n")
            print("--------------------------------------------------------------------")
        return physicsClientId

    def load_assets(self, urdf_list=[], basePos=None, baseOri=None, fixedBaseIdx=[]):
        if basePos is None:
            basePos = [[0, 0, 0] for _ in range(len(urdf_list))]
        elif len(basePos) != len(urdf_list):
            raise ValueError("Arguments 'urdf_list' & 'basePos' must be of same length!")
        if baseOri is None:
            baseOri = [[0, 0, 0, 1] for _ in range(len(urdf_list))]
        elif len(baseOri) != len(urdf_list):
            raise ValueError("Arguments 'urdf_list' & 'baseOri' must be of same length!")
        
        objectId = []
        if self.verbose:
            print("Loading assets...")
        for i in range(len(urdf_list)):
            if i in fixedBaseIdx:
                objectId.append(p.loadURDF(urdf_list[i], basePos[i], baseOri[i], useFixedBase=True))
            else:
                objectId.append(p.loadURDF(urdf_list[i], basePos[i], baseOri[i], useFixedBase=True))
        if self.verbose:
            print("Assets Loaded!")
        return objectId

    def set_init_config(self, objectId, init_config=None):
        # Set initial configuration
        num_joints = p.getNumJoints(objectId)
        joint_indices = list(range(num_joints))
        if init_config is None:
            init_config = [0.0]*num_joints
        elif init_config in self.EC66CONFIG:
            init_config = self.EC66CONFIG[init_config]
        elif len(init_config) != num_joints:
            raise ValueError("Arguments 'init_config' must have same length to number of joints of 'objectId'!")
        for idx in joint_indices:
            p.resetJointState(objectId, idx, init_config[idx])
        p.setJointMotorControlArray(objectId, joint_indices, p.POSITION_CONTROL, targetPositions=init_config)
        # positionGains=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0], velocityGains=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    def focus_object(self, objectId, cameraDistance=3, cameraYaw=0, cameraPitch=-40):
        focus_position, _ = p.getBasePositionAndOrientation(objectId)
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition=focus_position)

    def check_collision(self, objectA=None, objectB=None):
        if objectA is None:
            contact_points = p.getContactPoints()
        elif objectB is None:  
            contact_points = p.getContactPoints(bodyA=objectA)
        else:
            contact_points = p.getContactPoints(bodyA=objectA, bodyB=objectB)    
        if contact_points:
            print("Collision detected!")
            p.addUserDebugText(text="Collision Detected!", textPosition=[-1, 0, 1], textColorRGB=[1, 0, 0], textSize=3)
            for contact in contact_points:
                print(f"Contact info: {contact}")
            return True
        return False
    
    def drawAABB(self, objectId, linkId=None):
        if linkId:
            aabb = p.getAABB(objectId, linkId)
        else:
            aabb = p.getAABB(objectId)
        aabbMin = aabb[0]
        aabbMax = aabb[1]
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMax[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMin[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 0, 0])

        return aabbMin, aabbMax

class RobotControl():
    def __init__(self, robot_id, useSequentialControl=0):
        self.id = robot_id
        self.num_joints = p.getNumJoints(self.id)
        self.target_position = None #xyz
        self.target_orientation = None # quaternion
        self.end_effector_position = None
        self.end_effector_orientation = None
        self.joint_control_sequences = {}
        self.sent_control_command = {}
        self.active_sequence_control_id = None
        self.target_joint_config = None
        self.calc_time = None
        self.frame_plot = []
        self.body_plot = []
        self.ik_start_run_time = None
        self.ik_run_time = None
        self.ik_run_complete = False
        self.useSequentialControl = useSequentialControl
        self.poserr = 1e-2
        self.orierr = 1e-2
        self.trajectory = {}
        self.trajectory_idx = 0

        # Initialize end_effector_config
        self.update_end_effector_config()
        # Initialize TRAC-IK
        self.ik_solver = IK("base_link", "wrist3Joint_Link", solve_type="Distance", urdf_string=open("models/ec66/urdf/ec66.urdf").read())
        self.ik_solver.set_joint_limits([-6.28, -1.57, -6.28, -6.28, -6.28, -6.28], [6.28, 0.0, 6.28, 6.28, 6.28, 6.28])

    def get_current_config(self):
        current_config = []
        for joint_idx in range(self.num_joints):
            joint_state = p.getJointState(self.id, joint_idx)
            current_config.append(joint_state[0]) # Current joint position
        return current_config
    
    def update_end_effector_config(self, plot_trajectory=False):
        previous_end_effector_position = self.end_effector_position
        end_effector_link_state = p.getLinkState(self.id, self.num_joints-1)
        self.end_effector_position = end_effector_link_state[4]
        self.end_effector_orientation = end_effector_link_state[5]
        if plot_trajectory and previous_end_effector_position != self.end_effector_position:
            self.plot_trajectory(previous_end_effector_position, self.end_effector_position, [1,0,0])

    def plot_frame(self, position, orientation, axis_length=0.5):
        '''
        Plot a coordinate frame to visualize the position and orientation of a point on pybullet GUI
        '''
        position = np.array(position)
        orientation = np.array(orientation)

        # Visualize position by marking a small sphere
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 0, 0, 1])
        sphere_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_visual, basePosition=position)
        self.body_plot.append(sphere_id)

        # Plot x,y,z axes to visualize orientation
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        x_axis = np.array([rot_matrix[0], rot_matrix[1], rot_matrix[2]])
        y_axis = np.array([rot_matrix[3], rot_matrix[4], rot_matrix[5]])
        z_axis = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]])
        line_x = p.addUserDebugLine(position, position + x_axis * axis_length, lineColorRGB=[1, 0, 0], lineWidth=2)  # X axis in red
        line_y = p.addUserDebugLine(position, position + y_axis * axis_length, lineColorRGB=[0, 1, 0], lineWidth=2)  # Y axis in green
        line_z = p.addUserDebugLine(position, position + z_axis * axis_length, lineColorRGB=[0, 0, 1], lineWidth=2)  # Z axis in blue
        self.frame_plot.extend([line_x, line_y, line_z])

        return sphere_id, line_x, line_y, line_z
    
    def plot_trajectory(self, previous_pos, current_pos, rgb=None):
        p.addUserDebugLine(previous_pos, current_pos, lineColorRGB=rgb)
    
    def erase_plot(self):
        if len(self.body_plot) > 0:
            for body_id in self.body_plot:
                p.removeBody(body_id)
        if len(self.frame_plot) > 0:
            for line_id in self.frame_plot:
                p.removeUserDebugItem(line_id)
        self.body_plot = []
        self.frame_plot = []

    def generate_random_target(self, plot_target=True):
        # Generate random position within the given ranges
        self.target_position = [
            np.random.choice([np.random.uniform(-0.6, -0.2), np.random.uniform(0.2, 0.6)]),
            np.random.choice([np.random.uniform(-0.6, -0.2), np.random.uniform(0.2, 0.6)]),
            np.random.uniform(0.2, 0.6)
        ]
        # Generate random roll, pitch, yaw within -π to π
        random_rpy = np.random.uniform(-np.pi, np.pi, 3)
        self.target_orientation = p.getQuaternionFromEuler(random_rpy)
        self.erase_plot() # Erase previous target plot
        if plot_target:
            self.plot_frame(self.target_position, self.target_orientation, axis_length=0.2)

    def run_ik(self):
        # Calculate ik to target position and orientation and send motor control command
        seed_state = self.get_current_config()
        start_time = time.time()
        self.target_joint_config = self.ik_solver.get_ik(
            seed_state,
            *self.target_position,
            *self.target_orientation,
        )
        self.calc_time = time.time() - start_time
        if not self.target_joint_config:
            print("No valid solution!")
            return False
        elif self.useSequentialControl:
            self.ik_start_run_time = time.time()
            self.ik_run_time = None
            self.ik_run_complete = False
            self.active_sequence_control_id = self.set_sequential_control(self.target_joint_config, ascending=False)
            self.step_sequential_control(self.active_sequence_control_id)
        else:
            self.ik_start_run_time = time.time()
            self.ik_run_time = None
            self.ik_run_complete = False
            p.setJointMotorControlArray(self.id, list(range(self.num_joints)), p.POSITION_CONTROL, targetPositions=self.target_joint_config)
        return True

    def check_ik_run_completion(self, print_result=False):
        self.update_end_effector_config()
        if self.get_position_error() < self.poserr and self.get_orientation_error() < self.orierr:
            self.ik_run_time = time.time() - self.ik_start_run_time
            self.ik_run_complete = True
            if print_result:
                # plot_frame(rc.end_effector_position, rc.end_effector_orientation, axis_length=0.2)
                print("\n--------------------------------------------------------------------")
                print(f"Calculation time: {self.calc_time}")
                print(f"Motor run time: {self.ik_run_time} (Only accurate if use real-time sim)")
                print(f"IK Solution: {[round(theta, 4) for theta in self.target_joint_config]}")
                print("--------------------------------------------------------------------")
                print(f"Target position: {[round(pos, 4) for pos in self.target_position]}")
                print(f"End effector position (xyz, m): {[round(pos, 4) for pos in self.end_effector_position]}")
                print(f"Normalized position error (m): {np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position))}")
                print("--------------------------------------------------------------------")
                print(f"Target orientation: {[round(ori, 4) for ori in self.target_orientation]}")
                print(f"End effector orientation (rpy, rad): {[round(ori, 4) for ori in self.end_effector_orientation]}")
                print(f"Normalized orientation error (rad): {quaternion_distance(self.end_effector_orientation, self.target_orientation)}")
                print("--------------------------------------------------------------------\n")
            return True
        return False

    def joint_reached_target(self, joint_id, target_angle, tolerance=0.01):
        # Check if a joint has reached its target
        current_angle = p.getJointState(self.id, joint_id)[0]  # Get current joint angle
        return abs(current_angle - target_angle) < tolerance

    def set_sequential_control(self, thetalist, joint_indices=None, ascending=True):
        # Set a sequence of positional control
        # Default joint_indices is [0, 1, 2, ...]
        if joint_indices is None:
            joint_indices = list(range(len(thetalist)))
        elif len(joint_indices) != len(thetalist):
            raise ValueError("The length of thetalist must equal to joint_indices")
        
        # Generate a unique_id for this sequence control
        while True:
            sequence_id = np.random.randint(1e6, 1e9)
            if sequence_id not in self.joint_control_sequences:
                break
        if not ascending:
            joint_indices = joint_indices[::-1]
            thetalist = thetalist[::-1]
        self.joint_control_sequences[sequence_id] = [(joint_id, theta) for joint_id, theta in zip(joint_indices, thetalist)]
        self.sent_control_command[sequence_id] = [] # initialize
        return sequence_id

    def step_sequential_control(self, sequence_id=None):
        # Send next joint position control command if previous position reached
        # If sequence_id not specified, assume only exist one sequence
        if sequence_id is None:
            if not self.joint_control_sequences:
                raise ValueError("No control sequences available")
            sequence_id = list(self.joint_control_sequences.keys())[0]
             
        control_sequence = self.joint_control_sequences[sequence_id]
        if control_sequence:
            joint_id, target_theta = control_sequence[0]
            if self.joint_reached_target(joint_id, target_theta):
                self.joint_control_sequences[sequence_id] = control_sequence[1:]
                # Send next command if there are remaining steps
                if self.joint_control_sequences[sequence_id]:
                    next_joint_id, next_target_theta = self.joint_control_sequences[sequence_id][0]
                    p.setJointMotorControl2(self.id, next_joint_id, p.POSITION_CONTROL, targetPosition=next_target_theta)
                    # Record sent command
                    self.sent_control_command[sequence_id].append((next_joint_id, next_target_theta))
            elif (joint_id, target_theta) not in self.sent_control_command[sequence_id]:
                p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, targetPosition=target_theta)
                self.sent_control_command[sequence_id].append((joint_id, target_theta))

    def update_motion_control(self):
        if self.useSequentialControl:
            self.step_sequential_control(sequence_id=self.active_sequence_control_id)

    def get_position_error(self):
        return np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position))
    
    def get_orientation_error(self):
        return quaternion_distance(self.end_effector_orientation, self.target_orientation)
    
    def set_random_target(self):
        self.generate_random_target()
        success = self.run_ik()
        if not success:
            return False
        return True

    def check_motion(self):
        for i in range(self.num_joints):
            joint_velocity = p.getJointState(self.id, i)[1]
            if joint_velocity > 1e-3:
                print(f"{YELLOW}[WARNING]: Joint {i} is still moving!   Velocity: {joint_velocity}{RESET}")

    def set_trajectory(self, pos_list, orn_list=None):
        if orn_list is None:
            orn_list = [[0, 0, 0, 1]] * len(pos_list)
        elif len(pos_list) != len(orn_list):
            raise ValueError("Length of arguments 'pos_list' & 'orn_list' must match!")
        # Set trajectory        
        self.trajectory["pos"] = pos_list
        self.trajectory["orn"] = orn_list
        self.trajectory_idx = 0

    def step_trajectory(self, plot_trajectory=False):
        if self.trajectory_idx < len(self.trajectory["pos"]):
            target_pos = self.trajectory["pos"][self.trajectory_idx]
            target_orn = self.trajectory["orn"][self.trajectory_idx]
            success = self.set_target(target_pos, target_orn)
            if not success:
                return False
            if plot_trajectory:
                if self.trajectory_idx > 0:
                    self.plot_trajectory(self.trajectory["pos"][self.trajectory_idx - 1], target_pos, [0,0,1])
            self.trajectory_idx += 1
        return True

    def set_target(self, target_pos, target_orn, plot_target=False):
        if len(target_pos) != 3:
            raise ValueError("Argument 'target_pos' must have 3 elements!")
        if len(target_orn) != 4:
            raise ValueError("Argument 'target_orn' must have 4 elements!")
        self.target_position = target_pos
        self.target_orientation = target_orn
        if plot_target:
            self.erase_plot() # erase previous target plot
            self.plot_frame(self.target_position, self.target_orientation, axis_length=0.2)
        success = self.run_ik()
        if not success:
            return False
        return True



