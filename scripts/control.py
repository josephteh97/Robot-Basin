import numpy as np
import mujoco_py
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Transform
from threading import Thread
from tf import transformations as tft

class MujocoController:
    def __init__(self, model_path, init_sim=True):
        if init_sim:
            self.model = mujoco_py.load_model_from_path(model_path)
            self.sim = mujoco_py.MjSim(self.model)
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.max_velo = [0.25, 0.25, 0.25]
        self.max_velo_rot = 20
        self.current_velocity = [0, 0, 0, 0, 0, 0]
        self.safe_work_space = [-0.5, 0.5, -0.5, 0.5, 0, 1]
        self.desktop_depth = 0.3
        self.current_pos_base = [0, 0, 0]
        self.gp_goal = [0, 0, 0, 0]
        self.y_offset = 0.005
        self.z_offset = -0.005
        self.openloop = False
        self.out_of_range = False

        rospy.init_node('mujoco_control')
        rospy.Subscriber('/control/command', Float32MultiArray, self.command_callback)
        rospy.Subscriber('/control/joint_states', JointState, self.joint_state_callback)

        self.joint_vel_pub = rospy.Publisher('/control/joint_vel_cmd', Float32, queue_size=1)

        self.control_thread = Thread(target=self.control_loop)
        self.control_thread.setDaemon(True)
        self.control_thread.start()

    def command_callback(self, msg):
        d = list(msg.data)
        if d[2] > 0.10:  
            gp = Pose()
            gp.position.x = d[0]
            gp.position.y = d[1]
            gp.position.z = d[2]
            q = tft.quaternion_from_euler(-1 * d[3], 0, 0)
            gp.orientation.x = q[0]
            gp.orientation.y = q[1]
            gp.orientation.z = q[2]
            gp.orientation.w = q[3]

            gp_base = self.convert_pose(gp)
            gpbo = gp_base.orientation
            e = tft.euler_from_quaternion([gpbo.x, gpbo.y, gpbo.z, gpbo.w])
            av = np.array([gp_base.position.x, gp_base.position.y, gp_base.position.z, e[2]])

            gp_base.position.x = av[0]
            gp_base.position.y = av[1]
            gp_base.position.z = av[2]

            ang = av[3] - np.pi / 2
            q = tft.quaternion_from_euler(np.pi, 0, ang)
            gp_base.orientation.x = q[0]
            gp_base.orientation.y = q[1]
            gp_base.orientation.z = q[2]
            gp_base.orientation.w = q[3]

            dx = (gp_base.position.x - self.current_pos_base[0])
            dy = (gp_base.position.y - self.current_pos_base[1])
            dz = (gp_base.position.z - self.current_pos_base[2])

            vx = max(min(dx * 3, self.max_velo[0]), -1.0 * self.max_velo[0])
            vy = max(min(dy * 0.15, self.max_velo[1]), -1.0 * self.max_velo[1])
            vz = max(min(dz * 3, self.max_velo[2]), -1.0 * self.max_velo[2])
            v = np.array([vx, vy, vz])

            if not self.openloop:
                self.gp_goal[0] = gp_base.position.x
                self.gp_goal[1] = min(gp_base.position.y, self.desktop_depth)
                self.gp_goal[2] = gp_base.position.z
                self.gp_goal[3] = av[3] - 3.14 if av[3] > 0 else av[3] - 0.5

                self.current_velocity[0] = v[0]
                self.current_velocity[1] = v[1]
                self.current_velocity[2] = v[2]
                self.current_velocity[3] = 0
                self.current_velocity[4] = 0
                self.current_velocity[5] = max(min(1.5 * av[3], self.max_velo_rot), -1 * self.max_velo_rot) * 5

    def joint_state_callback(self, msg):
        self.current_pos_base = [msg.position[0], msg.position[1], msg.position[2]]

    def convert_pose(self, pose):
        # Simulated function to convert pose
        return pose

    def control_loop(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if not self.out_of_range:
                if not self.openloop:
                    self.sim.data.ctrl[:] = self.current_velocity[:6]
                else:
                    self.sim.data.ctrl[:] = [0] * 6
            else:
                self.sim.data.ctrl[:] = [0] * 6
            self.sim.step()
            self.viewer.render()
            rate.sleep()

if __name__ == "__main__":
    model_path = "path/to/your/mujoco/model.xml"
    controller = MujocoController(model_path)
    rospy.spin()