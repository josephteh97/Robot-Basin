import numpy as np
from modern_robotics import se3ToVec, MatrixLog6, TransInv, FKinBody, JacobianBody

'''
This module consists of modified function IKinBody() from modern robotics and example usage.
Modification: Hardcoded "maxiterations" --> Variable "maxiterations" takes value from argument "max_iter"
'''

def IKinBody(Blist, M, T, thetalist0, eomg=0.1, ev=0.01, max_iter=20):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = max_iter
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(JacobianBody(Blist, \
                                                         thetalist)), Vb)
        i = i + 1
        Vb \
        = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                       thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)



'''
Example usage: 
Note: The Blist & M is for zero config of ec66

Blist = np.array([[ 0.00000000e+00,1.00000000e+00 ,0.00000000e+00, -3.30000000e-02
  , 0.00000000e+00, -8.16000000e-01],
 [ 0.00000000e+00, -3.62462919e-06, -1.00000000e+00, -9.79998887e-02,
  -8.16000000e-01,  2.95769742e-06],
 [ 0.00000000e+00 ,-3.62462919e-06, -1.00000000e+00, -9.79998812e-02,
  -3.98000017e-01,  1.44260248e-06],
 [ 0.00000000e+00, -3.62462919e-06, -1.00000000e+00, -9.79998689e-02,
  -1.52587890e-08,  5.53074521e-14],
 [ 0.00000000e+00, -1.00000000e+00,  7.34641026e-06, -8.89996437e-02,
   1.12097324e-13,  1.52587890e-08],
 [ 0.00000000e+00,  1.09581808e-05,  1.00000000e+00,  2.04559220e-07,
   1.52587890e-08, -1.67208569e-13]]).T

M = np.array([[1, 0, 0, 0.816],
            [0, 0, -1, 0.033],
            [0, 1, 0, -0.002],
            [0, 0, 0, 1]])

T = transformations.euler_matrix(0, -1.12, 0)
T[:3, 3] = target_position

current_joint_angles = [p.getJointState(targid, i)[0] for i in range(num_joints)]
thetalist, success = IKinBody(Blist=Blist, M=M, T=T, thetalist0=current_joint_angles, max_iter=20)
thetalist = normalize_theta(thetalist)
'''