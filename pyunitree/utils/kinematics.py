import numpy as np
import math
from pyunitree.robots.a1.constants import HIP_OFFSETS,MOTOR_DIRECTION
from pyunitree.robots.a1.constants import LEG_LENGTH, BASE_TO_HIPS, COM_TO_HIPS

def leg_kinematics(motor_angles, link_lengths, base_position):

    q1, q2, q3 = motor_angles
    r0_x, r0_y = base_position
    l1, l2, l3 = link_lengths

    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)

    # calculate the position of the foot
    position = np.array([-l2*s2 - l3*s23 + r0_x,
                      -l1*c1 + (l2*c2 + l3*c23)*s1 + r0_y,
                      -l1*s1 - (l2*c2 + l3*c23)*c1])

    # jacobian of the foot position with respect to
    jacobian = np.array([[0, -l2*c2 - l3*c23, -l3*c23],
                      [l1*s1 + (l2*c2 + l3*c23)*c1, -
                          (l2*s2 + l3*s23)*s1, -l3*s1*s23],
                      [-l1*c1 + (l2*c2 + l3*c23)*s1, (l2*s2 + l3*s23)*c1, l3*s23*c1]])

    rotation_matrix = np.array([[c23, s1*s23, -s23*c1],
                             [0, c1, s1],
                             [s23, -s1*c23, c1*c23]])

    return position, jacobian, rotation_matrix

def QuaternionToEulerMatrix(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return  2*np.array([[ 0.5-y**2-z**2, x*y+w*z  , x*z-w*y],
                        [ x*y-w*z  , 0.5-x**2-z**2, y*z+w*x],
                        [ x*z+w*y  , y*z-w*x  , 0.5-x**2-y**2]])

def EulerFromQuaternion(quat):
        x,y,z,w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z]

#HIP_COEFFICIENT = 0.08505 #original motion imitation
HIP_COEFFICIENT = 0.0838
def FootPositionInHipFrame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]

    l_up = 0.2
    l_low = 0.2
    l_hip = HIP_COEFFICIENT * ((-1)**(l_hip_sign + 1))

    leg_distance = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    theta_ab_cos = np.cos(theta_ab)
    theta_ab_sin = np.sin(theta_ab)
    
    off_x = off_x_hip
    off_y = theta_ab_cos * off_y_hip - theta_ab_sin * off_z_hip
    off_z = theta_ab_sin * off_y_hip + theta_ab_cos * off_z_hip
    return [off_x, off_y, off_z]
    
def FootPositionInHipFrameToJointAngle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = HIP_COEFFICIENT * ((-1)**(l_hip_sign + 1))
    x, y, z = foot_position[0], foot_position[1], foot_position[2]

    theta_knee = -math.acos(
        (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
        (2 * l_low * l_up))

    l = math.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))

    theta_hip = math.asin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * math.cos(theta_hip + theta_knee / 2) * z
    s1 = l * math.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = math.atan2(s1, c1)
    return [theta_ab, theta_hip, theta_knee]

def ComputeMotorAnglesFromFootLocalPosition(leg_id,
                                                foot_local_position):
                                                
        joint_position_idxs = list(range(leg_id * 3,leg_id * 3 + 3))

        try:
            joint_angles = FootPositionInHipFrameToJointAngle(
                foot_local_position - HIP_OFFSETS[leg_id],
                l_hip_sign=leg_id)
        except:
            joint_angles = [math.nan,math.nan,math.nan]
        joint_angles = joint_angles*MOTOR_DIRECTION[joint_position_idxs]

        return joint_position_idxs, joint_angles.tolist()

def AnalyticalLegJacobian(leg_angles, sign):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = HIP_COEFFICIENT* (-1)**(sign + 1)
    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
        t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J

def CompactAnalyticalJacobian(leg_angles, sign):
    """
    Computes the analytical Jacobian in a single vector
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = HIP_COEFFICIENT * (-1)**(sign + 1)
    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros(9)
    J[0] = 0
    J[1] = -l_eff * np.cos(t_eff)
    J[2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
        t_eff) / 2
    J[3] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[4] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[5] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[6] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[7] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[8] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


    # MAKE THINGS CLEAR

    """
    There are the following CSs in robot:
    Base Frame is located in the center of the robot with x axis directed forward and z axis directed upward
    Hip Frame is located in the hips of the robot
    COM Frame is located in Center of mass/
    All three CSs are different only in TRANSLATION

    Angles are in radians with order [hip joint, thigh joint, calf joint]
    leg_id are in range(4) for [FR LR RR RL] 
    """

def AnglesFromPositionInHipFrame(leg_id, foot_position):
    l_up = LEG_LENGTH[1]
    l_low = LEG_LENGTH[2]
    l_hip = LEG_LENGTH[0] * ((-1)**(leg_id + 1))
    x, y, z = foot_position[0], foot_position[1], foot_position[2]

    # Here we have two solutions for knee joint and choose the one with negative sign (more natural)
    theta_knee = -math.acos(
        (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
        (2 * l_low * l_up))

    l = math.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))

    theta_hip = math.asin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * math.cos(theta_hip + theta_knee / 2) * z
    s1 = l * math.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = math.atan2(s1, c1)
    return [theta_ab, theta_hip, theta_knee]

def PositionInHipFrameFromAngles(leg_id, angles):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]

    l_up = LEG_LENGTH[1]
    l_low = LEG_LENGTH[2]
    l_hip = LEG_LENGTH[0] * ((-1)**(leg_id + 1))

    leg_distance = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    theta_ab_cos = np.cos(theta_ab)
    theta_ab_sin = np.sin(theta_ab)
    
    off_x = off_x_hip
    off_y = theta_ab_cos * off_y_hip - theta_ab_sin * off_z_hip
    off_z = theta_ab_sin * off_y_hip + theta_ab_cos * off_z_hip
    return [off_x, off_y, off_z]


# print(AnglesFromPositionInHipFrame(1,PositionInHipFrameFromAngles(1,[0.15,0.22,0.35])))