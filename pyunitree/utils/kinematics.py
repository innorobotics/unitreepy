import numpy as np
import math

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

def EulerFromQuaternion(quat):
        x,y,z,w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z]

def RotFromQuaternion(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    return np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])

def FootPositionInHipFrame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * ((-1)**(l_hip_sign + 1))
    leg_distance = np.sqrt(l_up**2 + l_low**2 +
                            2 * l_up * l_low * math.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * math.sin(eff_swing)
    off_z_hip = -leg_distance * math.cos(eff_swing)
    off_y_hip = l_hip

    theta_ab_cos = math.cos(theta_ab)
    theta_ab_sin = math.sin(theta_ab)
    
    off_x = off_x_hip
    off_y = theta_ab_cos * off_y_hip - theta_ab_sin * off_z_hip
    off_z = theta_ab_sin * off_y_hip + theta_ab_cos * off_z_hip
    return [off_x, off_y, off_z]

def FootPositionInHipFrameToJointAngle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * ((-1)**(l_hip_sign + 1))
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
        (2 * l_low * l_up))
    l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])

def AnalyticalLegJacobian(leg_angles, sign):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(sign + 1)

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
