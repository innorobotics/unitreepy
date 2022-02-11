import numpy as np
# Number of motors
NUM_MOTORS = 12
# Number of legs
NUM_LEGS = 4

# //////
#  Legs
# //////

LEG_NAMES = ["FR",  # Front Right
             "FL",  # Front Left
             "RR",  # Rear Right
             "RL"]  # Rear Left

# //////////////
#  Joint Types:
# //////////////

JOINT_TYPES = [0,  # Hip
               1,  # Thigh
               2]  # Knee

# ///////////////
#  Joint Mapping
# ///////////////
#  Joint names are given by concatenation of
#  LEG_NAME + JOINT_TYPE as in following table
#
# ______| Front Right | Front Left | Rear Right | Rear Left
# Hip   | FR_0 = 0    | FL_0 = 3   | RR_0 = 6   | RL_0 = 9
# Thigh | FR_1 = 1    | FL_1 = 4   | RR_1 = 7   | RL_1 = 10
# Knee  | FR_2 = 2    | FL_2 = 5   | RR_2 = 8   | RL_2 = 11


# ///////////////
# Joint Constants 
# ///////////////

JOINT_LIMITS_MIN = 4*[-0.802, -1.05, -2.7]
JOINT_LIMITS_MAX = 4*[0.802, 4.19, 0.916]

TORQUE_LIMITS_MIN = -4*[-10, -10, -10]
TORQUE_LIMITS_MAX = 4*[10, 10, 10]

# Motor angles in initial stand pose
STAND_ANGLES = 4*[0.0, 0.8, -1.65]

# Motor angles of initialization pose
INIT_ANGLES = [-0.25,  1.14, -2.72,  # FR
               0.25,  1.14, -2.72,  # FL
               -0.25,  1.14, -2.72,  # RR
               0.25,  1.14, -2.72]  # RL

POSITION_GAINS = 4*[100, 100, 100]

DAMPING_GAINS = 4*[1, 2, 2]

# ////////////////////

# TODO: Add high level commands scaling factors


# ////////////////////
# KINEMATIC_PARAMETERS
# ////////////////////

TRUNK_LENGTH = 0.1805 * 2
TRUNK_WIDTH = 0.047 * 2

LEGS_BASES = {'FR': [TRUNK_LENGTH/2, -TRUNK_WIDTH/2],
              'FL': [TRUNK_LENGTH/2, TRUNK_WIDTH/2],
              'RR': [-TRUNK_LENGTH/2, -TRUNK_WIDTH/2],
              'RL': [-TRUNK_LENGTH/2, TRUNK_WIDTH/2]}

LEG_LINKS_LENGTH = {'FR': [0.0838, 0.2, 0.2], # Here the sign of the first offset is OK with 'leg_kinematics' but not clear at first 
                    'FL': [-0.0838, 0.2, 0.2],
                    'RR': [0.0838, 0.2, 0.2],
                    'RL': [-0.0838, 0.2, 0.2]}

# LEG_DYNAMICS =
# BODY_DYNAMICS =
BODY_MASS = None
BODY_INERTIA = None
MOTOR_INERTIAS = None
MOTOR_DAMPING = None


COM_OFFSET = -np.array([-0.0165, 0.0008, 0.0]) #decrease x to shift forward, increase y to shift right
HIP_OFFSETS = np.array([[0.1805, -0.047, 0.], [0.1805, 0.047, 0.],
                        [-0.1805, -0.047, 0.], [-0.1805, 0.047, 0.]
                        ]) + COM_OFFSET

MOTOR_DIRECTION = np.ones(12)


# MAKE THINGS CLEAR

LEG_LENGTH = [0.0838, 0.2, 0.2] # [hip y-axis offset, thigh length, calf length]
BASE_TO_HIPS = np.array([[0.1805, -0.047, 0.], [0.1805, 0.047, 0.],
                        [-0.1805, -0.047, 0.], [-0.1805, 0.047, 0.]
                        ])
COM_TO_HIPS = np.array([[0.1805, -0.047, 0.], [0.1805, 0.047, 0.],
                        [-0.1805, -0.047, 0.], [-0.1805, 0.047, 0.]
                        ])+ COM_OFFSET
ANGLE_DIRECTION = np.ones((4,3)) # +1 means rotation is around original X or Y axes

#original motion imitation
MPC_BODY_MASS = 108 / 9.8
MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))

# This is actually default foot positions in robot CS
DEFAULT_HIP_POSITIONS = (
    (0.17, -0.135, 0),
    (0.17, 0.135, 0),
    (-0.195, -0.135, 0),
    (-0.195, 0.135, 0),
)

'''
MPC_BODY_MASS = 13.52
MPC_BODY_INERTIA = np.array((0.032, 0, 0, 0, 0.283, 0, 0, 0, 0.308))
_DEFAULT_HIP_POSITIONS = (
    (0.18, -0.14, 0),
    (0.18, 0.14, 0),
    (-0.18, -0.14, 0),
    (-0.18, 0.14, 0),
)

_DEFAULT_HIP_POSITIONS = (
    (0.1805, -0.047, 0),
    (0.1805, 0.047, 0),
    (-0.1805, -0.047, 0),
    (-0.1805, 0.047, 0),
)

'''

KPS = POSITION_GAINS
KDS = DAMPING_GAINS

MPC_BODY_HEIGHT = 0.24
MPC_VELOCITY_MULTIPLIER = 0.5
FOOT_FORCE_THRESHOLD = 10