from logging import info
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyunitree.utils.kinematics import FootPositionInHipFrame,EulerFromQuaternion
from pyunitree.robots.a1.constants import POSITION_GAINS,DAMPING_GAINS,HIP_OFFSETS
from pyunitree.base.daemon import SHM_IMPORTED
from pyunitree.parsers.remote import WirelessRemote


class A1SharedState:
    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))

    _DEFAULT_HIP_POSITIONS = (
        (0.17, -0.135, 0),
        (0.17, 0.13, 0),
        (-0.195, -0.135, 0),
        (-0.195, 0.13, 0),
    )
    
    KPS = POSITION_GAINS
    KDS = DAMPING_GAINS

    MPC_BODY_HEIGHT = 0.24
    MPC_VELOCITY_MULTIPLIER = 0.5
    FOOT_FORCE_THRESHOLD = 10
    
    DATA_TYPE = np.float32
    """
    self.stateVec
     quaternion  |4|
           gyro  |3|
          accel  |3|
     footforces  |4|
              q  |12|
             dq  |12|
            tick |1|
    """
    def __init__(self,stateBufferPtr=None,modelBufferPtr=None,observerBufferPtr=None,wirelessRemotePtr=None,commandPtr=None):

        if SHM_IMPORTED:
            from multiprocessing.shared_memory import SharedMemory
            self.rawStateShm = SharedMemory(name="RobotState")
            self.stateVec = np.frombuffer(self.rawStateShm.buf, dtype=self.DATA_TYPE)

            try:
                self.rawJacShm = SharedMemory(name="Model")
                self.rawJacobiansBuffer = np.frombuffer(self.rawJacShm.buf, dtype=self.DATA_TYPE)
            except FileNotFoundError:
                self.rawJacobiansBuffer = None

            try:
                self.rawVelShm = SharedMemory(name="Observer")
                self.rawVelocitiesBuffer = np.frombuffer(self.rawVelShm.buf, dtype=self.DATA_TYPE)
            except:
                self.rawVelocitiesBuffer = None

            try:
                self.rawRemoteShm = SharedMemory(name="WirelessRemote")
                self.rawRemoteBuffer =  np.frombuffer(self.rawRemoteShm.buf,dtype=np.bytes_)
            except:
                self.rawRemoteBuffer = None

            try:
                self.rawCommandShm = SharedMemory(create=True,name="Command",size=12)
                self.createdCommandShm = True
            except FileExistsError:
                self.createdCommandShm = False
                self.rawCommandShm = SharedMemory(name="Command")
            
            self.rawCommandBuffer =  np.frombuffer(self.rawCommandShm.buf,dtype=np.float32)

        else:
            if stateBufferPtr is not None:
                self.stateVec = np.frombuffer(stateBufferPtr, dtype=self.DATA_TYPE)

            if modelBufferPtr is not None:
                self.rawJacobiansBuffer = np.frombuffer(modelBufferPtr, dtype=self.DATA_TYPE)
            
            if observerBufferPtr is not None:
                self.rawVelocitiesBuffer = np.frombuffer(observerBufferPtr, dtype=self.DATA_TYPE)

            if wirelessRemotePtr is not None:
                self.rawRemoteBuffer = np.frombuffer(wirelessRemotePtr, dtype=np.float32)

            if commandPtr is not None:
                self.rawCommandBuffer = np.frombuffer(commandPtr, dtype=np.float32)

        if wirelessRemotePtr is not None:
            self.wirelessParser = WirelessRemote()

    def __del__(self):
        if SHM_IMPORTED: 
            if hasattr(self, 'rawStateShm'):
                self.rawStateShm.close()
            
            if hasattr(self, 'rawVelShm'):
                self.rawVelShm.close()

            if hasattr(self, 'rawJacShm'):
                self.rawJacShm.close()

            if hasattr(self, 'rawRemoteShm'):
                self.rawRemoteShm.close()
            
            if hasattr(self,'rawCommandShm'):
                self.rawCommandShm.close()
                if self.createdCommandShm:
                    try:
                        self.rawCommandShm.unlink()
                    except:
                        info("A1SharedState failed to unlink shared memory containing the command")

    def GetBaseOrientationQuaternion(self):
        q = self.stateVec[:4]
        return [q[1],q[2],q[3],q[0]]
    
    def GetBaseOrientationMatrix(self):
        return R.from_quat(self.GetBaseOrientationQuaternion()).as_matrix()

    def GetBaseRollPitchYaw(self):
        return EulerFromQuaternion(self.GetBaseOrientationQuaternion())

    def GetBaseRollPitchYawRate(self):
        return self.stateVec[4:7]

    def GetBaseAcceleration(self):
        return self.stateVec[7:10]

    def GetFootForces(self):
        return self.stateVec[10:14]

    def GetJointAngles(self):
        return self.stateVec[14:26]

    def GetMotorVelocities(self):
        return self.stateVec[26:38]
    
    def GetCurrentTick(self):
        return self.stateVec[38]

    def GetFootPositionsInBaseFrame(self):
        foot_angles = self.GetJointAngles()
        foot_angles = np.reshape(foot_angles,(4,3))
        footPositions = np.zeros((4, 3))

        for i in range(4):
            footPositions[i] = FootPositionInHipFrame(foot_angles[i],
                                                        l_hip_sign=i)
        return footPositions + HIP_OFFSETS

    def GetHipPositionsInBaseFrame(self):
        return self._DEFAULT_HIP_POSITIONS

    def GetMotorPositionGains(self):
        return self.KPS

    def GetMotorVelocityGains(self):
        return self.KDS

    def GetFootContacts(self):
        footForce = self.GetFootForces()
        return footForce > self.FOOT_FORCE_THRESHOLD
    
    def GetVelocities(self):
        velocities = self.rawVelocitiesBuffer
        return velocities.reshape((2,3))

    def GetJacobians(self):
        jacobians = self.rawJacobiansBuffer
        return jacobians.reshape((12,3))

    def GetBaseVelocity(self):
        return self.GetVelocities()[0]
    
    def GetComVelocity(self):
        return self.GetVelocities()[1]

    def ComputeJacobian(self,idx):
        return self.GetJacobians()[idx*3:idx*3+3]
    
    def SetCommand(self,command):
        np.copyto(self.rawCommandBuffer,command)

    def GetCommand(self):
        return np.copy(self.rawCommandBuffer)

    def printWirelessState(self):
        remote = self.wirelessParser
        state = np.copy(self.rawRemoteBuffer).tobytes()
        remote.set_state(state)
        remote.update_state()
        print(f'\n{15*"/"} WIRELESS REMOTE STATE {15*"/"}')
        print(f' BUTTONS:\n A {remote.button.A} B {remote.button.B} X {remote.button.X} Y {remote.button.Y}',
            f'\n  Right {remote.button.right} Left {remote.button.left} Up {remote.button.up} Down {remote.button.down}',
            f'\n  R1 {remote.button.R1} R2 {remote.button.R2} L1 {remote.button.L1} L2 {remote.button.L2}',
            f'\n  Start {remote.button.start} Select {remote.button.select} F1 {remote.button.F1} F2 {remote.button.F2}',
            f'\n JOYSTICKS:\n LEFT: X {round(remote.left_joystick[0],2)} Y {round(remote.left_joystick[1],2)}',
            f'\n  RIGHT: X {round(remote.right_joystick[0],2)}  Y {round(remote.right_joystick[1],2)}',
            end=10*" " + "\n", flush=True)
        print(f'{52*"/"}\n')
        print(12*'\033[A', end="\r", flush=True)

    def printfullState(self):
        print("GetBaseOrientationQuaternion")
        print(self.GetBaseOrientationQuaternion())
        print("GetBaseAcceleration")
        print(self.GetBaseAcceleration())
        print("GetFootForces")
        print(self.GetFootForces())
        print("GetJointAngles")
        print(self.GetJointAngles())
        print("GetMotorVelocities")
        print(self.GetMotorVelocities())
        print("GetBaseVelocity")
        print(self.GetBaseVelocity())
