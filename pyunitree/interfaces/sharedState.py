from logging import info
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyunitree.utils.kinematics import EulerFromQuaternion,QuaternionToEulerMatrix
from pyunitree.robots.a1.constants import POSITION_GAINS,DAMPING_GAINS
from pyunitree.base.daemon import SHM_IMPORTED
from pyunitree.parsers.remote import WirelessRemote


class A1SharedState:
    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))

    _DEFAULT_HIP_POSITIONS = (
        (0.18, -0.14, 0),
        (0.18, 0.14, 0),
        (-0.18, -0.14, 0),
        (-0.18, 0.14, 0),
    )

    '''
    MPC_BODY_MASS = 13.52
    MPC_BODY_INERTIA = np.array((0.032, 0, 0, 0, 0.283, 0, 0, 0, 0.308))

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
    
    DATA_TYPE = np.float32
    """
    self.rawStateBuffer
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
            self.rawStateBuffer = np.frombuffer(self.rawStateShm.buf, dtype=self.DATA_TYPE)

            try:
                self.rawModelShm = SharedMemory(name="Model")
                self.rawModelBuffer = np.frombuffer(self.rawModelShm.buf, dtype=self.DATA_TYPE)
            except FileNotFoundError:
                self.rawModelBuffer = None

            try:
                self.rawObserverShm = SharedMemory(name="Observer")
                self.rawObserverBuffer = np.frombuffer(self.rawObserverShm.buf, dtype=self.DATA_TYPE)
            except:
                self.rawObserverBuffer = None

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
            np.copyto(self.rawCommandBuffer,np.zeros(3,dtype=np.float32))
        else:
            if stateBufferPtr is not None:
                self.rawStateBuffer = np.frombuffer(stateBufferPtr, dtype=self.DATA_TYPE)

            if modelBufferPtr is not None:
                self.rawModelBuffer = np.frombuffer(modelBufferPtr, dtype=self.DATA_TYPE)
            
            if observerBufferPtr is not None:
                self.rawObserverBuffer = np.frombuffer(observerBufferPtr, dtype=self.DATA_TYPE)

            if wirelessRemotePtr is not None:
                self.rawRemoteBuffer = np.frombuffer(wirelessRemotePtr, dtype=np.float32)

            if commandPtr is not None:
                self.rawCommandBuffer = np.frombuffer(commandPtr, dtype=np.float32)

        if wirelessRemotePtr is not None:
            self.wirelessParser = WirelessRemote()

        self.stateCopy = np.zeros(self.rawStateBuffer.shape) if self.rawStateBuffer is not None else None
        self.modelCopy = np.zeros(self.rawModelBuffer.shape) if self.rawModelBuffer is not None else None
        self.observerCopy = np.zeros(self.rawObserverBuffer.shape) if self.rawObserverBuffer is not None else None
        self.remoteCopy = np.zeros(self.rawRemoteBuffer.shape) if self.rawRemoteBuffer is not None else None
        self.commandCopy = np.zeros(self.rawCommandBuffer.shape) if self.rawCommandBuffer is not None else None

    def __del__(self):
        if SHM_IMPORTED: 
            if hasattr(self, 'rawStateShm'):
                self.rawStateShm.close()
            
            if hasattr(self, 'rawObserverShm'):
                self.rawObserverShm.close()

            if hasattr(self, 'rawModelShm'):
                self.rawModelShm.close()

            if hasattr(self, 'rawRemoteShm'):
                self.rawRemoteShm.close()
            
            if hasattr(self,'rawCommandShm'):
                self.rawCommandShm.close()
                if self.createdCommandShm:
                    try:
                        self.rawCommandShm.unlink()
                    except:
                        info("A1SharedState failed to unlink shared memory containing the command")

    def copyBuffers(self,buffers=["state","model","observer","remote","command"]):
        if self.rawStateBuffer is not None and "state" in buffers:
            np.copyto(self.stateCopy,self.rawStateBuffer)
        if self.rawModelBuffer is not None and "model" in buffers:
            np.copyto(self.modelCopy,self.rawModelBuffer)
        if self.rawObserverBuffer is not None and "observer" in buffers:
            np.copyto(self.observerCopy,self.rawObserverBuffer)
        if self.rawRemoteBuffer is not None and "remote" in buffers:
            np.copyto(self.remoteCopy,self.rawRemoteBuffer)
        if self.rawCommandBuffer is not None and "command" in buffers:
            np.copyto(self.commandCopy,self.rawCommandBuffer)

    def getStateBuffer(self,useCached=False):
        if useCached:
            if self.stateCopy is not None:
                return self.stateCopy
            else:
                raise Exception("State buffer was not initialized in the instance of this shared state object")
        else:
            return self.rawStateBuffer

    def getModelBuffer(self,useCached=False):
        if useCached:
            if self.modelCopy is not None:
                return self.modelCopy
            else:
                raise Exception("Model buffer was not initialized in the instance of this shared state object")
        else:
            return self.rawModelBuffer

    def getObserverBuffer(self,useCached=False):
        if useCached:
            if self.observerCopy is not None:
                return self.observerCopy
            else:
                raise Exception("Observer buffer was not initialized in the instance of this shared state object")
        else:
            return self.rawObserverBuffer

    def getRemoteBuffer(self,useCached=False):
        if useCached:
            if self.remoteCopy is not None:
                return self.remoteCopy
            else:
                raise Exception("Remote buffer was not initialized in the instance of this shared state object")
        else:
            return self.rawRemoteBuffer
            
    def getCommandBuffer(self,useCached=False):
        if useCached:
            if self.commandCopy is not None:
                return self.commandCopy
            else:
                raise Exception("Command buffer was not initialized in the instance of this shared state object")
        else:
            return self.rawCommandBuffer

    def GetBaseOrientationQuaternion(self,useCached=False):
        q = self.getStateBuffer(useCached)[:4].copy()
        return [q[1],q[2],q[3],q[0]]
    
    def GetBaseOrientationMatrix(self,useCached=False):
        return QuaternionToEulerMatrix(self.GetBaseOrientationQuaternion(useCached=useCached))

    def GetBaseRollPitchYaw(self,useCached=False):
        return EulerFromQuaternion(self.GetBaseOrientationQuaternion(useCached=useCached))

    def GetBaseRollPitchYawRate(self,useCached=False):
        return self.getStateBuffer(useCached)[4:7].copy()

    def GetBaseAcceleration(self,useCached=False):
        return self.getStateBuffer(useCached)[7:10].copy()

    def GetFootForces(self,useCached=False):
        return self.getStateBuffer(useCached)[10:14].copy()

    def GetJointAngles(self,useCached=False):
        return self.getStateBuffer(useCached)[14:26].copy()

    def GetMotorVelocities(self,useCached=False):
        return self.getStateBuffer(useCached)[26:38].copy()
    
    def GetCurrentTick(self,useCached=False):
        return self.getStateBuffer(useCached)[38].copy()

    def GetFootPositionsInBaseFrame(self,useCached=False):
        #NOTE copy is important
        return self.getModelBuffer(useCached)[36:].copy().reshape((4,3))
        
    def GetHipPositionsInBaseFrame(self):
        return self._DEFAULT_HIP_POSITIONS

    def GetMotorPositionGains(self):
        return self.KPS

    def GetMotorVelocityGains(self):
        return self.KDS

    def GetFootContacts(self,useCached=False):
        footForce = self.GetFootForces(useCached)
        return footForce > self.FOOT_FORCE_THRESHOLD
    
    def GetVelocities(self,useCached=False):
        return self.getObserverBuffer(useCached).copy().reshape((2,3))

    def GetJacobians(self,useCached=False):
        return self.getModelBuffer(useCached)[:36].copy().reshape((12,3))

    def GetBaseVelocity(self):
        return self.GetVelocities()[0]
    
    def GetComVelocity(self):
        return self.GetVelocities()[1]

    def ComputeJacobian(self,idx):
        return self.GetJacobians()[idx*3:idx*3+3]
    
    def SetCommand(self,command):
        np.copyto(self.rawCommandBuffer,command)

    def GetCommand(self,useCached=False):
        return self.getCommandBuffer(useCached).copy()

    def printWirelessState(self):
        remote = self.wirelessParser
        state = self.rawRemoteBuffer.copy().tobytes()
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
