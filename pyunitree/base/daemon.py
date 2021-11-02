from multiprocessing import Process
from multiprocessing.sharedctypes import RawValue
from time import perf_counter,sleep
from logging import info
import numpy as np


try:
    from multiprocessing.shared_memory import SharedMemory
    SHM_IMPORTED = True 
except ModuleNotFoundError:
    SHM_IMPORTED = False

class Daemon:
    '''
    Class that is used to perform action() in a separate process
    A set of virtual methods has to be implemented on demand:
    
    processInit():
        to be executed in the handler process prior to the execution loop
    
    action():
        action to be executed in the main loop with the specified update rate
        
        returns True if the action is succeded otherwise returns False and the 
        handler process loop terminates

    onStart()
        to be executed in the main process after the handler process start() call

    onStop()
        to be executed in the main process prior to the handler process stop() call
    '''

    def __init__(self,updateRate=-1,name="Unnamed daemon"):
        """
        Negative update rate will cause the class to execute action at max update rate possible
        """
        self.name = name
        self.handlerProc = Process(target=self.handler,daemon=True)
        self.updateRate = updateRate
        self.hasSharedState = False
        self._processReady = RawValue("b",0)

    def handler(self):
        self.processInit()
        try:
            self._processReady.value = 1
            initial_time = perf_counter()
            tick = 0
            while True:
                actual_time = perf_counter() - initial_time
                if actual_time - tick >= 1/self.updateRate or self.updateRate<0:
                    result = self.action()
                    tick = actual_time
                    if not result:
                        info(f"Process {self.name} terminated due to the incorrect action() result")
                        break
        except KeyboardInterrupt:
            info(f"Process {self.name} was interrupted")



    def initSharedStateArray(self,size,name,dataType=np.float32):
        self.hasSharedState = True
        self.sharedStateName = name
        self.sharedStateSize = size
        self.sharedStateType = dataType

        data = np.zeros(self.sharedStateSize,dtype=dataType)
        
        if SHM_IMPORTED:
            try:
                self.rawStateShm = SharedMemory(create=True, size=data.nbytes,name=name)
            except FileExistsError:
                self.rawStateShm = SharedMemory(name=name)

            self.rawStatePtr =  self.rawStateShm.buf
        else:
            from multiprocessing import RawArray
            self.rawStatePtr = RawArray("f",self.sharedStateSize)

        self.rawStateBuffer = np.frombuffer(self.rawStatePtr, dtype=self.sharedStateType)
    
        np.copyto(self.rawStateBuffer, data)

    def start(self):
        self.preStart()
        self.handlerProc.start()
        while self._processReady.value == 0:
            sleep(0.01)
        info(f"Process {self.name} has started")
        self.onStart()

    def preStart(self):
        pass
    
    def stop(self):
        self.onStop()
        self.handlerProc.terminate()
        info(f"Process {self.name} terminated")

    def __del__(self):
        if self.hasSharedState == True and SHM_IMPORTED:
            self.rawStateShm.close()
            self.rawStateShm.unlink()

        self.stop()
    
    def init(self):
        pass
    
    def processInit(self):
        pass

    def action(self):
        pass

    def onStart(self):
        pass

    def onStop(self):
        pass


