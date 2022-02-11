from multiprocessing import Process
from multiprocessing.sharedctypes import RawValue
from time import perf_counter,sleep
from logging import info
import numpy as np

from unitreepy.utils.exception_parser import parse_exception
from unitreepy.interfaces.shared_state import SharedState,SHM_IMPORTED

class Daemon:
    '''
    Class that is used to perform action() in a separate process
    A set of virtual methods has to be implemented on demand:
    
    process_init():
        to be executed in the handler process prior to the execution loop
    
    action():
        action to be executed in the main loop with the specified update rate
        
        returns True if the action is succeded otherwise returns False and the 
        handler process loop terminates

    on_start()
        to be executed in the main process after the handler process start() call

    on_stop()
        to be executed in the main process prior to the handler process stop() call
    '''
    def __init__(self,update_rate=-1,name="Unnamed daemon"):
        """
        Negative update rate will cause the class to execute action at max update rate possible
        """
        self.name = name
        self.handler_proc = Process(target=self.handler,daemon=True)
        self.update_rate = update_rate
        self.has_shared_state = False
        self.__sh_process_running = RawValue("b",0)

    def handler(self):
        self.process_init()
        try:
            self.__sh_process_running.value = 1
            initial_time = perf_counter()
            tick = 0
            while self.__sh_process_running.value:
                actual_time = perf_counter() - initial_time
                if actual_time - tick >= 1/self.update_rate or self.update_rate<0:
                    result = self.action()
                    tick = actual_time
                    if not result:
                        info(f"Process {self.name} terminated due to the incorrect action() result")
                        break
        except KeyboardInterrupt:
            info(f"Process {self.name} was interrupted")
        except Exception as e:
            info(f"Daemon process {self.name} was interrupted by an exception inside the handler \n \
                Exception: \n {parse_exception(e)}")
            
        
    def init_shared_state_array(self,size,name,data_type=np.float32):
        self.has_shared_state = True
        self.shared_state = SharedState()
        self.shared_state.register_shared_memory(name,size,data_type)
        data = np.zeros(size,dtype=data_type)
        self.raw_state_buffer = self.shared_state[name]
        np.copyto(self.raw_state_buffer, data)

    def start(self):
        self.pre_start()
        self.handler_proc.start()
        while self.__sh_process_running.value == 0:
            sleep(0.01)
        info(f"Process {self.name} has started")
        self.on_start()

    def pre_start(self):
        pass
    
    def stop(self):
        self.on_stop()
        self.__sh_process_running.value = 0
        self.handler_proc.join(timeout=1)
        self.cleanup()
        info(f"Process {self.name} terminated")

    def cleanup(self):
        if self.has_shared_state:
            self.shared_state.cleanup()
    
    def init(self):
        pass
    
    def process_init(self):
        pass

    def action(self):
        pass

    def on_start(self):
        pass

    def on_stop(self):
        pass


