from logging import info
import logging
import numpy as np
import subprocess
import sys

try:
    from multiprocessing.shared_memory import SharedMemory
    SHM_IMPORTED = True 
except ModuleNotFoundError:
    SHM_IMPORTED = False
    print("unitreepy only supports versions of python 3.8+")
    sys.exit()

class SharedState:

    def __init__(self):

        self.shms = {}
        self.update_shared_dict()

    def update_shared_dict(self):
        """
            Updates a dictionary that maps all of the unitreepy's allocated shared memory names to their SharedMemory objects and their numpy proxies
            Naming convention for the allocations is unitreepy.robotname.arrayname.arraytype
            Note that there is no access management which may cause race conditions
        """
        shm_names = str(subprocess.check_output(["ls /dev/shm"], shell=True)).replace("\\n", " ").replace("\'", " ").split()
        for full_name in shm_names:
            if "unitreepy" in full_name:
                shm_dirs = full_name.split(".")
                shm_type = shm_dirs[-1]
                shm_id = ".".join(shm_dirs[1:-1]) # removes unitreepy and datatype
                if shm_id not in self.shms.keys():
                    dtype = None
                    if shm_type == "float32":
                        dtype = np.float32
                    if shm_type == "float64":
                        dtype = np.float64
                    if shm_type == "int":
                        dtype = np.int32
                    if shm_type == "byte":
                        dtype = np.bytes_

                    if dtype == None:
                        logging.error(f"unitreepy: Cannot deduce shared memory type for {shm_id}")
                    else:
                        shm = SharedMemory(name=full_name)
                        buffer = np.frombuffer(shm.buf, dtype=dtype)
                        self.shms[shm_id] = (shm, buffer)

    def register_shared_memory(self, name, size, dtype):
        """[summary]

        Args:
            name (str): new shared memory block name
            size (int): number of dtype items in the block
            dtype (np.float32/float64/int32/bytes_): block data type
        """
        if name in self.shms.keys():
            logging.error(f"unitreepy: Attempting to register existing shared memory name : {str(name)}")

        str_dtype = None
        
        if dtype == np.float32:
            str_dtype = ".float32"

        if dtype == np.float64:
            str_dtype = ".float64"   
    
        if dtype == np.int32:
            str_dtype = ".int"
    
        if dtype == np.bytes_:
            str_dtype = ".byte"
        
        if str_dtype == None:
            logging.error(f"unitreepy: unsupported shared memory type : {str(dtype)}")
        else:
            shm_name = "unitreepy."+name+str_dtype

            data = np.zeros(size,dtype=dtype)
            try:
                shm = SharedMemory(create=True, size=data.nbytes,name=shm_name)
            except FileExistsError:
                logging.warning(f"unitreepy: Failed to register {str(name)} because some other SharedState instance already registered it. Attempting to use it")
                shm = SharedMemory(name=shm_name)

            buffer = np.frombuffer(shm.buf, dtype=dtype)
            np.copyto(buffer, data)
            self.shms[name] = (shm, buffer)

    def unlink_shared_memory(self):
        """
        Unlinks all shared memory blocks
        """
        for name, data in self.shms.items():
            shm = data[0]
            try:
                shm.unlink()
            except FileNotFoundError:
                info("UNLINK: Shared memory {shm_name} block not found")

    def destroy_shared_memory(self):
        """
        Attempts to destroy all known shared memory blocks
        """
        for name, data in self.shms.items():
            shm = data[0]
            try:
                shm.close()
            except FileNotFoundError:
                info("DESTROY: Shared memory {shm_name} block not found")

    def cleanup(self, destroy=False):
        self.unlink_shared_memory()
        if destroy:
            self.destroy_shared_memory()

    def names(self):
        return self.shms.keys()

    def __getitem__(self,name):
        """[summary]

        Args:
            name (str): name of the shared memory as in unitreepy.NAME.dtype

        Raises:
            KeyError: No shared memory was found by this name

        Returns:
            numpy buffer
        """
        if name in self.shms.keys():
            return self.shms[name][1]
        else:
            raise KeyError(f"No shm named {name}")