# example of extending the Process class and adding shared attributes
from time import sleep
from multiprocessing import Process
from multiprocessing import Value
from multiprocessing import Array
from multiprocessing import Event
from multiprocessing import RawArray
import numpy as np


def raw_to_numpy(raw, shape = (600, 800, 3)):
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def numpy_to_raw(np_arr):
    return RawArray(np.ctypeslib.as_ctypes_type(np_arr.dtype), np_arr.flatten())
    
img_shape = (2, 3)

# custom process class
class CustomProcess(Process):
#     override the constructor
    def __init__(self, event, sharr):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.event = event
        self.sharr = sharr
        self.data = Value('i', 0)

    # override the run function
    def run(self):
        while(True):
            sleep(1)    
            # store the data variable
            self.data.value = int(100*np.random.random())
            # for i in range(0, len(self.sharr)):
                # self.sharr[i] = int(100*np.random.random())
            
            data = (np.random.random(6)*100).astype(np.uint8).reshape(*img_shape)
            self.sharr_np = np.frombuffer(self.sharr, dtype=np.uint8).reshape(*img_shape)
            np.copyto(self.sharr_np, data)
            
            if self.event.is_set():
                break
        
# create the process
if __name__ == '__main__':
    event = Event()
    sharr = RawArray('c', (1, 2, 3, 4, 5, 6))
    process = CustomProcess(event, sharr)

    # start the process
    process.start()
    # wait for the process to finish
    print('Waiting for the child process to finish')

    for i in range(0, 100):
        arr = raw_to_numpy(sharr, shape = (2, 3))
        print(arr)
        sleep(0.1)

    event.set()

    # process.join()
    # report the process attribute
    print(f'Parent got: {process.data.value}')