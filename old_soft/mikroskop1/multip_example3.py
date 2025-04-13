import queue, threading
from tkinter import *
import time
import random


class TkThread:
    def __init__(self):
        self.tk = Tk()
        self.message_queue = queue.Queue()
        self.message_event = '<<message>>'
        self.tk.bind(self.message_event, self.process_message_queue)
        

    def run_tk(self):
        self.tk.title('My Window') # add buttons, etc.
        self.tk.lift(); self.tk.mainloop()

    def send_message_to_ui(self, message):
        self.message_queue.put(message)
        self.tk.event_generate(self.message_event, when='tail')

    def process_message_queue(self, event):
        while self.message_queue.empty() is False:
            message = self.message_queue.get(block=False)
            # process the message here
            print(message)
            
      
class BackgroundThread:

    def __init__(self, tk_thread):
        self.tk_thread = tk_thread
        self.thread = threading.Thread(target=self.run_thread); self.thread.start()

    def run_thread(self):
        while(1):
            jamnik_rnd = int(100*random.random())
            
            time.sleep(1)         # just an example... this gives the Tk some time to start up
            tk_thread.send_message_to_ui('jamnik = %d'%jamnik_rnd)

if __name__ == '__main__':
    tk_thread = TkThread()
    back_thread = BackgroundThread(tk_thread) 
    tk_thread.run_tk()    # initiate last, since this runs tk.main_loop() which is a