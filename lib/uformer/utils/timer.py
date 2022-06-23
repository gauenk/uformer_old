import time

class ExpTimer():

    def __init__(self):
        self.times = []
        self.names = []
        self.start_times = []

    def __str__(self):
        msg = "--- Exp Times ---"
        for k,v in self.items():
            msg += "\n%s: %2.3f\n" % (k,v)
        return msg

    def items(self):
        names = ["timer_%s" % name for name in self.names]
        return zip(names,self.times)

    def start(self,name):
        if name in self.names:
            raise ValueError("Name [%s] already in list." % name)
        self.names.append(name)
        start_time = time.perf_counter()
        self.start_times.append(start_time)

    def stop(self,name):
        end_time = time.perf_counter() # at start
        idx = self.names.index(name)
        start_time = self.start_times[idx]
        exec_time = end_time - start_time
        self.times.append(exec_time)
