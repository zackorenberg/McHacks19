import serial
import numpy as np
import LocalVars



class DataTaking(object):

    def __init__(self, port="COM4"):
        self.port = port
        # do connection stuff
        self.conn = serial.Serial(self.port, 9600)

    def readOne(self):
        data = self.conn.readline().decode()
        try:
            Thot, Tcold = data.strip("\n").split(",")
        except:
            return None
        return float(Tcold), float(Thot)

    def read(self):
        while True:
            Tcold, Thot = self.readOne()
            print(Tcold, Thot)

class DebugData(object):

    def __init__(self, port="COM4"):
        self.i = 0
        self.sin = 2 + 3*np.sin(np.linspace(0,2*np.pi,100))
        self.cos = 20 + 5*np.cos(np.linspace(0,2*np.pi,100))

    def readOne(self):
        self.i += 1
        #if self.i % 2 == 0:
        return self.sin[self.i % 100], self.cos[self.i % 100]
        #else:
         #   return self.cos[self.i % 100], self.sin[self.i % 100]

    def read(self):
        pass

class VirtualTaking(object):
    def __init__(self, file):
        self.data = []
        self.debug = None
        try:
            with open(file, 'r') as f:
                for line in f:
                    Th, Tc = line.strip("\n").split(",")
                    self.data.append((float(Th),float(Tc)))
        except:
            print("Unable to open file: %s"%file)
            self.debug = DebugData()

        self.i = 0
        self.length = len(self.data)

    def readOne(self):
        if self.debug is not None:
            return self.debug.readOne()
        if self.i >= self.length and not LocalVars.CYCLE_VIRTUAL:
            return None
        elif not LocalVars.CYCLE_VIRTUAL:
            print("%d of %d"%(self.i, self.length))
        dat = self.data[self.i%self.length]
        self.i += 1
        return dat

if __name__ == "__main__":
    dtt = DataTaking("COM3")
    dtt.read()