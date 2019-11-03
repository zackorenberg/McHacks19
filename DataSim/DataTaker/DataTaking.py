import serial
import numpy as np




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
        self.sin = 2 + 3*np.sin(np.linspace(0,3*np.pi,100))
        self.cos = 20 + 5*np.cos(np.linspace(0,3*np.pi,100))

    def readOne(self):
        self.i += 1
        return self.sin[self.i % 100], self.cos[self.i % 100]

    def read(self):
        pass



if __name__ == "__main__":
    dtt = DataTaking("COM3")
    dtt.read()