

import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtGui

import sys
import glob
import serial


def serial_ports(max_ports=20):
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(max_ports)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result



class ConnectionWidget(QtWidgets.QWidget):

    def __init__(self, parent=None,*args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.parent = parent
        self.dropbox = QtWidgets.QComboBox()
        self.dropbox.setEditable(True)

        self.ports = serial_ports()
        for port in self.ports:
            self.dropbox.addItem(port)

        self.button = QtWidgets.QPushButton('Connect')
        self.button.clicked.connect(self.connect)


        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.dropbox)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def connect(self):
        if self.parent is None:
            print(self.dropbox.currentText())
        else:
            self.parent.connectSerial(self.dropbox.currentText())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    cw = ConnectionWidget()
    cw.show()
    sys.exit(app.exec_())