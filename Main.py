
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.Qt as Qt
import PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DataTaker import DataTaking, DataSaver
import sys

from Tools import MatPlotLibWidget_Conv
import LocalVars
from Layout import ConnectionWidget

class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

class DataSim(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        #self.left = 10
        #self.top = 10
        self.title = "Configurations"
        #self.width = 640
        #self.height = 400

        # modules
        self.dtt_debug = DataTaking.DebugData()
        #self.pltWidget = QtWidgets.QWidget(self)
        #self.pltWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.dss = DataSaver.DataSaver()
        self.pltWindow = MatPlotLibWidget_Conv.ApplicationWindow()


        self.pltWindow.setWindowTitle(LocalVars.PlotWindowTitle)


        self.pltModule = self.pltWindow.dc

        self.connectWidget = ConnectionWidget.ConnectionWidget()

        self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)

        self.flayout = QtWidgets.QFormLayout()


        #self.horizontalGroupBox = QtWidgets.QGroupBox("Grid")
        #self.horizontalGroupBox.setLayout(self.grid)
        #self.layout.addLayout(self.grid)


        self.flayout.addRow(QtWidgets.QLabel("Serial Connection"),self.connectWidget)

        self.flayout.addWidget(QHLine())
        self.density = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.flayout.addRow(QtWidgets.QLabel("Density"),self.density)
        self.viscosity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.flayout.addRow(QtWidgets.QLabel("Viscosity"), self.viscosity)
        self.multiplier = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.flayout.addRow(QtWidgets.QLabel("Multiplier"), self.multiplier)
        self.panes = QtWidgets.QSpinBox()
        self.flayout.addRow(QtWidgets.QLabel("Panes"), self.panes)


        self.density.valueChanged.connect(self.sliderChange)
        self.viscosity.valueChanged.connect(self.sliderChange)
        self.multiplier.valueChanged.connect(self.sliderChange)
        self.panes.valueChanged.connect(self.sliderChange)

        self.density.setMinimum(LocalVars.DensityRange[0])
        self.density.setMaximum(LocalVars.DensityRange[1])
        self.density.setValue(LocalVars.DensityDefault)

        self.viscosity.setMinimum(LocalVars.ViscosityRange[0])
        self.viscosity.setMaximum(LocalVars.ViscosityRange[1])
        self.viscosity.setValue(LocalVars.ViscosityDefault)

        self.multiplier.setMinimum(LocalVars.MultiplierRange[0])
        self.multiplier.setMaximum(LocalVars.MultiplierRange[1])
        self.multiplier.setValue(LocalVars.MultiplierDefault)

        self.panes.setMaximum(LocalVars.PanesRange[1])
        self.panes.setMinimum(LocalVars.PanesRange[0])
        self.panes.setValue(LocalVars.PanesDefault)


        self.flayout.addWidget(QHLine())

        self.outputfile = QtWidgets.QLineEdit("images/")
        self.flayout.addRow(QtWidgets.QLabel("SaveDir"), self.outputfile)

        self.flayout.addWidget(QHLine())

        self.start = QtWidgets.QPushButton("Start")
        self.start.clicked.connect(self.startDTT)
        self.stop = QtWidgets.QPushButton("Stop")
        self.stop.clicked.connect(self.stopDTT)

        self.flayout.addWidget(self.start)
        self.flayout.addWidget(self.stop)

        self.flayout.addWidget(QHLine())
        self.importpath = QtWidgets.QLineEdit("")
        self.load = QtWidgets.QPushButton("Load")
        self.load.clicked.connect(self.loadDTT)
        self.flayout.addRow(QtWidgets.QLabel("Data Path"),self.importpath)
        self.flayout.addWidget(self.load)


        self.setLayout(self.flayout)

        self.show()

    def startDTT(self):
        self.pltWindow.start()
        self.pltModule.savepath = self.dss.createSavePath(self.outputfile.text())
        self.dss.path = self.pltModule.savepath

        self.pltModule.all_data = []

    def stopDTT(self):
        self.pltModule.stop()
        dat = self.pltModule.all_data
        if len(dat) != 0:
            self.dss.saveData(dat)
            pass
        else:
            print("hello")
        self.dss.createGif()
    def createGrid(self):
        layout = QtWidgets.QGridLayout()
        #layout.setColumnStretch(3, 9)
        #layout.setColumnStretch(1, 9)
        self.grid = layout

    def loadDTT(self):
        path = self.importpath.text()
        self.stopDTT()
        self.pltModule.setDTT(DataTaking.VirtualTaking(path))
        self.startDTT()


    def connectSerial(self, port):
        if port == "":
            return
        else:
            self.pltModule.setDTT(DataTaking.DataTaking(port))

    def sliderChange(self):
        rho = float(self.density.value())/10
        nu = float(self.viscosity.value())/100
        multiplier = float(self.multiplier.value())
        panes = int(self.panes.value())
        self.pltModule.update_data(rho,nu,multiplier,panes)





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = DataSim()
    sys.exit(app.exec_())