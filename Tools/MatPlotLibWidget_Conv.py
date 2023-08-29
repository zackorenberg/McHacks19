"""
Created by @zackorenberg

This is meant to embed matplotlib file
"""

import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import random

from DataTaker import DataTaking

from Tools import Math
import LocalVars

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import time
import numpy as np


class mplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=5, dpi=100):
        self.fig = Figure(figsize=(width,height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.axes = self.fig.add_subplot(111)

        self.colorinterpolation = 50
        self.colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm
        self.data = (0,0)
        self.compute_initial_figure()


        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class DynamicMplCanvas(mplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, Tcold, Thot, dtt=None, savepath=None,*args, **kwargs):
        self.savepath = savepath
        self.dtt = dtt
        self.data = (Tcold, Thot)
        self.all_data = []
        mplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)
        self.rho = 1
        self.nu = 0.1
        self.panes = 2
        self.magnefying = 500
        self.colorinterpolation = 50
        self.colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm
        self.xs = (0,LocalVars.PaneWidth)
        self.ys = (0,LocalVars.PaneHeight)



    def compute_initial_figure(self):
        X,Y,T = Math.PlottableAxes(LocalVars.CMin, LocalVars.CMax)
        self.conduction, self.convection = Math.PlottableAxesBoth(*self.data)
        cp = self.axes.contourf(X, Y, T, self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax,
                                     cmap=self.colourMap)
        self.axes.set_title(LocalVars.PlotTitle)
        self.axes.set_xlabel(LocalVars.PlotXLabel)
        self.axes.set_ylabel(LocalVars.PlotYLabel)
        self.xs = (np.min(X),np.max(X))
        self.ys = (np.min(Y),np.max(Y))
        self.xlim = self.axes.set_xlim(self.xs)
        self.ylim = self.axes.set_xlim(self.ys)

        # Set Colorbar
        self.fig.colorbar(cp, label=LocalVars.PlotCMLabel, extend='max')

        self.cp = self.axes.contourf(*self.conduction,self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax, cmap=self.colourMap)
        self.sp = self.axes.streamplot(*self.convection)
        self.changed = True
        self.i = 0
        self.on = False
        #self.update_figure()

    def start(self):
        self.on = True
        self.i = 0
        self.update_figure()
    def stop(self):
        self.on = False


    def update_data(self, rho, nu, magnefying, panes):
        self.rho, self.nu, self.magnefying, self.panes = rho, nu, magnefying, panes

    def setDTT(self, dtt):
        self.dtt = dtt


    def update_plot(self, Tcold, Thot):
        """ useless function """
        pass
        self.axes.cla()
        self.conduction, self.convection = Math.PlottableAxesBoth(Tcold, Thot)
        self.axes.set_title(LocalVars.PlotTitle)
        self.axes.set_xlabel(LocalVars.PlotXLabel)
        self.axes.set_ylabel(LocalVars.PlotYLabel)
        self.cp = self.axes.contourf(*self.conduction, self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax,
                           cmap=self.colourMap)

        self.sp = self.axes.streamplot(*self.convection)


        self.changed = True
        self.draw()

    def update_figure(self):
        if not self.on:
            return
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        """if self.changed:
            self.draw()
            self.changed = False
            self.fig.savefig("figure_%d.png"%self.i)
            self.i += 1
        """
        if self.dtt is None:
            l = random.randint(0, 5)
            r = random.randint(20, 30)
        else:
            try:
                l, r = self.dtt.readOne()
                print(l,r)
            except:
                return
        self.all_data.append((l,r))
        self.axes.cla()
        self.conduction, self.convection = Math.PlottableAxesBoth(l,r,self.rho, self.nu, self.magnefying,self.panes)
        self.axes.set_title(LocalVars.PlotTitle)
        self.axes.set_xlabel(LocalVars.PlotXLabel)
        self.axes.set_ylabel(LocalVars.PlotYLabel)

        self.cp = self.axes.contourf(*self.conduction, self.colorinterpolation, vmin=LocalVars.CMin,
                                     vmax=LocalVars.CMax,
                                     cmap=self.colourMap)
        if self.panes > 2:
            length = len(self.convection[2])
            lens = []
            for i in range(1,self.panes):
                self.axes.streamplot(*[c[:,(i-1)*int(length/(self.panes-1))+1:(i)*int(length/(self.panes-1))-1] for c in self.convection],color='k',density=[0.5,1])
                #self.axes.streamplot(*[c[:,(i-1)*int(length/(self.panes-1)):(i)*int(length/(self.panes-1))+1] for c in self.convection])
                #self.axes.streamplot(*[c[:,(i-1)*int(length/(self.panes-1)):(i)*int(length/(self.panes-1))] for c in self.convection],color='k',density=[0.5,1]) #this plots 0 velocity parts as well
                lens.append((i)*int(length/(self.panes-1)))
            for l in lens:
                self.axes.axvline(self.xs[0]+(self.xs[1]-self.xs[0])*((l-2.5)/length),0,1, color='k', linewidth=2)

        else:
            self.sp = self.axes.streamplot(*self.convection,color='k',density=[0.5,1]) # integration_direction = 'backward')

        self.axes.set_xlim(self.xlim)
        self.axes.set_ylim(self.ylim)
        self.changed = True
        if self.savepath is not None:
            self.fig.savefig(self.savepath+"figure%d.png"%self.i)
            self.i += 1
        self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, debug=False):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Main Plot Window")

        """self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)"""

        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.resize(self.main_widget.height()*4, self.main_widget.width())
        l = QtWidgets.QVBoxLayout(self.main_widget)
        if debug:
            dtt = DataTaking.DebugData()
        else:
            try:
                dtt = DataTaking.DataTaking("COM3")
            except:
                dtt = DataTaking.DebugData()
        dc = DynamicMplCanvas(0,25,parent=self.main_widget, dtt=dtt, height=50, dpi=100)
        l.addWidget(dc)
        x = np.sin(np.linspace(0,np.pi,50))
        y = np.cos(np.linspace(0,np.pi,50))


        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(dc, self))

        #dc.start()
        self.dc = dc
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Thermo ftw", 2000)
        self.show()
        """for X,Y in zip(x,y):
            dc.update_plot(X,Y)
            time.sleep(1)"""
    def start(self):
        self.dc.start()
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
    def stop(self):
        self.dc.stop()
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)

    aw = ApplicationWindow(debug=True)
    aw.setWindowTitle(LocalVars.PlotWindowTitle)
    aw.show()
    sys.exit(qApp.exec_())