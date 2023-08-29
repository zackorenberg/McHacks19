"""
Created by @zackorenberg

This is meant to embed matplotlib file

DEPRICATED
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
from matplotlib.figure import Figure
import time
import numpy as np


class mplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=8, dpi=100):
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

    def __init__(self, Tcold, Thot, dtt=None, *args, **kwargs):
        self.dtt = dtt
        self.data = (Tcold, Thot)
        mplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)
        self.colorinterpolation = 50
        self.colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm



    def compute_initial_figure(self):
        X,Y,T = Math.PlottableAxes(LocalVars.CMin, LocalVars.CMax)
        cp = self.axes.contourf(X, Y, T, self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax,
                                     cmap=self.colourMap)
        self.axes.set_title(LocalVars.PlotTitle)
        self.axes.set_xlabel(LocalVars.PlotXLabel)
        self.axes.set_ylabel(LocalVars.PlotYLabel)
        # Set Colorbar
        self.fig.colorbar(cp, label="Temperature (Celsius)", extend='max')
        self.X,self.Y,self.T = Math.PlottableAxes(*self.data)
        self.cp = self.axes.contourf(self.X,self.Y,self.T,self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax, cmap=self.colourMap)
        self.changed = True
        self.i = 0
        #self.update_figure()


    def update_plot(self, Tcold, Thot):
        pass
        self.axes.cla()
        self.X,self.Y,self.T = Math.PlottableAxes(Tcold, Thot)
        self.axes.set_title(LocalVars.PlotTitle)
        self.axes.set_xlabel(LocalVars.PlotXLabel)
        self.axes.set_ylabel(LocalVars.PlotYLabel)
        self.axes.contourf(self.X, self.Y, self.T, self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax,
                           cmap=self.colourMap)
        self.changed = True
        self.draw()

    def update_figure(self):
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

        self.axes.cla()
        X, Y, T = Math.PlottableAxes(l,r)
        self.axes.set_title(LocalVars.PlotTitle)
        self.axes.set_xlabel(LocalVars.PlotXLabel)
        self.axes.set_ylabel(LocalVars.PlotYLabel)
        self.axes.contourf(X, Y, T, self.colorinterpolation, vmin=LocalVars.CMin, vmax=LocalVars.CMax,
                            cmap=self.colourMap)
        self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)
        dtt = DataTaking.DataTaking("COM3")
        dc = DynamicMplCanvas(0,25,parent=self.main_widget, dtt=dtt, width=5, height=4, dpi=100)
        l.addWidget(dc)
        x = np.sin(np.linspace(0,np.pi,50))
        y = np.cos(np.linspace(0,np.pi,50))


        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

        """for X,Y in zip(x,y):
            dc.update_plot(X,Y)
            time.sleep(1)"""

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                )

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.setWindowTitle("plotwindow" )
    aw.show()
    sys.exit(qApp.exec_())