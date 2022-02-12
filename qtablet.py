#!/usr/bin/python3

import sys
from PyQt5 import QtGui, QtWidgets, QtCore

class Canvas(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.resize(1000, 1000)
        self.setAutoFillBackground(True)
        self.setAttribute(QtCore.Qt.WA_TabletTracking)

        self.strokes = []

        self._prev_width = None
        self._prev_height = None

        self.device_down = False
        self.pixmap = QtGui.QPixmap()
        self._update_pixmap()

    def _get_pixmap(self, w, h):
        #if (w, h) != (self._prev_width, self._prev_height):
        dpr = self.devicePixelRatioF()
        pixmap = QtGui.QPixmap(round(self.width() * dpr), round(self.height() * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(QtCore.Qt.white)
        return pixmap

    def _update_pixmap(self):
        pixmap = self._get_pixmap(self.width(), self.height())
        painter = QtGui.QPainter(pixmap)
        self._paint_on_pixmap(painter)
        painter.end()
        self.pixmap = pixmap

    def _paint_on_pixmap(self, painter):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.black)
        for stroke in self.strokes:
            path = QtGui.QPainterPath(stroke[0])
            for pt in stroke[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

    def _get_last_stroke(self):
        if self.strokes:
            return self.strokes[-1]
        else:
            stroke = []
            self.strokes.append(stroke)
            return self.strokes[-1]

    def _new_stroke(self):
        stroke = []
        self.strokes.append(stroke)
        return self.strokes[-1]

    def tabletEvent(self, ev):

        t = ev.type()
        if t == QtCore.QEvent.TabletPress:
            self.device_down = True
            self._new_stroke().append(ev.posF())
        elif t == QtCore.QEvent.TabletRelease:
            self.device_down = False
        elif t == QtCore.QEvent.TabletMove:
            if self.device_down:
                #print(ev.posF())
                self._get_last_stroke().append(ev.posF())
                self._update_pixmap()
                self.update(self.rect())

    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        dpr = self.devicePixelRatioF()
        pixmap_portion = QtCore.QRect(ev.rect().topLeft()*dpr, ev.rect().size()*dpr)
        painter.drawPixmap(ev.rect().topLeft(), self.pixmap, pixmap_portion)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_CompressHighFrequencyEvents)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

