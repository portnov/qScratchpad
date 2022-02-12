#!/usr/bin/python3

import sys
import numpy as np
from math import sqrt
from PyQt5 import QtGui, QtWidgets, QtCore

class Stroke(object):
    def paint(self, painter):
        raise Exception("Not implemented")

class SegmentStroke(Stroke):
    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(point)

    def paint(self, painter):
        path = QtGui.QPainterPath(self.points[0])
        for pt in self.points[1:]:
            path.lineTo(pt)
        painter.drawPath(path)

    def recognize_any(self):
        circle = CircularStroke.recognize([(pt.x(), pt.y()) for pt in self.points])
        return circle

class CircularStroke(Stroke):
    def __init__(self):
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius = 0.0

    def paint(self, painter):
        painter.drawEllipse(self.center_x - self.radius, self.center_y - self.radius,
                            2*self.radius, 2*self.radius)

    @classmethod
    def recognize(cls, data, mean_is_zero=False):
        """
        Calculate best approximation of set of 2D vertices
        by a 2D circle.

        input: list of 2-tuples or np.array of shape (n, 2). 
        output: an instance of CircularStroke class.
        """
        data = np.asarray(data)
        data_x = data[:,0]
        data_y = data[:,1]
        n = len(data)
        if mean_is_zero:
            mean_x = 0
            mean_y = 0
        else:
            mean_x = data_x.mean()
            mean_y = data_y.mean()
            data_x = data_x - mean_x
            data_y = data_y - mean_y

        # One can show that the solution of linear system below
        # gives the solution to least squares problem
        #
        # (xi - x0)^2 + (yi - y0)2 - R^2 --> min
        #
        # knowing that mean(xi) == mean(yi) == 0.

        su2 = (data_x*data_x).sum()
        sv2 = (data_y*data_y).sum()
        su3 = (data_x*data_x*data_x).sum()
        sv3 = (data_y*data_y*data_y).sum()
        suv = (data_x*data_y).sum()
        suvv = (data_x*data_y*data_y).sum()
        svuu = (data_y*data_x*data_x).sum()

        A = np.array([
                [su2, suv],
                [suv, sv2]
            ])

        B = np.array([[(su3 + suvv)/2.0], [(sv3 + svuu)/2.0]])

        C = np.linalg.solve(A, B)
        r2 = (C[0]*C[0]) + (C[1]*C[1]) + (su2 + sv2)/n

        circle = CircularStroke()
        circle.radius = sqrt(r2)
        center = C[:2].T[0] + np.array([mean_x, mean_y])
        circle.center_x, circle.center_y = center

        data_r2 = data_x**2 + data_y**2
        min_r2, max_r2 = data_r2.min(), data_r2.max()
        delta1 = abs(sqrt(min_r2) - circle.radius)
        delta2 = abs(sqrt(max_r2) - circle.radius)
        delta = max(delta1, delta2)

        #min_x, max_x = data_x.min(), data_x.max()
        #min_y, max_y = data_y.min(), data_y.max()
        #diam = max(max_x - min_x, max_y - min_y)
        diam = 2*circle.radius

        rel_delta = delta / diam
        print("D:", delta, rel_delta)
        if delta < 30.0 or rel_delta < 0.1:
            return circle

        return None

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
            stroke.paint(painter)

    def _get_last_stroke(self):
        if self.strokes:
            return self.strokes[-1]
        else:
            stroke = SegmentStroke()
            self.strokes.append(stroke)
            return self.strokes[-1]

    def _new_stroke(self):
        stroke = SegmentStroke()
        self.strokes.append(stroke)
        return self.strokes[-1]

    def _recognize_stroke(self):
        if not self.strokes:
            return
        stroke = self.strokes[-1]
        recognized = stroke.recognize_any()
        if recognized:
            self.strokes[-1] = recognized
        return (recognized is not None)
    
    def _update(self):
        self._update_pixmap()
        self.update(self.rect())

    def tabletEvent(self, ev):

        t = ev.type()
        if t == QtCore.QEvent.TabletPress:
            self.device_down = True
            self._new_stroke().add_point(ev.posF())
        elif t == QtCore.QEvent.TabletRelease:
            self.device_down = False
            recognized = self._recognize_stroke()
            if recognized:
                self._update()
        elif t == QtCore.QEvent.TabletMove:
            if self.device_down:
                #print(ev.posF())
                self._get_last_stroke().add_point(ev.posF())
                self._update()

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

