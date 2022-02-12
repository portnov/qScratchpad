#!/usr/bin/python3

import sys
import numpy as np
import scipy.interpolate
from math import sqrt
from PyQt5 import QtGui, QtWidgets, QtCore

import nurbs

class Stroke(object):
    def paint(self, painter):
        raise Exception("Not implemented")

class SegmentStroke(Stroke):
    def __init__(self):
        self.points = []
        self.weights = []
        self.is_finished = False

    def add_point(self, point, weight):
        self.points.append(point)
        self.weights.append(weight)

    def paint(self, painter):
        path = QtGui.QPainterPath(self.points[0])
        for pt in self.points[1:]:
            path.lineTo(pt)
        painter.drawPath(path)

    def recognize_any(self):
        points = [(pt.x(), pt.y()) for pt in self.points]
        bezier = BezierStroke.recognize(points, weights=None, smoothing=None, degree=1)
        if bezier is not None:
            return bezier
        bezier = BezierStroke.recognize(points, weights=None, degree=3)
        if bezier is not None:
            return bezier
        circle = CircularStroke.recognize(points)
        if circle:
            return circle
        print("not recognized")

class CircularStroke(Stroke):
    def __init__(self):
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius = 0.0
        self.is_finished = False

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
        if delta < 30.0 or rel_delta < 0.1:
            print("Circle")
            circle.is_finished = True
            return circle

        return None

class BezierStroke(Stroke):
    def __init__(self, curve, degree = 3):
        self.segments = curve.to_bezier_segments()
        self.degree = degree
        self.is_finished = False

    @classmethod
    def recognize(cls, points, weights, smoothing = 0.1, degree = 3):
        points = np.asarray(points)
        if weights is not None and len(points) != len(weights):
            raise Exception("Number of weights must be equal to number of points")

        filter_doubles = 0.1
        good = np.where(np.linalg.norm(np.diff(points, axis=0), axis=1) > filter_doubles)
        points = np.r_[points[good], points[-1][np.newaxis]]
        if weights is not None:
            weights = np.r_[weights[good], weights[-1]]

        points = points.T

        kwargs = dict()
        kwargs['k'] = degree
        if smoothing is not None:
            kwargs['s'] = smoothing
        kwargs['full_output'] = True
        if weights is not None:
            kwargs['w'] = np.asarray(weights)

        result = scipy.interpolate.splprep(points, **kwargs)
        (tck, u), fp, ier, msg = result[:4]
        if ier > 0:
            print(msg)
            return None
        knotvector = tck[0]
        control_points = np.stack(tck[1]).T
        degree = tck[2]

        if degree == 3:
            ok = (fp < 0.1)
        elif degree == 1:
            n = len(control_points)
            print("B1", n, fp)
            ok = (n < 5)# and (fp < 40.0)
        else:
            ok = False

        if ok:
            curve = nurbs.SvNurbsCurve(degree, knotvector, control_points)
            stroke = BezierStroke(curve, degree)
            stroke.is_finished = True
            print("Bezier", degree, len(control_points))
            return stroke

    def paint(self, painter):
        if not self.segments:
            return
        pt0 = self.segments[0].get_control_points()[0]
        path = QtGui.QPainterPath(QtCore.QPointF(pt0[0], pt0[1]))
        for segment in self.segments:
            if self.degree == 1:
                pt = segment.get_control_points()[-1]
                path.lineTo(pt[0], pt[1])
            elif self.degree == 3:
                ct_points = [QtCore.QPointF(p[0], p[1]) for p in segment.get_control_points()[1:]]
                path.cubicTo(ct_points[0], ct_points[1], ct_points[2])
        painter.drawPath(path)

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
        self._current_stroke = None
        self._redraw_pixmap()

    def _new_pixmap(self, w, h):
        #if (w, h) != (self._prev_width, self._prev_height):
        dpr = self.devicePixelRatioF()
        print("New pixmap", w, h)
        pixmap = QtGui.QPixmap(round(self.width() * dpr), round(self.height() * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(QtCore.Qt.white)
        return pixmap
    
    def _get_pixmap(self, w, h):
        is_empty = False
        if (w, h) != (self._prev_width, self._prev_height):
            self.pixmap = self._new_pixmap(w, h)
            self._prev_width, self._prev_height = w, h
            is_empty = True
        return self.pixmap, is_empty

    def _setup_painter(self, pixmap):
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.black)
        return painter

    def _redraw_pixmap(self):
        pixmap = self._new_pixmap(self.width(), self.height())
        painter = self._setup_painter(pixmap)
        self._paint_on_pixmap(painter)
        painter.end()
        self.pixmap = pixmap

    def _paint_on_pixmap(self, painter):
        for stroke in self.strokes:
            if stroke.is_finished:
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
        return stroke

    def _recognize_stroke(self):
        if not self.strokes:
            return
        stroke = self._current_stroke
        recognized = stroke.recognize_any()
        if recognized is not None:
            self._current_stroke = recognized
        return recognized

    def _begin_stroke(self, stroke):
        self._current_stroke = stroke

    def _end_stroke(self):
        self.strokes.append(self._current_stroke)
        pixmap, is_empty = self._get_pixmap(self.width(), self.height())
        painter = self._setup_painter(pixmap)
        if is_empty:
            self._paint_on_pixmap(painter)
        self._current_stroke.paint(painter)
        self._current_stroke.is_finished = True
        self.pixmap = pixmap
        self._current_stroke = None
        self.update(self.rect())
    
    def _update_stroke(self, stroke):
        self._current_stroke = stroke
        self.update(self.rect())

    def tabletEvent(self, ev):

        t = ev.type()
        if t == QtCore.QEvent.TabletPress:
            self.device_down = True
            stroke = self._new_stroke()
            stroke.add_point(ev.posF(), ev.pressure())
            self._begin_stroke(stroke)
        elif t == QtCore.QEvent.TabletRelease:
            self.device_down = False
            recognized = self._recognize_stroke()
            if recognized is not None:
                self._update_stroke(recognized)
            self._end_stroke()
        elif t == QtCore.QEvent.TabletMove:
            if self.device_down:
                #print(ev.posF())
                stroke = self._current_stroke
                stroke.add_point(ev.posF(), ev.pressure())
                self.update()

    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        dpr = self.devicePixelRatioF()
        pixmap_portion = QtCore.QRect(ev.rect().topLeft()*dpr, ev.rect().size()*dpr)
        painter.drawPixmap(ev.rect().topLeft(), self.pixmap, pixmap_portion)
        if self._current_stroke is not None:
            self._current_stroke.paint(painter)

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

