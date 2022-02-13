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
    
    def transformed(self, transform):
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

    def transformed(self, transform):
        stroke = SegmentStroke()
        stroke.points = [transform.map(p) for p in self.points]
        stroke.weights = self.weights
        stroke.is_finished = self.is_finished
        return stroke

    def recognize_any(self, prefer_straight=False, use_circles=True):
        points = [(pt.x(), pt.y()) for pt in self.points]
        if use_circles:
            circle = CircularStroke.recognize(points)
            if circle:
                return circle
        if prefer_straight:
            bezier = BezierStroke.recognize(points, weights=self.weights, smoothing=None, degree=1)
            if bezier is not None:
                return bezier
        else:
            bezier = BezierStroke.recognize(points, weights=self.weights, smoothing=0.03, degree=3)
            if bezier is not None:
                return bezier
        print("not recognized")

class CircularStroke(Stroke):
    def __init__(self):
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius = 0.0
        self.is_finished = False

    def transformed(self, transform):
        stroke = CircularStroke()
        ct = transform.map(QtCore.QPointF(self.center_x, self.center_y))
        stroke.center_x = ct.x()
        stroke.center_y = ct.y()
        scale = sqrt(transform.determinant())
        stroke.radius = scale *self.radius
        stroke.is_finished = self.is_finished
        return stroke

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

        delta = abs((data_x**2 + data_y**2).mean() - r2)
        sigma = sqrt(delta)

        diam = 2*circle.radius

        rel_delta = sigma / diam
        #print(f"C {sigma}, R2={r2}, Diam={diam}, {rel_delta}")
        if rel_delta < 0.5:
            print(f"Circle: R {circle.radius}")
            circle.is_finished = True
            return circle

        return None

def calc_diameter(points):
    min_ = points.min(axis=0)
    max_ = points.max(axis=0)
    return (max_ - min_).max()

MIN_WEIGHT = 0.001

class BezierStroke(Stroke):
    def __init__(self, curve=None, degree = 3):
        if curve is not None:
            self.segments = curve.to_bezier_segments()
        else:
            self.segments = []
        self.degree = degree
        self.is_finished = False

    def transformed(self, transform):
        stroke = BezierStroke(degree=self.degree)
        dx, dy = transform.dx(), transform.dy()
        matrix = np.array([[transform.m11(), transform.m12()], [transform.m21(), transform.m22()]])
        stroke.segments = [s.transformed(matrix, (dx, dy)) for s in self.segments]
        stroke.is_finished = self.is_finished
        return stroke

    @classmethod
    def recognize(cls, points, weights, smoothing = 0.1, degree = 3):
        points = np.asarray(points)
        if len(points) < degree+1:
            return None
        if weights is not None:
            weights = np.asarray(weights)
        if weights is not None and len(points) != len(weights):
            raise Exception("Number of weights must be equal to number of points")

        filter_doubles = 0.1
        good = np.where(np.linalg.norm(np.diff(points, axis=0), axis=1) > filter_doubles)
        points = np.r_[points[good], points[-1][np.newaxis]]
        if weights is not None:
            weights = np.r_[weights[good], weights[-1]]
            weights[abs(weights) < MIN_WEIGHT] = MIN_WEIGHT

        points = points.T

        kwargs = dict()
        kwargs['k'] = degree
        if smoothing is not None:
            smoothing = smoothing * calc_diameter(points)
            kwargs['s'] = smoothing
        kwargs['full_output'] = True
        if weights is not None:
            kwargs['w'] = np.asarray(weights)

        try:
            result = scipy.interpolate.splprep(points, **kwargs)
        except TypeError as e:
            print(e)
            return None
        (tck, u), fp, ier, msg = result[:4]
        if ier > 0:
            print(msg)
            return None
        knotvector = tck[0]
        control_points = np.stack(tck[1]).T
        degree = tck[2]

        if degree == 3:
            #print("B3", fp, smoothing)
            ok = (fp < smoothing * 1.5)
        elif degree == 1:
            n = len(control_points)
            print("B1", n, fp)
            ok = (n < 30)# and (fp < 40.0)
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
        self._prev_translate = None
        self.panned_pixmap = None

        self.device_down = False
        self.tablet_used = False
        self.pan_start_pos = None
        self.prev_pan_pos = None
        self.current_pan_start = None
        self.current_pan_translation = None
        self.transformation = QtGui.QTransform()
        self.pixmap = QtGui.QPixmap()
        self._current_stroke = None
        self._redraw_pixmap()

    def _new_pixmap(self, w, h):
        dpr = self.devicePixelRatioF()
        #print("New pixmap", w, h)
        pixmap = QtGui.QPixmap(round(self.width() * dpr), round(self.height() * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(QtCore.Qt.white)
        return pixmap
    
    def _get_pixmap(self, w, h):
        is_empty = False
        if self.pixmap is None or (w, h) != (self._prev_width, self._prev_height):
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

    def _paint_on_pixmap(self, painter, translated=True):
        for stroke in self.strokes:
            if stroke.is_finished:
                if translated:
                    stroke.transformed(self.transformation).paint(painter)
                else:
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
        if stroke is None:
            return
        prefer_straight = QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier
        use_circles = QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier
        recognized = stroke.recognize_any(prefer_straight, use_circles)
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
        self._current_stroke.transformed(self.transformation).paint(painter)
        self._current_stroke.is_finished = True
        self.pixmap = pixmap
        self._current_stroke = None
        self.update(self.rect())
    
    def _update_stroke(self, stroke):
        self._current_stroke = stroke
        self.update(self.rect())

    def _to_scene(self, pos):
        inv, ok = self.transformation.inverted()
        return inv.map(pos)
    
    def translation(self):
        return QtCore.QPointF(self.transformation.dx(), self.transformation.dy())
    
    def _on_press(self, is_pan, pos, pressure=1.0):
        if not is_pan:
            self.device_down = True
            stroke = self._new_stroke()
            stroke.add_point(self._to_scene(pos), pressure)
            self._begin_stroke(stroke)
        else:
            self.pan_start_pos = pos - self.translation()
            self.prev_pan_pos = pos
            self.current_pan_start = pos

            self._redraw_pixmap()
            self.panned_pixmap = self.pixmap

    def _on_release(self):
        if not self.pan_start_pos and self._current_stroke is not None:
            self.device_down = False
            recognized = self._recognize_stroke()
            if recognized is not None:
                self._update_stroke(recognized)
            self._end_stroke()
        if self.pan_start_pos:
            self.pan_start_pos = None
            self.prev_pan_pos = None
            self.panned_pixmap = None
            self.current_pan_start = None
            self.current_pan_translation = None
            self._redraw_pixmap()

    def _on_drag(self, is_pan, pos, pressure=1.0):
        do_update = False
        if not is_pan:
            if self.device_down:
                stroke = self._current_stroke
                stroke.add_point(self._to_scene(pos), pressure)
                do_update = True

        if self.pan_start_pos:
            delta = pos - self.prev_pan_pos
            delta = delta / sqrt(self.transformation.determinant())
            self.prev_pan_pos = pos
            self.transformation = self.transformation.translate(delta.x(), delta.y())
            self.current_pan_translation = pos - self.current_pan_start
            do_update = True
        
        if do_update:
            self.update()

    def tabletEvent(self, ev):
        t = ev.type()
        is_pan = ev.buttons() & QtCore.Qt.MiddleButton
        if t == QtCore.QEvent.TabletPress:
            self._on_press(is_pan, ev.posF(), ev.pressure())
        elif t == QtCore.QEvent.TabletRelease:
            self._on_release()
        elif t == QtCore.QEvent.TabletMove:
            self._on_drag(is_pan, ev.posF(), ev.pressure())

    def mousePressEvent(self, ev):
        if self.tablet_used:
            ev.ignore()
            return
        is_pan = ev.buttons() & QtCore.Qt.MiddleButton
        self._on_press(is_pan, ev.pos())

    def mouseReleaseEvent(self, ev):
        if self.tablet_used:
            ev.ignore()
            return
        self._on_release()

    def mouseMoveEvent(self, ev):
        if self.tablet_used:
            ev.ignore()
            return
        is_pan = ev.buttons() & QtCore.Qt.MiddleButton
        self._on_drag(is_pan, ev.pos())

    def wheelEvent(self, ev):
      angle = ev.angleDelta().y()
      z = 1.5 ** (angle / 360.0)
      pos = self._to_scene(ev.posF())
      self.transformation = self.transformation.translate(pos.x(), pos.y()).scale(z, z).translate(-pos.x(), -pos.y())
      self._redraw_pixmap()
      self.update()

    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        dpr = self.devicePixelRatioF()
        #pixmap_portion = QtCore.QRect(ev.rect().topLeft()*dpr, ev.rect().size()*dpr)
        if self.panned_pixmap is not None:
            painter.translate(self.current_pan_translation)
            painter.drawPixmap(ev.rect().topLeft(), self.panned_pixmap)
            painter.resetTransform()
        else:
            painter.drawPixmap(ev.rect().topLeft(), self.pixmap)#, pixmap_portion)
        if self._current_stroke is not None:
            self._current_stroke.transformed(self.transformation).paint(painter)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_CompressHighFrequencyEvents)

class Application(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        QtWidgets.QApplication.__init__(self, *args, **kwargs)
        self.canvas = None

    def event(self, ev):
        t = ev.type()
        if t == QtCore.QEvent.TabletEnterProximity:
            self.canvas.tablet_used = True
            print("Tablet on")
            return True
        elif t == QtCore.QEvent.TabletLeaveProximity:
            self.canvas.tablet_used = False
            print("Tablet off")
            return True
        return super().event(ev)

if __name__ == "__main__":
    app = Application(sys.argv)
    window = MainWindow()
    app.canvas = window.canvas
    window.show()

    sys.exit(app.exec_())

