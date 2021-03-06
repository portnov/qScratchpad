# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

import numpy as np
import scipy.optimize
from math import pi, sqrt

import knotvector as sv_knotvector

NDIM=2

def from_homogenous(control_points):
    weights = control_points[:,NDIM]
    weighted = control_points[:,0:NDIM]
    points = weighted / weights[np.newaxis].T
    return points, weights

class SvNurbsBasisFunctions(object):
    def __init__(self, knotvector):
        self.knotvector = np.array(knotvector)
        self._cache = dict()

    def function(self, i, p, reset_cache=True):
        if reset_cache:
            self._cache = dict()
        def calc(us):
            value = self._cache.get((i,p, 0))
            if value is not None:
                return value

            u = self.knotvector
            if p <= 0:
                if i < 0 or i >= len(u):

                    value = np.zeros_like(us)
                    self._cache[(i,p,0)] = value
                    return value
                        
                else:

                    if i+1 >= len(u):
                        u_next = u[-1]
                        is_last = True
                    else:
                        u_next = u[i+1]
                        is_last = u_next >= u[-1]
                    if is_last:
                        c2 = us <= u_next
                    else:
                        c2 = us < u_next
                    condition = np.logical_and(u[i] <= us, c2)
                    value = np.where(condition, 1.0, 0.0)
                    self._cache[(i,p,0)] = value
                    return value

            else:
                denom1 = (u[i+p] - u[i])
                denom2 = (u[i+p+1] - u[i+1])

                if denom1 != 0:
                    n1 = self.function(i, p-1, reset_cache=False)(us)
                if denom2 != 0:
                    n2 = self.function(i+1, p-1, reset_cache=False)(us)

                if denom1 == 0 and denom2 == 0:
                    value = np.zeros_like(us)
                    self._cache[(i,p,0)] = value
                    return value
                elif denom1 == 0 and denom2 != 0:
                    c2 = (u[i+p+1] - us) / denom2
                    value = c2 * n2
                    self._cache[(i,p,0)] = value
                    return value
                elif denom1 != 0 and denom2 == 0:
                    c1 = (us - u[i]) / denom1
                    value = c1 * n1
                    self._cache[(i,p,0)] = value
                    return value
                else: # denom1 != 0 and denom2 != 0
                    c1 = (us - u[i]) / denom1
                    c2 = (u[i+p+1] - us) / denom2
                    value = c1 * n1 + c2 * n2
                    self._cache[(i,p,0)] = value
                    return value
        return calc

    def derivative(self, i, p, k, reset_cache=True):
        if reset_cache:
            self._cache = dict()

        if k == 0:
            return self.function(i, p, reset_cache=False)

        def calc(us):
            value = self._cache.get((i, p, k))
            if value is not None:
                return value
            
            n1 = self.derivative(i, p-1, k-1, reset_cache=False)(us)
            n2 = self.derivative(i+1, p-1, k-1, reset_cache=False)(us)
            u = self.knotvector

            denom1 = u[i+p] - u[i]
            denom2 = u[i+p+1] - u[i+1]

            if denom1 == 0:
                s1 = np.zeros_like(us)
            else:
                s1 = n1 / denom1

            if denom2 == 0:
                s2 = np.zeros_like(us)
            else:
                s2 = n2 / denom2

            value = p*(s1 - s2)
            self._cache[(i,p,k)] = value
            return value
        
        return calc

class CantInsertKnotException(Exception):
    pass

class BoundingBox(object):
    def __init__(self, x0, y0, width, height):
        if not isinstance(x0, (float, int)):
            raise ValueError(f"x0 = {x0} is not a number")
        if not isinstance(y0, (float, int)):
            raise ValueError(f"y0 = {y0} is not a number")
        if not isinstance(width, (float, int)):
            raise ValueError(f"width = {width} is not a number")
        if not isinstance(height, (float, int)):
            raise ValueError(f"height = {height} is not a number")
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

    def __repr__(self):
        return f"<x {self.x0}, y {self.y0}, W {self.width}, H {self.height}>"

    @staticmethod
    def calc(points):
        points = np.asarray(points)
        _min = points.min(axis=0)
        _max = points.max(axis=0)
        box = BoundingBox(_min[0], _min[1], _max[0] - _min[0], _max[1] - _min[1])
        return box

    def union(self, other):
        x0 = min(self.x0, other.x0)
        y0 = min(self.y0, other.y0)
        x1 = max(self.x0 + self.width, other.x0 + other.width)
        y1 = max(self.y0 + self.height, other.y0 + other.height)
        width = x1 - x0
        height = y1 - y0
        box = BoundingBox(x0, y0, width, height)
        return box

    @staticmethod
    def union_list(boxes):
        if not boxes:
            return BoundingBox(0, 0, 100, 100)
        result = boxes[0]
        for box in boxes[1:]:
            result = result.union(box)
        return result

class SvNurbsCurve(object):
    def __init__(self, degree, knotvector, control_points, weights=None, normalize_knots=False):
        self.control_points = np.asarray(control_points) # (k, NDIM)
        k = len(control_points)
        if weights is not None:
            self.weights = np.array(weights) # (k, )
        else:
            self.weights = np.ones((k,))
        self.knotvector = np.array(knotvector)
        if normalize_knots:
            self.knotvector = sv_knotvector.normalize(self.knotvector)
        self.degree = degree
        self.basis = SvNurbsBasisFunctions(knotvector)
        self.u_bounds = None # take from knotvector

    @staticmethod
    def make_bezier(degree, control_points):
        control_points = np.asarray(control_points)
        knotvector = sv_knotvector.generate(degree, len(control_points))
        return SvNurbsCurve(degree, knotvector, control_points)

    def get_bounding_box(self):
        return BoundingBox.calc(self.control_points)

    def transformed(self, matrix, dv):
        matrix = np.asarray(matrix)
        dv = np.asarray(dv)
        new_control_points = [matrix @ pt + dv for pt in self.control_points]
        return SvNurbsCurve(self.degree, self.knotvector, new_control_points, self.weights)

    def is_rational(self, tolerance=1e-6):
        w, W = self.weights.min(), self.weights.max()
        return (W - w) > tolerance

    def get_control_points(self):
        return self.control_points

    def get_weights(self):
        return self.weights

    def get_knotvector(self):
        return self.knotvector

    def get_degree(self):
        return self.degree

    def get_u_bounds(self):
        if self.u_bounds is None:
            m = self.knotvector[0]
            M = self.knotvector[-1]
            return (m, M)
        else:
            return self.u_bounds

    def get_homogenous_control_points(self):
        """
        returns: np.array of shape (k, 4)
        """
        points = self.get_control_points()
        weights = self.get_weights()[np.newaxis].T
        weighted = weights * points
        return np.concatenate((weighted, weights), axis=1)

    def is_bezier(self):
        k = len(self.get_control_points())
        p = self.get_degree()
        return p+1 == k

    def insert_knot(self, u_bar, count=1, if_possible=False):
        # "The NURBS book", 2nd edition, p.5.2, eq. 5.11
        N = len(self.control_points)
        u = self.get_knotvector()
        s = sv_knotvector.find_multiplicity(u, u_bar)
        #print(f"I: kv {len(u)}{u}, u_bar {u_bar} => s {s}")
        #k = np.searchsorted(u, u_bar, side='right')-1
        k = sv_knotvector.find_span(u, N, u_bar)
        p = self.get_degree()
        new_knotvector = sv_knotvector.insert(u, u_bar, count)
        control_points = self.get_homogenous_control_points()

        if (u_bar == u[0] or u_bar == u[-1]):
            if s+count > p+1:
                if if_possible:
                    count = (p+1) - s
                else:
                    raise CantInsertKnotException(f"Can't insert first/last knot t={u_bar} for {count} times")
        else:
            if s+count > p:
                if if_possible:
                    count = p - s
                else:
                    raise CantInsertKnotException(f"Can't insert knot t={u_bar} for {count} times")

        for r in range(1, count+1):
            prev_control_points = control_points
            control_points = []
            for i in range(N+1):
                #print(f"I: i {i}, k {k}, p {p}, r {r}, s {s}, k-p+r-1 {k-p+r-1}, k-s {k-s}")
                if i <= k-p+r-1:
                    point = prev_control_points[i]
                    #print(f"P[{r},{i}] := {i}{prev_control_points[i]}")
                elif k - p + r <= i <= k - s:
                    denominator = u[i+p-r+1] - u[i]
                    if abs(denominator) < 1e-6:
                        raise Exception(f"Can't insert the knot t={u_bar} for {i}th time: u[i+p-r+1]={u[i+p-r+1]}, u[i]={u[i]}, denom={denominator}")
                    alpha = (u_bar - u[i]) / denominator
                    point = alpha * prev_control_points[i] + (1.0 - alpha) * prev_control_points[i-1]
                    #print(f"P[{r},{i}]: alpha {alpha}, pts {i}{prev_control_points[i]}, {i-1}{prev_control_points[i-1]} => {point}")
                else:
                    point = prev_control_points[i-1]
                    #print(f"P[{r},{i}] := {i-1}{prev_control_points[i-1]}")
                control_points.append(point)
            N += 1

        control_points, weights = from_homogenous(np.array(control_points))
        curve = SvNurbsCurve(self.degree, new_knotvector,
                    control_points, weights)
        return curve

    def copy(self, knotvector = None, control_points = None, weights = None):
        if knotvector is None:
            knotvector = self.get_knotvector()
        if control_points is None:
            control_points = self.get_control_points()
        if weights is None:
            weights = self.get_weights()
        return SvNurbsCurve(self.get_degree(), knotvector, control_points, weights)

    def _split_at(self, t):
        t_min, t_max = self.get_u_bounds()

        # corner cases
        if t <= t_min:
            return None, (self.get_knotvector(), self.get_control_points(), self.get_weights())
        if t >= t_max:
            return (self.get_knotvector(), self.get_control_points(), self.get_weights()), None

        current_multiplicity = sv_knotvector.find_multiplicity(self.get_knotvector(), t)
        to_add = self.get_degree() - current_multiplicity # + 1
        curve = self.insert_knot(t, count=to_add)
        knot_span = np.searchsorted(curve.get_knotvector(), t)

        ts = np.full((self.get_degree()+1,), t)
        knotvector1 = np.concatenate((curve.get_knotvector()[:knot_span], ts))
        knotvector2 = np.insert(curve.get_knotvector()[knot_span:], 0, t)

        control_points_1 = curve.get_control_points()[:knot_span]
        control_points_2 = curve.get_control_points()[knot_span-1:]
        weights_1 = curve.get_weights()[:knot_span]
        weights_2 = curve.get_weights()[knot_span-1:]

        #print(f"S: ctlpts1: {len(control_points_1)}, 2: {len(control_points_2)}")
        kv_error = sv_knotvector.check(curve.get_degree(), knotvector1, len(control_points_1))
        if kv_error is not None:
            raise Exception(kv_error)
        kv_error = sv_knotvector.check(curve.get_degree(), knotvector2, len(control_points_2))
        if kv_error is not None:
            raise Exception(kv_error)

        curve1 = (knotvector1, control_points_1, weights_1)
        curve2 = (knotvector2, control_points_2, weights_2)
        return curve1, curve2

    def split_at(self, t):
        c1, c2 = self._split_at(t)
        degree = self.get_degree()

        if c1 is not None:
            knotvector1, control_points_1, weights_1 = c1
            curve1 = SvNurbsCurve(degree, knotvector1,
                        control_points_1, weights_1)
        else:
            curve1 = None

        if c2 is not None:
            knotvector2, control_points_2, weights_2 = c2

            curve2 = SvNurbsCurve(degree, knotvector2,
                        control_points_2, weights_2)
        else:
            curve2 = None

        return curve1, curve2

    def to_bezier_segments(self):
        if self.is_bezier():
            return [self]

        segments = []
        rest = self
        for u in sv_knotvector.get_internal_knots(self.get_knotvector()):
            segment, rest = rest.split_at(u)
            segments.append(segment)
        segments.append(rest)
        return segments

class LineEquation2D(object):
    def __init__(self, a, b, c):
        epsilon = 1e-8
        if abs(a) < epsilon and abs(b) < epsilon:
            raise Exception(f"Direction is (nearly) zero: {a}, {b}")
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self):
        return f"{self.a}*x + {self.b}*y + {self.c} = 0"

    def distance_to_point(self, point):
        a, b, c = self.a, self.b, self.c
        x, y = tuple(point)
        value = a*x + b*y + c
        numerator = abs(value)
        denominator = sqrt(a*a + b*b)
        return numerator / denominator

    def distance_to_points(self, points):
        return np.array([self.distance_to_point(p) for p in points])

    def projection_of_point(self, point):
        normal = np.array([self.a, self.b])
        normal = normal / np.linalg.norm(normal)
        distance = self.distance_to_point(point)
        sign = self.side_of_point(point)
        return np.array(point) - sign * distance * normal

    def projection_of_points(self, points):
        return [self.projection_of_point(p) for p in points]

    def side_of_point(self, point, eps=1e-8):
        a, b, c = self.a, self.b, self.c
        x, y = tuple(point)
        value = a*x + b*y + c
        if abs(value) < eps:
            return 0
        elif value > 0:
            return +1
        else:
            return -1

    def projection_endpoints(self, points):
        points = np.asarray(points)
        direction = np.array([self.b, -self.a])
        dots = (points * direction).sum(axis=1)
        min_idx, max_idx = dots.argmin(), dots.argmax()
        p1 = self.projection_of_point(points[min_idx])
        p2 = self.projection_of_point(points[max_idx])
        return p1, p2

    @staticmethod
    def approximate(points):
        points = np.asarray(points)
        m = len(points)
        ctr = points.mean(axis=0)
        points_centered = points - ctr
        M = points_centered.T @ points_centered
        eigenvalues, eigenvectors = np.linalg.eig(M)
        l1, l2 = eigenvalues
        if abs(l1) > abs(l2):
            v1, v2 = eigenvectors[0]
        else:
            v1, v2 = eigenvectors[1]
        a, b = v2, v1
        c = -a * ctr[0] - b * ctr[1]
        return LineEquation2D(a, b, c)

