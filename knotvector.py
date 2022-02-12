# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

from collections import defaultdict
import numpy as np

def normalize(knot_vector):
    """ Normalizes the input knot vector to [0, 1] domain.

    :param knot_vector: knot vector to be normalized
    :type knot_vector: np.array of shape (X,)
    :return: normalized knot vector
    :rtype: np.array
    """
    if not isinstance(knot_vector, np.ndarray):
        knot_vector = np.array(knot_vector)
    m = knot_vector.min()
    M = knot_vector.max()
    if m >= M:
        raise Exception("All knot values are equal")
    return (knot_vector - m) / (M - m)

def find_multiplicity(knot_vector, u, tolerance=1e-6):
    pairs = to_multiplicity(knot_vector, tolerance)
    #print(f"kv {knot_vector} => {pairs}")
    for k, count in pairs:
        if tolerance is None:
            if k == u:
                return count
        else:
            if abs(k - u) < tolerance:
                return count
    return 0

def find_span(knot_vector, num_ctrlpts, knot):
    span = 0  # Knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1

    return span - 1

def insert(knot_vector, u, count=1):
    idx = np.searchsorted(knot_vector, u)
    result = knot_vector.tolist()
    for i in range(count):
        result.insert(idx, u)
        #result = np.insert(result, idx, u)
    return np.asarray(result)

def check(degree, knot_vector, num_ctrlpts):
    """ Checks the validity of the input knot vector.

    Please refer to The NURBS Book (2nd Edition), p.50 for details.

    :param degree: degree of the curve or the surface
    :type degree: int
    :param knot_vector: knot vector to be checked
    :type knot_vector: np.array of shape (X,)
    :param num_ctrlpts: number of control points
    :type num_ctrlpts: int
    :return: String with error description, if the knotvector is invalid;
            None, if the knotvector is valid.
    """
    if not isinstance(knot_vector, (list, tuple, np.ndarray)):
        raise TypeError("Knot vector must be a list, tuple, or numpy array")
    if knot_vector is None or len(knot_vector) == 0:
        raise ValueError("Input knot vector cannot be empty")

    # Check the formula; m = p + n + 1
    m = len(knot_vector)
    rhs = degree + num_ctrlpts + 1
    if m != rhs:
        return f"Knot vector has invalid length {m}; for degree {degree} and {num_ctrlpts} control points it must have {rhs} items"

    # Check ascending order
    prev_knot = knot_vector[0]
    for i, knot in enumerate(knot_vector):
        if prev_knot > knot:
            print(knot_vector)
            return f"Knot vector items are not all non-decreasing: u[{i-1}] = {prev_knot} > u[{i}] = {knot}"
        prev_knot = knot

    return None

def get_internal_knots(knot_vector, output_multiplicity = False, tolerance=1e-6):
    pairs = to_multiplicity(knot_vector)
    internal = pairs[1:-1]
    if output_multiplicity:
        return internal
    else:
        return [u for u,_ in internal]

def to_multiplicity(knot_vector, tolerance=1e-6):
    count = 0
    prev_u = None
    result = []
    for u in knot_vector:
        if prev_u is None:
            last_match = False
        else:
            last_match = abs(u - prev_u) < tolerance
            #print(f"Match: {u} - {prev_u} = {abs(u - prev_u)}, => {last_match}")
        if prev_u is None:
            count = 1
        elif last_match:
            count += 1
        else:
            result.append((prev_u, count))
            count = 1
        prev_u = u

    if last_match:
        result.append((u, count))
    else:
        result.append((u, 1))
    return result

def from_multiplicity(pairs):
    result = []
    for u, count in pairs:
        result.extend([u] * count)
    return np.array(result)

