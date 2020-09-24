import numpy as np


def length(v):
    return np.sqrt(np.sum(np.array(v) ** 2))


def normalize(v):
    return v / length(v)


def translate(point_set, v):
    return np.array([p + v for p in point_set])


def reflexion(point_set, axis):
    def reflexion_point(p, n):
        xp = p[0] + 2 * axis[0] * (axis[2] - n.dot(p)) / n.dot(n)
        yp = p[1] + 2 * axis[1] * (axis[2] - n.dot(p)) / n.dot(n)
        return np.array([xp, yp])

    return np.array([reflexion_point(p, np.array([axis[0], axis[1]])) for p in point_set])


def rotate(point_set, angle, centre):
    def rotate_point(p, a, c):
        xp = (p[0] - c[0]) * np.cos(a) - (p[1] - c[1]) * np.sin(a) + c[0]
        yp = (p[1] - c[1]) * np.cos(a) + (p[0] - c[0]) * np.sin(a) + c[1]
        return np.array([xp, yp])

    return np.array([rotate_point(p, angle, centre) for p in point_set])


def set_translation(set_size, max_trl, v, pt_per_set):
    assert (length(v) != 0)
    dim = np.shape(v)[0]
    v = normalize(v)
    final_set = []
    points = np.random.random((pt_per_set, dim))
    for i in range(set_size):
        trl = (np.random.random() * 2 * max_trl) * v
        final_set.append(translate(points, trl))
    return np.array(final_set)


def set_rotation(set_size, centre, pt_per_set):
    final_set = []
    points = np.random.random((pt_per_set, 2))
    for i in range(set_size):
        angle = np.random.random() * 2 * np.pi
        final_set.append(rotate(points, angle, centre))
    return np.array(final_set)


def set_isometric(set_size, pt_per_set):
    # axis : ax+by = c
    points = np.random.random((pt_per_set, 2))
    final_set = []
    for i in range(set_size):
        axis1, axis2, axis3 = np.random.random(3), np.random.random(3), np.random.random(3)
        final_set.append(reflexion(reflexion(reflexion(points, axis1), axis2), axis3))
    return np.array(final_set)
