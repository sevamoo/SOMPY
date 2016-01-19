import numpy as np


def gaussian_neighborhood(distance_matrix, radius, dim):
    return np.exp(-1.0*distance_matrix/(2.0*radius**2)).reshape(dim, dim)


def bubble_neighborhood(distance_matrix, radius, dim):
    def l(a, b):
        c = np.zeros(b.shape)
        c[a-b >= 0] = 1
        return c

    return l(radius, np.sqrt(distance_matrix.flatten())).reshape(dim, dim) + .000000000001

