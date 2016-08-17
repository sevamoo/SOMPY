import numpy as np
import inspect
import sys

small = .000000000001


class NeighborhoodFactory(object):

    @staticmethod
    def build(neighborhood_func):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and neighborhood_func == obj.name:
                    return obj()
        else:
            raise Exception(
                "Unsupported neighborhood function '%s'" % neighborhood_func)


class GaussianNeighborhood(object):

    name = 'gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        return np.exp(-1.0*distance_matrix/(2.0*radius**2)).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)


class BubbleNeighborhood(object):

    name = 'bubble'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        def l(a, b):
            c = np.zeros(b.shape)
            c[a-b >= 0] = 1
            return c

        return l(radius,
                 np.sqrt(distance_matrix.flatten())).reshape(dim, dim) + small

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)
