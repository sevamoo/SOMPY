import numpy as np
import sys
import inspect


class NormalizatorFactory(object):

    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % type_name)


class Normalizator(object):

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


class VarianceNormalizator(Normalizator):

    name = 'var'

    def _mean_and_standard_dev(self, data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        me, st = self._mean_and_standard_dev(data)
        return (data-me)/st

    def normalize_by(self, raw_data, data):
        me, st = self._mean_and_standard_dev(raw_data)
        return (data-me)/st

    def denormalize_by(self, data_by, n_vect):
        me, st = self._mean_and_standard_dev(data_by)
        return n_vect * st + me


class RangeNormalizator(Normalizator):

    name = 'range'


class LogNormalizator(Normalizator):

    name = 'log'


class LogisticNormalizator(Normalizator):

    name = 'logistic'


class HistDNormalizator(Normalizator):

    name = 'histd'


class HistCNormalizator(Normalizator):

    name = 'histc'
