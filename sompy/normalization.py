import numpy as np
import sys
import inspect


class NormalizerFactory(object):

    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % type_name)


class Normalizer(object):

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


class VarianceNormalizer(Normalizer):

    name = 'var'

    def _mean_and_standard_dev(self, data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        me, st = self._mean_and_standard_dev(data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def normalize_by(self, raw_data, data):
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def denormalize_by(self, data_by, n_vect):
        me, st = self._mean_and_standard_dev(data_by)
        return n_vect * st + me


class RangeNormalizer(Normalizer):

    name = 'range'


class LogNormalizer(Normalizer):

    name = 'log'


class LogisticNormalizer(Normalizer):

    name = 'logistic'


class HistDNormalizer(Normalizer):

    name = 'histd'


class HistCNormalizer(Normalizer):

    name = 'histc'
