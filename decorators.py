import logging
from functools import wraps
from time import time


def timeit(log_level=logging.INFO, alternative_title=None):
    def wrap(f):
        @wraps(f)  # keeps the f.__name__ outside the wrapper
        def wrapped_f(*args, **kwargs):
            t0 = time()
            title = alternative_title or f.__name__
            half_splitter = '-' * ((70 - 2 - len(title))/2)

            logging.log(log_level, "%s %s %s" % (half_splitter, title, half_splitter))

            f(*args, **kwargs)

            ts = round(time() - t0, 3)
            logging.getLogger().log(log_level, "Total time elapsed: %f seconds" % ts)
            logging.getLogger().log(log_level, '-'*70)

        return wrapped_f
    return wrap

