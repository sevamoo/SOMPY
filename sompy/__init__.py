
from logging.config import dictConfig
import matplotlib

#matplotlib.use('Agg')  # Use whatever backend is available

dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "root": {
        "level": "NOTSET",
        "handlers": ["console"]
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "basic"
        }
    },
    "formatters": {
        "basic": {
            "format": '%(message)s'
        }
    }
})



from .sompy import SOMFactory
from .visualization import *
