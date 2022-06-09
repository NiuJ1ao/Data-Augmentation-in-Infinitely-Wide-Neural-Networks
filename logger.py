import logging
from logging.handlers import RotatingFileHandler

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG

class context_disabled():
    def __enter__(self):
        disabled()
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        enabled()
             
def disabled():
    return logging.disable(logging.CRITICAL)

def enabled():
    return logging.disable(logging.NOTSET)
        
# class logger():
#     logger = logging.getLogger("test")

def init_logger(log_level=logging.INFO):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s] (%(module)s:%(lineno)d) %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def get_logger():
    return logging.getLogger(__name__)
