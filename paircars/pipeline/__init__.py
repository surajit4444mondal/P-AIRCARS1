import os

os.environ["PYTHONWARNINGS"] = "ignore"
import logging
from casatasks import casalog
from astropy.utils import iers

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("tzlocal").setLevel(logging.ERROR)
try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass
iers.conf.auto_download = False
iers.conf.auto_max_age = None
