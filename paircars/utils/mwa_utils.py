import types
import psutil
import numpy as np
import glob
import os
import traceback
import warnings
import astropy.units as u
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from astroquery.jplhorizons import Horizons
from astropy.visualization import ImageNormalize, LogStretch
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from casatools import msmetadata, ms as casamstool, table
from datetime import datetime as dt, timedelta
from .basic_utils import *
from .resource_utils import *
from .udocker_utils import *
from .ms_metadata import *
from .image_utils import *

warnings.simplefilter("ignore", category=FITSFixedWarning)


def get_MWA_OBSID(msname):
    """
    Get MWA OBSID from ms
    
    Parameters
    ----------
    msname : str
        Measurement set
        
    Returns
    -------
    int
        OBSid
    """
    msmd=msmetadata()
    msmd.open(msname)
    start_time=msmd.timerangeforobs(0)['begin']['m0']['value']*86400
    msmd.close()
    t = Time(start_time * u.s, format="mjd", scale="utc")
    gps = t.gps
    obsid = int((gps//8)*8)
    return obsid


def get_bad_chans(msname):
    """
    Get bad channels to flag

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    str
        SPW string of bad channels
    """
    msmd = msmetadata()
    msmd.open(msname)
    chanfreqs = msmd.chanfreqs(0) / 10**6
    msmd.close()
    msmd.done()
    bandname = get_band_name(msname)
    if bandname == "U":
        bad_freqs = [
            (-1, 580),
            (925, 960),
            (1010, -1),
        ]
    elif bandname == "L":
        bad_freqs = [
            (-1, 879),
            (925, 960),
            (1166, 1186),
            (1217, 1237),
            (1242, 1249),
            (1375, 1387),
            (1526, 1626),
            (1681, -1),
        ]
    else:
        print("MeerKAT data is not in UHF or L-band.")
        bad_freqs = []
    if min(chanfreqs) <= bad_freqs[0][1] and max(chanfreqs) >= bad_freqs[-1][0]:
        spw = "0:"
        count = 0
        for freq_range in bad_freqs:
            start_freq = freq_range[0]
            end_freq = freq_range[1]
            if start_freq == -1:
                start_chan = 0
            else:
                start_chan = np.argmin(np.abs(start_freq - chanfreqs))
            if count > 0 and start_chan <= end_chan:
                break
            if end_freq == -1:
                end_chan = len(chanfreqs) - 1
            else:
                end_chan = np.argmin(np.abs(end_freq - chanfreqs))
            if end_chan > start_chan:
                spw += str(start_chan) + "~" + str(end_chan) + ";"
            else:
                spw += str(start_chan) + ";"
            count += 1
        spw = spw[:-1]
    else:
        spw = ""
    return spw


def get_good_chans(msname):
    """
    Get good channel range to perform gaincal

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    str
        SPW string
    """
    msmd = msmetadata()
    msmd.open(msname)
    chanfreqs = msmd.chanfreqs(0) / 10**6
    meanfreq = msmd.meanfreq(0) / 10**6
    msmd.close()
    msmd.done()
    bandname = get_band_name(msname)
    if bandname == "U":
        good_freqs = [(580, 620)]  # For UHF band
    elif bandname == "L":
        good_freqs = [(890, 920)]  # For L band
    else:
        good_freqs = []  # For S band
    if min(chanfreqs) <= good_freqs[0][1] and max(chanfreqs) >= good_freqs[-1][0]:
        spw = "0:"
        for freq_range in good_freqs:
            start_freq = freq_range[0]
            end_freq = freq_range[1]
            start_chan = np.argmin(np.abs(start_freq - chanfreqs))
            end_chan = np.argmin(np.abs(end_freq - chanfreqs))
            spw += str(start_chan) + "~" + str(end_chan) + ";"
        spw = spw[:-1]
    else:
        spw = f"0:0~{len(chanfreqs)-1}"
    return spw

# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]

