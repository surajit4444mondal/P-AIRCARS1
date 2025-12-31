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
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass
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
    msmd = msmetadata()
    msmd.open(msname)
    start_time = msmd.timerangeforobs(0)["begin"]["m0"]["value"] * 86400
    msmd.close()
    t = Time(start_time * u.s, format="mjd", scale="utc")
    gps = t.gps
    obsid = int((gps // 8) * 8)
    return obsid


def get_ncoarse(msname):
    """
    Get number of coarse channels

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    int
        Number of coarse channels
    """
    msmd = msmetadata()
    msmd.open(msname)
    freqs = msmd.chanfreqs(0, unit="MHz")
    bw = max(freqs) - min(freqs)
    ncoarse = int(bw / 1.28)
    return ncoarse


def freq_to_MWA_coarse(freq):
    """
    Frequency to MWA coarse channel conversion.

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    int
        MWA coarse channel number
    """
    return int(freq // 1.28)


def get_MWA_bad_freqs():
    """
    Get bad frequencies of the MWA band

    Returns
    -------
    list
        Bad frequency range in MHz
    """
    coarse_channels = np.arange(55, 235)
    bad_freqs = []
    for coarse_chan in coarse_channels:
        cent_freq = coarse_chan * 1.28
        start_freq = cent_freq - 0.64
        end_freq = cent_freq + 0.64
        bad_freqs.append([start_freq, start_freq + 0.16])
        bad_freqs.append([cent_freq])
        bad_freqs.append([end_freq - 0.16, end_freq])
    return bad_freqs


def get_MWA_coarse_bands(msname):
    """
    Get coarse channel bands of the MWA

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    list
        Coarse channel list
    """
    msmd = msmetadata()
    msmd.open(msname)
    freqs = msmd.chanfreqs(0, unit="MHz")
    freqres = msmd.chanres(0, unit="MHz")[0]
    msmd.close()
    nchan_coarse = int(1.28 / freqres)
    start_ms_freq = round(np.nanmin(freqs), 2)
    end_ms_freq = round(np.nanmax(freqs), 2)
    coarse_channels = np.arange(55, 235)
    coarse_chans = []
    for coarse_chan in coarse_channels:
        cent_freq = round(coarse_chan * 1.28, 2)
        start_freq = round(cent_freq - 0.64, 2)
        end_freq = round(cent_freq + 0.64, 2)
        if cent_freq >= start_ms_freq and cent_freq <= end_ms_freq:
            start_chan = np.argmin(np.abs(start_freq - freqs))
            end_chan = np.argmin(np.abs(end_freq - freqs))
            if start_chan > 0:
                start_chan = max(
                    (start_chan // nchan_coarse) * nchan_coarse, nchan_coarse
                )
            if end_chan > 0:
                end_chan = max((end_chan // nchan_coarse) * nchan_coarse, nchan_coarse)
            coarse_chans.append([start_chan, end_chan])
    return coarse_chans


def get_MWA_good_freqs():
    """
    Get good frequencies of the MWA band

    Returns
    -------
    list
        Good frequency range in MHz
    """
    coarse_channels = np.arange(55, 235)
    good_freqs = []
    for coarse_chan in coarse_channels:
        cent_freq = coarse_chan * 1.28
        start_freq = cent_freq - 0.64
        end_freq = end_freq + 0.64
        good_freqs.append([start_freq + 0.16, cent_freq - 0.01])
        good_freqs.append([cent_freq + 0.01, end_freq - 0.16])
    return good_freqs


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
    bad_freqs = get_MWA_bad_freqs()
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
    Get good channel range of MWA

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
    bad_freqs = get_MWA_good_freqs()
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


def get_mwa_bad_ants(metafits):
    """
    Function to determine non-working MWA tiles for a observation

    Parameters
    ----------
    metafits : str
        Name of the metafits file

    Returns
    -------
    str
        Non-working antenna names
    """
    data = fits.getdata(metafits)
    flags = np.array(data["Flag"])
    tiles = np.array(data["TileName"])
    pos = np.where(flags == 1)
    bad_tiles = tiles[pos]
    bad_tiles = np.unique(bad_tiles)
    bad_antennas = ""
    if len(bad_tiles) > 0:
        for ant in bad_tiles:
            bad_antennas += str(ant) + ","
        bad_antennas = bad_antennas[:-1]
    return bad_antennas


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
