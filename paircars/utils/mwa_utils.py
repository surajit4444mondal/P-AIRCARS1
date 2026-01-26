import types
import psutil
import numpy as np
import glob
import os
import traceback
import warnings
import astropy.units as u
import requests
from urllib.request import urlretrieve
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
    return int(round(freq // 1.28))


def get_MWA_coarse_chan(msname):
    """
    Get MWA coarse channel number

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    int
        Coarse channel corresponding to central frequency of the measurement set
    """
    msmd = msmetadata()
    msmd.open(msname)
    meanfreq = msmd.meanfreq(0, unit="MHz")
    msmd.close()
    ncoarse = freq_to_MWA_coarse(meanfreq)
    return ncoarse


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
            if end_chan > start_chan:
                coarse_chans.append([start_chan, end_chan])
    return coarse_chans


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
    chanres = msmd.chanres(0, unit="MHz")[0]
    nchan = msmd.nchan(0)
    msmd.close()
    msmd.done()
    n_per_coarse_chan = int(1.28 / chanres)
    n_edge_chan = max(1, int(0.16 / chanres))
    spw = ""
    for i in range(0, int(nchan / n_per_coarse_chan), n_per_coarse_chan):
        if i == i + n_edge_chan - 1:
            spw += f"{i};"
        else:
            spw += f"{i}~{i+n_edge_chan-1};"
        if n_edge_chan >= 1:
            spw += f"{i+int(nchan/2)-1};"
        if i + nchan - n_edge_chan == i + nchan - 1:
            spw += f"{i+nchan-1};"
        else:
            spw += f"{i+nchan-n_edge_chan}~{i+nchan-1};"
    if spw != "":
        spw = f"0:{spw[:-1]}"
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
    chanres = msmd.chanres(0, unit="MHz")[0]
    nchan = msmd.nchan(0)
    msmd.close()
    msmd.done()
    n_per_coarse_chan = int(1.28 / chanres)
    n_edge_chan = int(0.16 / chanres)
    spw = "0:"
    for i in range(0, nchan, n_per_coarse_chan):
        if n_edge_chan > 1:
            spw += f"{i+n_edge_chan}~{i+int(n_chan/2)-1};{i+int(n_chan/2)+1}~{i+nchan-n_edge_chan};"
        else:
            spw += f"{i+n_edge_chan}~{i+nchan-n_edge_chan};"
    spw = spw[:-1]
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


def download_MWA_metafits(OBSID, outdir="."):
    """
    Download MWA metafits file for a given OBSID.

    Parameters
    ----------
    OBSID : int
        MWA observation ID
    outdir : str
        Output directory

    Returns
    -------
    str or None
        Path to metafits file or None if failed
    """
    os.makedirs(outdir, exist_ok=True)
    metafits = os.path.join(outdir, f"{OBSID}.metafits")
    if os.path.isfile(metafits):
        return metafits
    url = f"https://ws.mwatelescope.org/metadata/fits?obs_id={OBSID}"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(metafits, "wb") as f:
                    f.write(r.content)
                return metafits
        except Exception:
            pass
    print(f"Metafits file could not be downloaded after {max_tries} tries.")
    return None


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
