import types
import psutil
import numpy as np
import traceback
import warnings
import glob
import os
from casatools import msmetadata, ms as casamstool, table
from .basic_utils import *
from .ms_metadata import *
from .imaging import *

#####################################
# Calibration related
#####################################


def merge_caltables(caltables, merged_caltable, append=False, keepcopy=False):
    """
    Merge multiple same type of caltables

    Parameters
    ----------
    caltables : list
        Caltable list
    merged_caltable : str
        Merged caltable name
    append : bool, optional
        Append with exisiting caltable
    keepcopy : bool, opitonal
        Keep input caltables or not

    Returns
    -------
    str
        Merged caltable
    """
    if not isinstance(caltables, list) or len(caltables) == 0:
        print("Please provide a list of caltable.")
        return
    if os.path.exists(merged_caltable) and append:
        pass
    else:
        if os.path.exists(merged_caltable):
            os.system("rm -rf " + merged_caltable)
        if keepcopy:
            os.system("cp -r " + caltables[0] + " " + merged_caltable)
        else:
            os.system("mv " + caltables[0] + " " + merged_caltable)
        caltables.remove(caltables[0])
    if len(caltables) > 0:
        tb = table()
        for caltable in caltables:
            if os.path.exists(caltable):
                tb.open(caltable)
                tb.copyrows(merged_caltable)
                tb.close()
                if not keepcopy:
                    os.system("rm -rf " + caltable)
    return merged_caltable


def get_psf_size(msname, chan_number=-1):
    """
    Function to calculate PSF size in arcsec

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
            PSF size in arcsec
    """
    maxuv_m, maxuv_l = calc_maxuv(msname, chan_number=chan_number)
    psf = np.rad2deg(1.2 / maxuv_l) * 3600.0  # In arcsec
    return round(psf, 2)


def calc_bw_smearing_freqwidth(msname, full_FoV=False, FWHM=True):
    """
    Function to calculate spectral width to procude bandwidth smearing

    Parameters
    ----------
    msname : str
        Name of the measurement set
    full_FoV : bool, optional
        Consider smearing within solar disc or full FoV
    FWHM : bool, optional
        If using full FoV, consider upto FWHM or first null

    Returns
    -------
    float
        Spectral width in MHz
    """
    tb = table()
    tb.open(f"{msname}/SPECTRAL_WINDOW")
    freq = float(tb.getcol("REF_FREQUENCY")[0]) / 10**6
    freqres = float(tb.getcol("CHAN_WIDTH")[0][0]) / 10**6
    tb.close()
    R = 0.9
    if full_FoV:
        fov = calc_field_of_view(msname, FWHM=FWHM)  # In arcsec
    else:
        fov = 2 * calc_sun_dia(np.nanmean(freq)) * 60  # 2 times sun size
    psf = get_psf_size(msname)
    delta_nu = np.sqrt((1 / R**2) - 1) * (psf / fov) * freq
    delta_nu = ceil_to_multiple(delta_nu, freqres)
    return round(delta_nu, 2)


def calc_time_smearing_timewidth(msname, full_FoV=False, FWHM=True):
    """
    Calculate maximum time averaging to avoid time smearing over full FoV.

    Parameters
    ----------
    msname : str
        Measurement set name
    full_FoV : bool, optional
        Consider smearing within solar disc or full FoV
    FWHM : bool, optional
        If using full FoV, consider upto FWHM or first null

    Returns
    -------
    delta_t_max : float
        Maximum allowable time averaging in seconds.
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq_Hz = msmd.chanfreqs(0)[0]
    times = msmd.timesforspws(0)
    msmd.close()
    timeres = times[1] - times[0]
    c = 299792458.0  # speed of light in m/s
    omega_E = 7.2921159e-5  # Earth rotation rate in rad/s
    lam = c / freq_Hz  # wavelength in meters
    freq = freq_Hz / 10**6
    if full_FoV:
        fov = calc_field_of_view(msname, FWHM=FWHM)  # In arcsec
    else:
        fov = 2 * calc_sun_dia(np.nanmean(freq)) * 60  # 2 times sun size
    fov_deg = fov / 3600.0
    fov_rad = np.deg2rad(fov_deg)
    uv, uvlambda = calc_maxuv(msname)
    # Approximate maximum allowable time to avoid >10% amplitude loss
    delta_t_max = lam / (2 * np.pi * uv * omega_E * fov_rad)
    delta_t_max = ceil_to_multiple(delta_t_max, timeres)
    return round(delta_t_max, 2)


def max_time_solar_smearing(msname):
    """
    Max allowable time averaging to avoid solar motion smearing.

    Parameters
    ----------
    msname : str
        Measurement set name

    Returns
    -------
    t_max : float
        Maximum time averaging in seconds.
    """
    omega_sun = 2.5 / (60.0)  # solar apparent motion (2.5 arcsec/min to arcsec/sec)
    psf = get_psf_size(msname)
    t_max = 0.5 * (psf / omega_sun)  # seconds
    return t_max

# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
