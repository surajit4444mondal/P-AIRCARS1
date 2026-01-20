import types
import psutil
import numpy as np
import traceback
import warnings
import glob
import os
import dask
from casatools import msmetadata, ms as casamstool, table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from numpy.linalg import inv
from .basic_utils import *
from .ms_metadata import *
from .imaging import *

#####################################
# Calibration related
#####################################


def fluxcal_caltable(caltable, attn=10):
    """
    Function to scale scale MWA bandpass table for attenuation (Digital gain corrections should already been applied)

    Parameters
    ----------
    caltable : str
        Name of the caltable
    attn : float, optional
        Attenuation in dB

    Returns
    -------
    str
        Flux calibrated caltable
    """
    datadir = get_datadir()
    tb = table()
    tb.open(f"{caltable}/SPECTRAL_WINDOW")
    freqlist = tb.getcol("CHAN_FREQ") / 10**6  # In MHz
    tb.close()
    fluxscale_poly = np.poly1d(
        np.load(f"{datadir}/Ref_mean_bandpass_final.npy", allow_pickle=True)[0]
    )
    gain_scale = fluxscale_poly(freqlist)
    att_scaling = 10 ** (-(attn - 1) / 10.0)
    gain_scale_att = gain_scale * np.sqrt(att_scaling)
    tb.open(caltable, nomodify=False)
    gain = tb.getcol("CPARAM")
    for i in range(gain.shape[1]):
        gain[:, i, :] *= gain_scale_att[i]
    tb.putcol("CPARAM", gain)
    tb.flush()
    tb.close()
    return caltable


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
    Function to calculate spectral width to produce bandwidth smearing

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


def get_caltable_metadata(caltable):
    """
    Function to get caltable metadata.

    Parameters
    ----------
    caltable : str
        Name of the caltable

    Returns
    -------
    dict
        A python dictionary with keywords MSNAME, JonesType, Channel 0 frequency (MHz), Central channel frequency (MHz), Channel width (kHz), Bandwidth (MHz), Start time, End time
    """
    tb = table()
    tb.open(caltable)
    caltype = tb.getkeywords()["VisCal"]
    msname = tb.getkeywords()["MSName"]
    tb.close()
    tb.open(f"{caltable}/SPECTRAL_WINDOW")
    ch0 = (tb.getcol("REF_FREQUENCY")[0]) / 10**6  # In MHz
    chanwidth = (tb.getcol("CHAN_WIDTH")[0] / 10**3)[0]  # In kHz
    freqlist = tb.getcol("CHAN_FREQ")
    chm = (freqlist[int(len(freqlist) / 2)] / 10**6)[0]  # In MHz
    bw = np.nanmean(tb.getcol("EFFECTIVE_BW")) / 10**6  # In MHz
    tb.close()
    tb.open(caltable)
    timerange = tb.getcol("TIME")
    start_time = mjdsec_to_timestamp(int(np.min(timerange)), str_format=0)
    end_time = mjdsec_to_timestamp(int(np.max(timerange)), str_format=0)
    tb.close()
    result = {
        "MSNAME": msname,
        "JonesType": caltype,
        "Channel 0 frequency (MHz)": ch0,
        "Central channel frequency (MHz)": chm,
        "Channel width (kHz)": chanwidth,
        "Bandwidth (MHz)": bw,
        "Start time": start_time,
        "End time": end_time,
    }
    return result


def get_nearest_bandpass_table(caltable_list, freq):
    """
    Function to get nearest bandpass table of a given frequency

    Parameters
    ----------
    caltable_list : list
        List of bandpass table
    freq : float
        Frequency in MHz

    Returns
    -------
    str
        Name of the nearest bandpass table
    """
    if len(caltable_list) == 0:
        print("No caltable is provided.")
        return
    if freq == None:
        print("No frequency information is given.")
        return
    caltable_list = np.array(caltable_list)
    freq_list = []
    for caltable in caltable_list:
        result = get_caltable_metadata(caltable)
        freq_list.append(float(result["Central channel frequency (MHz)"]))
    freq_list = np.array(freq_list)
    pos = np.argmin(np.abs(freq - freq_list))
    nearest_caltable = caltable_list[pos]
    return nearest_caltable


def get_nearest_gaincal_table(caltable_list, timestamp):
    """
    Function to get nearest gaincal table of a given time

    Parameters
    ----------
    caltable_list : list
        List of gaincal table
    timestamp : str
        Timestamp (format : 'YYYY/MM/DD/hh:mm:ss')

    Returns
    -------
    str
        Name of the nearest gaincal table
    """
    if len(caltable_list) == 0:
        print("No caltable is provided.\n")
        return None
    if timestamp == None:
        print("No time information is given.\n")
        return None
    try:
        caltable_list = np.array(caltable_list)
        time_list = []
        for caltable in caltable_list:
            result = get_caltable_metadata(caltable)
            starttime = result["Start time"]
            endtime = result["End time"]
            startime_mjd = timestamp_to_mjdsec(starttime, date_format=0)
            endtime_mjd = timestamp_to_mjdsec(endtime, date_format=0)
            time_list.append((startime_mjd + endtime_mjd) / 2.0)
        time_list = np.array(time_list)
        time_mjd = timestamp_to_mjdsec(timestamp, date_format=0)
        pos = np.argmin(np.abs(time_mjd - time_list))
        nearest_caltable = caltable_list[pos]
        return nearest_caltable
    except Exception as e:
        print("Nearest caltable could not be found.\n")
        return None


def get_gleam_uvrange(msname):
    """
    Get UV-range for GLEAM model

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    str
        UV-range in CASA format
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    wavelength = (3 * 10**8) / freq
    minuv_m = 112
    maxuv_m = 2500
    minuv_l = round(minuv_m / wavelength, 1)
    maxuv_l = round(maxuv_m / wavelength, 1)
    uvrange = f"{minuv_l}~{maxuv_l}lambda"
    return uvrange


def uvrange_casa_to_quartical(msname, uvrange=""):
    """
    Get quartical uv-range from CASA format uv-range

    Parameters
    ----------
    msname : str
        Measurement set
    uvrange : str
        UV-range in CASA format

    Returns
    -------
    float
        Minimum UV in meter
    float
        Maximum UV in meter
    """
    if uvrange == "":
        return [0.0, 0.0]
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    wavelength = (3 * 10**8) / freq
    uvrange = uvrange.rstrip("lambda")
    if "~" in uvrange:
        minuv_l = float(uvrange.split("~")[0])
        maxuv_l = float(uvrange.split("~")[-1])
    elif ">" in uvrange:
        minuv_l = float(uvrange.split(">")[-1])
        maxuv_l = 0.0
    elif "<" in uvrange:
        minuv_l = 0.0
        maxuv_l = float(uvrange.split("<")[0])
    else:
        minuv_l = 0.0
        maxuv_l = 0.0
    return float(minuv_l * wavelength), float(maxuv_l * wavelength)


def solint_in_float(solint):
    """
    Convert solution interval to seconds

    Parameters
    ----------
    solint : str
        Solution interval

    Returns
    -------
    float
        Solution interval in seconds
    """
    if solint.endswith("s"):
        solint = float(solint.rstrip("s"))
    elif solint.endswith("m"):
        solint = float(solint.rstrip("m") * 60.0)
    elif solint.endswith("h"):
        solint = float(solint.rstrip("h") * 3600.0)
    else:
        try:
            solint = float(solint)
        except:
            solint = None
    return solint


def quartical_matrix_normalize(caltable, overwrite=False):
    """
    Function to make matrix normalization (Normalization of full Jones solutions)
    Note : for mathematical expression, look at equation 21 of Kansabanik et al. 2022, ApJ, 932:110

    Parameters
    ----------
    caltable : str
        Name of the full Jones QuartiCal caltable
    overwrite : bool, optional
        Overwrite the input caltable (if not, a new caltable will be written)

    Returns
    -------
    str
        New caltable name
    """
    caltable = caltable.rstrip("/")
    caltable_dirs = os.listdir(caltable)
    soltype = caltable_dirs[0]
    gains = xds_from_zarr(f"{caltable}::{soltype}")
    gain_data = gains[0].gains.to_numpy()  # Shape: ntime, nchan, nant, ndir, npol
    gain_flag = gains[0].gain_flags.to_numpy()
    gain_flag = gains[0].gain_flags.values.astype(bool)
    gain_data[gain_flag, :] = np.nan
    gain_data = gain_data.reshape(*gain_data.shape[:-1], 2, 2)
    for t in range(gain_data.shape[0]):
        for f in range(gain_data.shape[1]):
            for d in range(gain_data.shape[3]):
                g = gain_data[t, f, :, d, ...]
                if np.abs(np.nansum(g)) != 0:
                    gH = g.conj().transpose(0, 2, 1)
                    gH_dot_g_sum_inv = inv(np.nansum(np.matmul(gH, g), axis=0))
                    X = inv(np.matmul(gH_dot_g_sum_inv, np.nansum(gH, axis=0)))
                    gain_data[t, f, :, d, ...] = np.matmul(g, inv(X))
    gain_data = gain_data.reshape(*gain_data.shape[:-2], 4)
    gain_flag = gain_flag.astype("int")
    gains[0].update(
        {
            "gain_flags": (
                ["gain_time", "gain_freq", "antenna", "direction"],
                gain_flag,
            )
        }
    )
    gains[0].update(
        {
            "gains": (
                ["gain_time", "gain_freq", "antenna", "direction", "correlation"],
                gain_data,
            )
        }
    )
    output_name = caltable if overwrite else f"{caltable}.poldist"
    if overwrite:
        os.system(f"rm -rf {caltable}*")
    write_xds_list = xds_to_zarr(gains, f"{output_name}::{soltype}")
    dask.compute(write_xds_list)
    return output_name


def get_quartical_table_metadata(caltable):
    """
    Function to get metadata of a quartical table.

    Parameters
    ----------
    caltable : str
        Name of the caltable

    Returns
    -------
    dict
        A python dictionary with keywords JonesType, Channel 0 frequency (MHz), Central channel frequency (MHz), Channel width (kHz), Bandwidth (MHz), Start time, End time
    """
    caltable = caltable.rstrip("/")
    caltable_dirs = os.listdir(caltable)
    soltype = caltable_dirs[0]
    gains = xds_from_zarr(f"{caltable}::{soltype}")
    jonestype = gains[0].TYPE
    freqs = gains[0].gain_freq.to_numpy()
    ch0 = freqs[0] / 10**6
    chm = np.nanmean(freqs) / 10**6
    try:
        chanwidth = abs(np.diff(freqs)[0]) / 10**3
    except:
        chanwidth = 160.0  # Assumed default value
    try:
        bw = (max(freqs) - min(freqs)) / 10**6
    except:
        bw = 1.28  # Assume a single coarse channel
    times = gains[0].gain_time.to_numpy()
    start_time = mjdsec_to_timestamp(min(times))
    end_time = mjdsec_to_timestamp(max(times))
    result = {
        "JonesType": jonestype,
        "Channel 0 frequency (MHz)": ch0,
        "Central channel frequency (MHz)": chm,
        "Channel width (kHz)": chanwidth,
        "Bandwidth (MHz)": bw,
        "Start time": start_time,
        "End time": end_time,
    }
    return result


def get_nearest_quartical_table(caltable_list, timestamp, freq):
    """
    Function to get nearest quartical table of a given time and frequency

    Parameters
    ----------
    caltable_list : list
        List of gaincal table
    timestamp : str
        Timestamp (format : 'YYYY/MM/DD/hh:mm:ss')
    freq : float
        Frequency in MHz

    Returns
    -------
    str
        Name of the nearest gaincal table
    """
    if len(caltable_list) == 0:
        print("No quartical caltable is provided.\n")
        return None
    if timestamp == None:
        print("No time information is given.\n")
        return None
    if freq == None:
        print("No frequency information is given.")
        return
    try:
        caltable_list = np.array(caltable_list)
        time_list = []
        for caltable in caltable_list:
            result = get_quartical_table_metadata(caltable)
            starttime = result["Start time"]
            endtime = result["End time"]
            startime_mjd = timestamp_to_mjdsec(starttime, date_format=0)
            endtime_mjd = timestamp_to_mjdsec(endtime, date_format=0)
            time_list.append((startime_mjd + endtime_mjd) / 2.0)
        time_list = np.array(time_list)
        time_mjd = timestamp_to_mjdsec(timestamp, date_format=0)
        pos = np.argmin(np.abs(time_mjd - time_list))
        nearest_caltables = caltable_list[pos]
        freq_list = []
        for caltable in nearest_caltables:
            result = get_quartical_table_metadata(caltable)
            mid_freq = result["Central channel frequency (MHz)"]
            freq_list.append(mid_freq)
        freq_list = np.array(freq_list)
        pos = np.argmin(np.abs(freq - freq_list))
        nearest_caltable = nearest_caltables[pos]
        return nearest_caltable
    except Exception as e:
        print("Nearest caltable could not be found.\n")
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
