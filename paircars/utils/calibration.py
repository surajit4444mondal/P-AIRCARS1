import numpy as np
import traceback
import os
import zarr
import dask
import warnings
import xarray as xr
from casatools import msmetadata, table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from numpy.linalg import inv
from .basic_utils import (
    get_datadir,
    ceil_to_multiple,
    timestamp_to_mjdsec,
    mjdsec_to_timestamp,
)
from .imaging import calc_sun_dia, calc_maxuv, calc_field_of_view

#####################################
# Calibration related
#####################################


def fill_nan_gains(x, data, x_new=None):
    """
    Interpolate gain values across frequency.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of input frequencies.
    data : numpy.ndarray
        1D array of gain values (real or imaginary part).
    x_new : numpy.ndarray, optional
        Frequencies at which to evaluate the interpolation.
        If None, interpolation is evaluated at the input frequencies.

    Returns
    -------
    numpy.ndarray
        Interpolated gain values.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    x = np.asarray(x)
    data = np.asarray(data, dtype=float)
    if x_new is None:
        x_new = x
    else:
        x_new = np.asarray(x_new)
    # Remove NaNs
    valid = np.isfinite(data)
    # Need at least two points for linear interpolation
    if np.sum(valid) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    # Sort by frequency
    order = np.argsort(x[valid])
    x_valid = x[valid][order]
    data_valid = data[valid][order]
    interp_func = interp1d(
        x_valid,
        data_valid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return interp_func(x_new)


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


def interpolate_bpass(caltables, overwrite=False):
    """
    Interpolate bandpass/crossphase tables for missing frequency solutions

    Parameters
    ----------
    caltables : list
        List of bandpass/crossphase tables
    overwrite : bool, optional
        Overwrite the input tables or not

    Returns
    -------
    list
        Output bandpass/crossphase tables
    """
    if len(caltables) < 2:
        return caltables
    tb = table()
    all_freqs = []
    all_gains = []
    for cal in caltables:
        tb.open(cal + "/SPECTRAL_WINDOW")
        freq = tb.getcol("CHAN_FREQ")
        tb.close()
        tb.open(cal)
        gain = tb.getcol("CPARAM")
        flag = tb.getcol("FLAG")
        gain[flag] = np.nan + 1j * np.nan
        tb.close()
        all_freqs.append(freq)
        all_gains.append(gain)
    all_freqs = np.concatenate(all_freqs, axis=0)
    all_gains = np.concatenate(all_gains, axis=1)
    all_freqs = all_freqs.flatten()
    pos = np.argsort(all_freqs)
    all_freqs_sorted = all_freqs[pos]
    all_gains_sorted = all_gains[:, pos, :]
    interpolated_gains = np.ones(all_gains_sorted.shape, dtype="complex")
    npol = all_gains_sorted.shape[0]
    nant = all_gains_sorted.shape[-1]
    for p in range(npol):
        for a in range(nant):
            interp_re = fill_nan_gains(
                all_freqs_sorted, np.real(all_gains_sorted[p, :, a])
            )
            interp_im = fill_nan_gains(
                all_freqs_sorted, np.imag(all_gains_sorted[p, :, a])
            )
            interp_gain = interp_re + 1j * interp_im
            nans = np.isnan(interp_gain)
            interp_gain[nans] = 1.0 + 1j * 0.0
            interpolated_gains[p, :, a] = interp_gain
            del interp_gain
    outlist = []
    for cal in caltables:
        cal = cal.rstrip("/")
        if overwrite is False:
            outcal = f"{cal}.interp"
            if os.path.exists(outcal):
                os.system(f"rm -rf {outcal}")
            os.system(f"cp -r {cal} {outcal}")
        else:
            outcal = cal
        tb.open(f"{outcal}/SPECTRAL_WINDOW")
        freqs = tb.getcol("CHAN_FREQ")
        tb.close()
        pos = np.searchsorted(all_freqs_sorted, freqs.flatten())
        interp_gain_out = interpolated_gains[:, pos, :]
        tb.open(outcal, nomodify=False)
        flags = tb.getcol("FLAG")
        gains = tb.getcol("CPARAM")
        gains[flags] = interp_gain_out[flags]
        tb.putcol("CPARAM", gains)
        flags *= False
        pos = np.where(np.isnan(np.abs(gains)))
        flags[pos] = True
        tb.putcol("FLAG", flags)
        tb.flush()
        tb.close()
        outlist.append(outcal)
    return outlist


def interpolate_quartical(caltables, overwrite=False):
    """
    Function to interpolate quartical caltable

    Parameters
    ----------
    caltables : list
        Name of the full Jones QuartiCal caltable caltables
    overwrite : bool, optional
        Overwrite the input caltable (if not, a new caltable will be written)

    Returns
    -------
    str
        New caltable name
    """
    all_freqs = []
    all_gains = []
    for caltable in caltables:
        caltable = caltable.rstrip("/")
        soltypes = get_quartical_soltype(caltable)
        if len(soltypes) == 0:
            print("No solution is present. Not performing interpolation.")
            pass
        else:
            soltype = soltypes[0]
            gains = xds_from_zarr(f"{caltable}::{soltype}")
            freqs = gains[0].gain_freq.to_numpy()
            gain_data = gains[
                0
            ].gains.to_numpy()  # Shape: ntime, nchan, nant, ndir, npol
            gain_flag = gains[0].gain_flags.to_numpy()
            gain_flag = gains[0].gain_flags.values.astype(bool)
            gain_data[gain_flag, :] = np.nan
            all_freqs.append(freqs)
            all_gains.append(gain_data)

    all_freqs = np.concatenate(all_freqs, axis=0)
    all_gains = np.concatenate(all_gains, axis=1)
    all_freqs = all_freqs.flatten()
    pos = np.argsort(all_freqs)
    all_freqs_sorted = all_freqs[pos]
    all_gains_sorted = all_gains[:, pos, ...]
    interpolated_gains = np.ones(all_gains_sorted.shape, dtype="complex")
    interpolated_gains[..., 1] *= 0.0
    interpolated_gains[..., 2] *= 0.0
    ntime = all_gains_sorted.shape[0]
    npol = all_gains_sorted.shape[-1]
    nant = all_gains_sorted.shape[2]
    for t in range(ntime):
        for p in range(npol):
            for a in range(nant):
                interp_re = fill_nan_gains(
                    all_freqs_sorted, np.real(all_gains_sorted[t, :, a, 0, p])
                )
                interp_im = fill_nan_gains(
                    all_freqs_sorted, np.imag(all_gains_sorted[t, :, a, 0, p])
                )
                interp_gain = interp_re + 1j * interp_im
                nans = np.isnan(interp_gain)
                if p == 0 or p == 3:
                    interp_gain[nans] = 1.0 + 1j * 0.0
                else:
                    interp_gain[nans] = 0.0 + 1j * 0.0
                interpolated_gains[t, :, a, 0, p] = interp_gain
                del interp_gain
    outlist = []
    for caltable in caltables:
        caltable = caltable.rstrip("/")
        soltypes = get_quartical_soltype(caltable)
        if len(soltypes) == 0:
            print("No solution is present. Not performing interpolation.")
            pass
        else:
            soltype = soltypes[0]
            gains = xds_from_zarr(f"{caltable}::{soltype}")
            gain_data = gains[
                0
            ].gains.to_numpy()  # Shape: ntime, nchan, nant, ndir, npol
            gain_flag = gains[0].gain_flags.to_numpy()
            bool_gain_flag = gains[0].gain_flags.values.astype(bool)
            freqs = gains[0].gain_freq.to_numpy()
            pos = np.searchsorted(all_freqs_sorted, freqs.flatten())
            interp_gain_out = interpolated_gains[:, pos, ...]
            gain_data[bool_gain_flag, :] = interp_gain_out[bool_gain_flag, :]
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
                        [
                            "gain_time",
                            "gain_freq",
                            "antenna",
                            "direction",
                            "correlation",
                        ],
                        gain_data,
                    )
                }
            )
            if overwrite:
                output_name = caltable
            else:
                output_name = f"{caltable}.interp"
            if overwrite:
                os.system(f"rm -rf {caltable}*")
            write_xds_list = xds_to_zarr(gains, f"{output_name}::{soltype}")
            dask.compute(write_xds_list)
            outlist.append(output_name)
    return outlist


def make_interpolated_quartical_table(caltables, target_freqs, output_name):
    """
    Interpolate one or more QuartiCal calibration tables onto a new frequency grid.

    Parameters
    ----------
    caltables : list
        List of QuartiCal Jones tables.
    target_freqs : ndarray
        Desired frequency grid (Hz).
    output_name : str
        Name of output QuartiCal table.

    Returns
    -------
    str
        Output table name.
    """
    if os.path.exists(output_name):
        os.system(f"rm -rf {output_name}")
    all_freqs = []
    all_gains = []
    template_ds = None
    soltype = None
    target_freqs = target_freqs.astype("float64")
    # ---------------------------------------------------------
    # Read all tables
    # ---------------------------------------------------------
    if type(caltables) is str:
        caltables = [caltables]
    for caltable in caltables:
        caltable = caltable.rstrip("/")
        soltypes = get_quartical_soltype(caltable)
        if len(soltypes) == 0:
            continue
        soltype = soltypes[0]
        gains = xds_from_zarr(f"{caltable}::{soltype}")
        if template_ds is None:
            template_ds = gains[0]
        freqs = gains[0].gain_freq.values
        gain_data = gains[0].gains.values.copy()
        gain_flag = gains[0].gain_flags.values.astype(bool)
        gain_data[gain_flag] = np.nan
        all_freqs.append(freqs)
        all_gains.append(gain_data)
    all_freqs = np.concatenate(all_freqs)
    all_gains = np.concatenate(all_gains, axis=1)
    order = np.argsort(all_freqs)
    all_freqs = all_freqs[order]
    all_gains = all_gains[:, order, ...]
    # ---------------------------------------------------------
    # Allocate output array
    # ---------------------------------------------------------
    shape = (
        all_gains.shape[0],
        len(target_freqs),
        all_gains.shape[2],
        all_gains.shape[3],
        all_gains.shape[4],
    )
    out_gains = np.ones(shape, dtype=np.complex64)
    out_gains[..., 1] = 0
    out_gains[..., 2] = 0
    ntime = shape[0]
    nant = shape[2]
    ndir = shape[3]
    npol = shape[4]
    # ---------------------------------------------------------
    # Interpolate
    # ---------------------------------------------------------
    for t in range(ntime):
        for a in range(nant):
            for d in range(ndir):
                for p in range(npol):
                    re = fill_nan_gains(
                        all_freqs,
                        np.real(all_gains[t, :, a, d, p]),
                        x_new=target_freqs,
                    )
                    im = fill_nan_gains(
                        all_freqs,
                        np.imag(all_gains[t, :, a, d, p]),
                        x_new=target_freqs,
                    )
                    g = re + 1j * im
                    bad = np.isnan(g)
                    if p in (0, 3):
                        g[bad] = 1 + 0j
                    else:
                        g[bad] = 0 + 0j
                    out_gains[t, :, a, d, p] = g
    # ---------------------------------------------------------
    # Build output dataset
    # ---------------------------------------------------------
    new_ds = xr.Dataset(attrs=template_ds.attrs)
    # Copy coordinates
    for coord in template_ds.coords:
        if coord == "gain_freq":
            new_ds = new_ds.assign_coords(gain_freq=("gain_freq", target_freqs))
        else:
            new_ds = new_ds.assign_coords({coord: template_ds.coords[coord]})
    # Copy all variables
    for name, var in template_ds.data_vars.items():
        dims = list(var.dims)
        if "gain_freq" in dims:
            shape = [
                len(target_freqs) if d == "gain_freq" else template_ds.sizes[d]
                for d in dims
            ]
            data = np.zeros(shape, dtype=var.dtype)
            new_ds[name] = (dims, data)
        else:
            new_ds[name] = var.copy(deep=True)
    gain_flags = np.zeros(
        out_gains.shape[:-1],
        dtype=bool,
    )
    new_ds = new_ds.assign_coords(gain_freq=("gain_freq", target_freqs))
    new_ds.update(
        {
            "gain_freq": (
                ["gain_freq"],
                target_freqs,
            )
        }
    )
    new_ds.update(
        {
            "gain_flags": (
                ["gain_time", "gain_freq", "antenna", "direction"],
                gain_flags,
            )
        }
    )
    new_ds.update(
        {
            "gains": (
                [
                    "gain_time",
                    "gain_freq",
                    "antenna",
                    "direction",
                    "correlation",
                ],
                out_gains,
            )
        }
    )
    write = xds_to_zarr([new_ds], f"{output_name}::{soltype}")
    dask.compute(write)
    return output_name


def get_cal_flag_info(caltable):
    """
    Get flag information of the caltable

    Parameters
    ----------
    caltable : str
        Caltable

    Returns
    -------
    list
        Flagged channel list
    list
        Flagged antenna list
    list
        Flagged timestamp list
    float
        Total flag fraction
    float
        Channel flag fraction
    float
        Antenna flag fraction
    float
        Time flag fraction
    """
    tb = table()
    tb.open(caltable)
    flags = tb.getcol("FLAG")
    times = tb.getcol("TIME")
    unique_times = np.unique(times)
    ntime = len(unique_times)
    tb.close()
    shape = flags.shape
    flags = flags.reshape(shape[0], shape[1], ntime, shape[2] // ntime)
    shape = flags.shape
    npol = shape[0]
    nchan = shape[1]
    ntime = shape[2]
    nant = shape[3]
    chans_flags = np.nansum(flags, axis=(0, 2, 3))
    ant_flags = np.nansum(flags, axis=(0, 1, 2))
    time_flags = np.nansum(flags, axis=(0, 1, 3))
    flag_chans = np.where(chans_flags == npol * nant * ntime)[0]
    flag_ants = np.where(ant_flags == npol * nchan * ntime)[0]
    flag_times = np.where(time_flags == npol * nchan * nant)[0]
    flag_frac = np.nansum(flags) / np.size(flags)
    chan_flag_frac = len(flag_chans) / nchan
    ant_flag_frac = len(flag_ants) / nant
    time_flag_frac = len(flag_times) / ntime
    return (
        flag_chans.tolist(),
        flag_ants.tolist(),
        flag_times.tolist(),
        flag_frac,
        chan_flag_frac,
        ant_flag_frac,
        time_flag_frac,
    )


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
    return round(t_max, 2)


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
    if freq is None:
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
    if timestamp is None:
        print("No time information is given.\n")
        return None
    try:
        caltable_list = np.array(caltable_list)
        time_list = []
        for caltable in caltable_list:
            result = get_caltable_metadata(caltable)
            starttime = result["Start time"]
            endtime = result["End time"]
            startime_mjd = timestamp_to_mjdsec(starttime, date_format=1)
            endtime_mjd = timestamp_to_mjdsec(endtime, date_format=1)
            time_list.append((startime_mjd + endtime_mjd) / 2.0)
        time_list = np.array(time_list)
        time_mjd = timestamp_to_mjdsec(timestamp, date_format=0)
        pos = np.argmin(np.abs(time_mjd - time_list))
        nearest_caltable = caltable_list[pos]
        return nearest_caltable
    except Exception:
        traceback.print_exc()
        print("Nearest caltable could not be found.\n")
        return None


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
    return round(float(minuv_l * wavelength), 1), round(float(maxuv_l * wavelength), 1)


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
    elif solint.endswith("min"):
        solint = float(solint.rstrip("min")) * 60.0
    else:
        try:
            solint = float(solint)
        except Exception:
            solint = None
    return solint


def get_quartical_soltype(quartical_table):
    """
    Get quartical solution types

    Parameters
    ----------
    quartical_table : str
        Quartical table

    Returns
    -------
    list
        Solutions types
    """
    try:
        z = zarr.open_group(quartical_table, mode="r")
        soltypes = [k for k in z.group_keys() if not k.startswith(".")]
        if len(soltypes) == 0:
            return []
        return soltypes
    except Exception:
        print("Could not read QuartiCal caltable.")
        return []


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
    soltypes = get_quartical_soltype(caltable)
    if len(soltypes) == 0:
        print("No solution is present. Not performing any normalisation.")
        return caltable
    soltype = soltypes[0]
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
    soltypes = get_quartical_soltype(caltable)
    if len(soltypes) == 0:
        print("No solution is present.")
        return {}
    soltype = soltypes[0]
    gains = xds_from_zarr(f"{caltable}::{soltype}")
    jonestype = gains[0].TYPE
    freqs = gains[0].gain_freq.to_numpy()
    ch0 = freqs[0] / 10**6
    chm = np.nanmean(freqs) / 10**6
    try:
        chanwidth = abs(np.diff(freqs)[0]) / 10**3
    except Exception:
        chanwidth = 160.0  # Assumed default value
    try:
        bw = (max(freqs) - min(freqs)) / 10**6
    except Exception:
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


def scale_bandpass(bandpass_table, cal_attn, target_attn):
    """
    Scale a bandpass calibration table using attenuation data.

    Parameters
    ----------
    bandpass_table : str
        Input bandpass calibration table.
    cal_attn : float
        Calibrator attenuation
    target_attn : float
        Target attenuation

    Returns
    -------
    str
        Name of the output table.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    bandpass_table = bandpass_table.rstrip("/")
    output_table = f"{bandpass_table}.att{target_attn}"
    if os.path.exists(output_table):
        os.system(f"rm -rf {output_table}")
    os.system(f"cp -r {bandpass_table} {output_table}")
    tb = table()
    tb.open(output_table, nomodify=False)
    gain = tb.getcol("CPARAM")
    tb.getcol("FLAG")
    if cal_attn == target_attn:
        scaling = 1.0
    else:
        scaling = 10 ** (-(target_attn - cal_attn) / 20.0)
    gain *= scaling
    tb.putcol("CPARAM", gain)
    tb.flush()
    tb.close()
    return output_table
