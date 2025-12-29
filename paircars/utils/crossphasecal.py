import os
import copy
import time
import psutil
import warnings
import argparse
import numexpr as ne
import numpy as np
import subprocess
import traceback
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from casacore.tables import table
from paircars.pipeline.import_model import suppress_output

warnings.filterwarnings("ignore")


def get_chans_flags(msname):
    """
    Get channels flagged or not

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    numpy.array
        A boolean array indicating whether the channel is completely flagged or not
    """
    with suppress_output():
        tb = table(msname)
        flag = tb.getcol("FLAG")
        tb.close()
    chan_flags = np.all(np.all(flag, axis=-1), axis=0)
    return chan_flags


def create_crossphase_table(msname, caltable, freqs, crossphase, flags):
    """
    Create cross phase CASA caltable

    Parameters
    ----------
    msname : str
        Measurement set
    caltable : str
        Caltable name
    freqs : numpy.array
        Frequency list
    crossphase : numpy.array
        Crossphase array
    flags : numpy.array
        Flags

    Returns
    -------
    str
        Caltable name
    """
    nchan = len(freqs)
    cmd = [
        "create-caltable",
        "--msname",
        msname,
        "--caltable",
        caltable,
        "--nchan",
        str(nchan),
    ]

    subprocess.run(cmd, check=True)
    freqres = freqs[1] - freqs[0]
    if os.path.exists(caltable) is not True:
        print("Caltable is not made.")
        return
    with suppress_output():
        tb = table(msname)
        mean_time = np.nanmean(tb.getcol("TIME"))
        tb.close()
        del tb
        tb = table(caltable + "/SPECTRAL_WINDOW", readonly=False)
        freqs = np.array(freqs)[np.newaxis, :]
        freqres_array = np.repeat(np.array([[freqres]]), nchan, axis=1)
        tb.putcol("CHAN_FREQ", freqs)
        tb.putcol("NUM_CHAN", nchan)
        tb.putcol("REF_FREQUENCY", np.nanmean(freqs))
        tb.putcol("CHAN_WIDTH", freqres_array)
        tb.putcol("EFFECTIVE_BW", freqres_array)
        tb.putcol("RESOLUTION", freqres_array)
        tb.close()
        tb = table(caltable, readonly=False)
        ant = tb.getcol("ANTENNA1")
        gain = tb.getcol("CPARAM")
        cross_phase_gain_X = np.repeat(
            np.exp(1j * np.deg2rad(crossphase))[np.newaxis, :], len(ant), axis=0
        )
        gain[..., 0] = cross_phase_gain_X
        gain[..., 1] = cross_phase_gain_X * 0 + 1
        tb.putcol("CPARAM", gain)
        times = np.array([mean_time] * len(ant))
        flags = flags[np.newaxis, :, np.newaxis]
        flags = np.repeat(np.repeat(flags, len(ant), axis=0), 2, axis=2)
        tb.putcol("FLAG", flags)
        tb.putcol("TIME", times)
        tb.close()
    return caltable


def average_with_padding(array, chanwidth, axis=0, pad_value=np.nan):
    """
    Averages an array along a specified axis with a given chunk width (chanwidth),
    padding the array if its size along that axis is not divisible by chanwidth.

    Parameters
    ----------
    array : ndarray
        Input array to average.
    chanwidth : int
        Width of chunks to average.
    axis : int
        Axis along which to perform the averaging.
    pad_value : float
        Value to pad with if padding is needed (default: np.nan).

    Returns
    --------
    ndarray
        Array averaged along the specified axis.
    """
    # Compute the shape along the specified axis
    original_size = array.shape[axis]
    pad_size = -original_size % chanwidth
    # If padding is needed, apply it directly along the target axis
    if pad_size > 0:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, pad_size)
        array = np.pad(array, pad_width, constant_values=pad_value)
    # Compute the new shape and reshape the array for chunking
    new_shape = list(array.shape)
    new_shape[axis] = array.shape[axis] // chanwidth
    new_shape.insert(axis + 1, chanwidth)
    reshaped_array = array.reshape(new_shape)
    # Use nanmean along the chunk axis for averaging
    averaged_array = np.nanmean(reshaped_array, axis=axis + 1)
    return averaged_array


def interpolate_nans(data):
    """Linearly interpolate NaNs in 1D array."""
    nans = np.isnan(data)
    if np.all(nans):
        raise ValueError("All values are NaN.")
    x = np.arange(len(data))
    interp_func = interp1d(
        x[~nans],
        data[~nans],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp_func(x)


def filter_outliers(data, threshold=5, max_iter=3):
    """
    Filter outliers and perform cubic spline fitting

    Parameters
    ----------
    y : numpy.array
        Y values
    threshold : float
        Threshold of filtering
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    numpy.array
        Clean Y-values
    """
    for c in range(max_iter):
        data = np.asarray(data, dtype=np.float64)
        original_nan_mask = np.isnan(data)
        # Interpolate NaNs for smoothing
        interpolated_data = interpolate_nans(data)
        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(
            interpolated_data, sigma=threshold, truncate=3 * threshold
        )
        # Compute residuals and std only on valid original data
        residuals = data - smoothed
        valid_mask = ~original_nan_mask
        std_dev = np.std(residuals[valid_mask])
        # Detect outliers
        outlier_mask = np.abs(residuals) > threshold * std_dev
        combined_mask = valid_mask & ~outlier_mask
        # Replace outliers with NaN
        filtered_data = np.where(combined_mask, data, np.nan)
        data = copy.deepcopy(filtered_data)
    return filtered_data


def fitted_crossphase(freqs, crossphase):
    """
    Fit cos/sin components of crossphase vs frequency with increasing-degree polynomials,
    choose the best by residual std, and reconstruct phase.

    Parameters
    ----------
    freqs : array-like
        Frequency values (same length as crossphase).
    crossphase : array-like
        Cross-phase in degrees (can include NaNs).

    Returns
    -------
    np.ndarray
        Fitted cross-phase in degrees (NaNs outside valid span).
    """
    gain_r = np.cos(np.radians(crossphase))
    gain_i = np.sin(np.radians(crossphase))
    gains = [gain_r, gain_i]

    for i in range(len(gains)):
        gain = filter_outliers(gains[i])
        valid = ~np.isnan(gain)

        best_fit, best_std = None, np.inf
        for deg in range(3, 9):
            coeffs = np.polyfit(np.asarray(freqs)[valid], gain[valid], deg)
            interp_func = np.poly1d(coeffs)
            interp_gain = interp_func(freqs)

            residuals = gains[i] - interp_gain
            new_std = np.nanstd(residuals[~np.isnan(gains[i])])

            if new_std >= best_std:
                break

            best_std = new_std
            best_fit = interp_gain

        # Limit interpolation to valid frequency range
        nanpos = np.where(~np.isnan(gains[i]))[0]
        minpos, maxpos = np.nanmin(nanpos), np.nanmax(nanpos)
        best_fit[:minpos] = np.nan
        best_fit[maxpos:] = np.nan
        gains[i] = best_fit

    crossphase = np.angle(gains[0] + 1j * gains[1], deg=True)
    return crossphase


def crossphasecal(
    msname,
    caltable,
    uvrange="",
    gaintable="",
    chanwidth=1,
):
    """
    Function to calculate MWA cross hand phase

    Parameters
    ----------
    msname : str
            Name of the measurement set
    caltable : str
        Name of the caltable
    uvrange : str, optional
        UV-range for calibration
    gaintable : str, optional
            Previous gaintable
    chanwidth : int, optional
        Channels to average

    Returns
    -------
    str
            Name of the caltable
    """
    starttime = time.time()
    ncpu = int(psutil.cpu_count() * 0.8)
    if ncpu < 1:
        ncpu = 1
    ne.set_num_threads(ncpu)
    starttime = time.time()
    if caltable == "":
        caltable = msname.split(".ms")[0] + ".kcross"
    #######################
    with suppress_output():
        tb = table(msname + "/SPECTRAL_WINDOW")
        freqs = tb.getcol("CHAN_FREQ").flatten()
        cent_freq = tb.getcol("REF_FREQUENCY")[0]
        wavelength = (3 * 10**8) / cent_freq
        tb.close()
        del tb
    with suppress_output():
        tb = table(msname)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        data = tb.getcol("DATA")
        model_data = tb.getcol("MODEL_DATA")
        flag = tb.getcol("FLAG")
        uvw = tb.getcol("UVW")
        weight = tb.getcol("WEIGHT")
        # Col shape, baselines, chans, corrs
        weight = np.repeat(weight[:, np.newaxis, 0], model_data.shape[1], axis=1)
        tb.close()
    if gaintable == "":
        gaintable_supplied = False
    else:
        gaintable_supplied = True
        with suppress_output():
            tb = table(gaintable)
            if type(gaintable) == list:
                gaintable = gaintable[0]
            gain = tb.getcol("CPARAM")
            tb.close()
            del tb
    if uvrange != "":
        uvdist = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)
        if "~" in uvrange:
            minuv_m = float(uvrange.split("lambda")[0].split("~")[0]) * wavelength
            maxuv_m = float(uvrange.split("lambda")[0].split("~")[-1]) * wavelength
        elif ">" in uvrange:
            minuv_m = float(uvrange.split("lambda")[0].split(">")[-1]) * wavelength
            maxuv_m = np.nanmax(uvdist)
        else:
            minuv_m = 0.1
            maxuv_m = float(uvrange.split("lambda")[0].split("<")[-1]) * wavelength
        uv_filter = (uvdist >= minuv_m) & (uvdist <= maxuv_m)
        # Filter data based on uv_filter
        data = data[uv_filter, :, :]
        model_data = model_data[uv_filter, :, :]
        flag = flag[uv_filter, :, :]
        weight = weight[uv_filter, :]
        ant1 = ant1[uv_filter]
        ant2 = ant2[uv_filter]
    #######################
    data[flag] = np.nan
    model_data[flag] = np.nan
    xy_data = data[..., 1]
    yx_data = data[..., 2]
    xy_model = model_data[..., 1]
    yx_model = model_data[..., 2]
    if gaintable_supplied:
        gainX1 = gain[ant1, :, 0]
        gainY1 = gain[ant1, :, -1]
        gainX2 = gain[ant2, :, 0]
        gainY2 = gain[ant2, :, -1]
        del gain
    del data, model_data, uvw, flag
    if chanwidth > 1:
        xy_data = average_with_padding(xy_data, chanwidth, axis=1, pad_value=np.nan)
        yx_data = average_with_padding(yx_data, chanwidth, axis=1, pad_value=np.nan)
        xy_model = average_with_padding(xy_model, chanwidth, axis=1, pad_value=np.nan)
        yx_model = average_with_padding(yx_model, chanwidth, axis=1, pad_value=np.nan)
        if gaintable_supplied:
            gainX1 = average_with_padding(gainX1, chanwidth, axis=1, pad_value=np.nan)
            gainX2 = average_with_padding(gainX2, chanwidth, axis=1, pad_value=np.nan)
            gainY1 = average_with_padding(gainY1, chanwidth, axis=1, pad_value=np.nan)
            gainY2 = average_with_padding(gainY2, chanwidth, axis=1, pad_value=np.nan)
        weight = average_with_padding(weight, chanwidth, axis=1, pad_value=np.nan)
    if gaintable_supplied:
        argument = ne.evaluate(
            "weight * xy_data * conj(xy_model * gainX1) * gainY2 + weight * yx_model * gainY1 * conj(gainX2 * yx_data)"
        )
    else:
        argument = ne.evaluate(
            "weight * xy_data * conj(xy_model) + weight * yx_model * conj(yx_data)"
        )
    crossphase = np.angle(np.nansum(argument, axis=0), deg=True)
    freqs = average_with_padding(freqs, chanwidth, axis=0, pad_value=np.nan)
    if chanwidth > 1:
        chan_flags = np.array([False] * len(crossphase))
    else:
        chan_flags = get_chans_flags(msname)
    crossphase[chan_flags] = np.nan
    crossphase = fitted_crossphase(freqs, crossphase)
    create_crossphase_table(msname, caltable, freqs, crossphase, chan_flags)
    return caltable
