import os
import numpy as np
import warnings
import types
from scipy.interpolate import interp1d
from casatools import msmetadata, table, ms as casamstool
from astropy.wcs import FITSFixedWarning
from .basic_utils import *
from .ms_metadata import *
from .mwa_utils import *
from .mwapb_utils import *
from .imaging import *
from .sunpos_utils import *

warnings.simplefilter("ignore", category=FITSFixedWarning)
kb = 1.38e-23  # Boltzmann constant
light_speed = 3.0e8  # Speed of light in vacuum


def fill_nan(arr):
    """
    Function to interpolate to nan values

    Parameters
    ----------
    arr : np.array
        1-D numpy array

    Returns
    -------
    np.array
        1-D nan interpolated numpy array
    """
    try:
        med_fill_value = np.nanmedian(arr)
        inds = np.arange(arr.shape[0])
        good = np.where(np.isfinite(arr))
        f = interp1d(
            inds[good],
            arr[good],
            bounds_error=False,
            kind="linear",
            copy=True,
            fill_value=med_fill_value,
        )
        out_arr = np.where(np.isfinite(arr), arr, f(inds))
    except:
        out_arr = arr
    return out_arr


def calc_T_rec(freq):
    """
    Function to calculate receiver temperature based on Noise Temperature of Phased Array Radio Telescope:
    The Murchison Widefield Array and the Engineering Development Array, Ung et al, 2020, IEEE

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Receiver temperature in K
    """
    x = [
        50.1336,
        52.4987,
        54.5253,
        58.5773,
        60.6017,
        63.2964,
        65.3203,
        68.3472,
        72.3884,
        73.7374,
        77.7740,
        82.1462,
        85.8443,
        89.2056,
        92.5666,
        96.5994,
        100.969,
        108.701,
        114.417,
        120.802,
        127.187,
        133.237,
        140.296,
        148.868,
        153.068,
        158.949,
        165.839,
        170.711,
        175.079,
        178.271,
        184.821,
        189.524,
        192.883,
        197.082,
        204.806,
        210.515,
        217.567,
        221.091,
        224.953,
        228.143,
        232.341,
        235.027,
        238.218,
        241.913,
        243.928,
        247.957,
        249.805,
        251.653,
        254.509,
        255.181,
        258.371,
        264.081,
        270.128,
        274.495,
        282.221,
        286.421,
        291.964,
        296.332,
        301.036,
        304.061,
        308.597,
        311.790,
        314.646,
        320.526,
        326.740,
    ]
    y = [
        2363.92,
        1600.02,
        1168.37,
        643.601,
        501.566,
        408.215,
        321.595,
        291.723,
        220.082,
        189.092,
        162.481,
        141.139,
        130.839,
        123.949,
        118.702,
        113.679,
        105.386,
        90.5671,
        80.3998,
        74.5398,
        70.6221,
        65.4739,
        58.1265,
        49.6844,
        48.3628,
        45.5723,
        41.7958,
        40.6851,
        40.0350,
        38.9687,
        40.7059,
        41.3804,
        40.9391,
        41.8431,
        45.1556,
        48.7268,
        52.2987,
        57.6677,
        60.8893,
        62.2315,
        65.7089,
        69.0016,
        69.0096,
        69.0188,
        70.9207,
        75.6996,
        76.5301,
        75.7098,
        74.4959,
        73.6942,
        76.9700,
        80.3988,
        80.4164,
        80.4291,
        81.7709,
        79.5959,
        80.0447,
        79.6244,
        77.9293,
        75.0352,
        71.8618,
        69.9478,
        68.8263,
        67.3640,
        68.1138,
    ]
    l = len(x)
    if freq < 50:
        return y[0]
    if freq > 326:
        return y[l - 1]
    tlna_cubic = interp1d(x, y, kind="cubic")
    trcv = tlna_cubic(freq)
    return trcv


def calc_T_pickup(freq):
    """
    Function to calculate ground pickup temperature (Oberoi et al. 2017)

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Pickup temperature in K
    """
    x = [
        75.0,
        100.404274,
        110.24117,
        119.56543,
        129.94064,
        140.06808,
        149.92746,
        159.51872,
        169.93245,
        179.54617,
        189.69287,
        199.59288,
        210.00659,
        219.64066,
        229.5599,
        240.28342,
        249.58629,
        259.69342,
        269.51212,
        279.3533,
        289.7649,
        299.1116,
        300.2,
    ]
    y = [
        20.142,
        20.141666,
        18.002377,
        16.929209,
        15.002632,
        14.035396,
        13.015087,
        11.941707,
        11.933383,
        11.978985,
        11.970875,
        12.9753895,
        12.967067,
        14.02508,
        15.988722,
        18.111578,
        15.972715,
        13.993066,
        10.947934,
        9.021784,
        8.90689,
        8.952705,
        8.952705,
    ]
    l = len(x)
    if freq < 75.0:
        return y[0]
    if freq > 300.2:
        return y[l - 1]
    tpick_cubic = interp1d(x, y, kind="cubic", fill_value="extrapolate")
    tpick = tpick_cubic(freq)
    return tpick


def cal_sun_solid_angle(freq):
    """
    Function to calculate the diameter of the Sun at a given frequency (White 2016)

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Solid angle of the Sun
    """
    dia = calc_sun_dia(freq)
    solidangle_sun = np.pi * ((dia * np.pi / (180 * 60)) ** 2)
    return solidangle_sun


def cal_norm_crosscorr(msname, ant1, ant2):
    """
    Function to obtain normalised cross correlation amplitude

    Parameters
    ----------
    msname : str
        Measurement set
    ant1 : int
        Antenna 1
    ant2 : int
        Antenna 2

    Returns
    -------
    np.array
        Normalized cross-correlation of polarization XX
    np.array
        Normalized cross-correlation of real part of polarization XY
    np.array
        Normalized cross-correlation of imaginary part of polarization XY
    np.array
        Normalized cross-correlation of polarization YY
    """
    msmd = msmetadata()
    msmd.open(msname)
    npol = int(msmd.ncorrforpol()[0])
    msmd.close()
    mstool = casamstool()
    mstool.open(msname)
    mstool.select({"antenna1": ant1, "antenna2": ant2})
    dataij = mstool.getdata("DATA")["data"]
    flag = mstool.getdata("FLAG")["flag"]
    mstool.close()
    mstool.open(msname)
    mstool.select({"antenna1": ant1, "antenna2": ant1})
    dataii = mstool.getdata("DATA")["data"]
    mstool.close()
    mstool.open(msname)
    mstool.select({"antenna1": ant2, "antenna2": ant2})
    datajj = mstool.getdata("DATA")["data"]
    mstool.close()
    rN_XX = np.abs(dataij[0, ...]) / np.sqrt(
        np.abs(dataii[0, ...]) * np.abs(datajj[0, ...])
    )
    rN_YY = np.abs(dataij[-1, ...]) / np.sqrt(
        np.abs(dataii[-1, ...]) * np.abs(datajj[-1, ...])
    )
    if npol == 4:
        rN_XY = (dataij[1, ...]) / np.sqrt(
            np.abs(dataii[0, ...]) * np.abs(datajj[-1, ...])
        )
        return rN_XX, np.real(rN_XY), np.imag(rN_XY), rN_YY
    else:
        return rN_XX, rN_YY


def get_short_baselines(msname, max_uv=100.0, nmax=6):
    """
    Get list of shortest baselines

    Parameters
    ----------
    msname : str
        Measurement set
    max_uv : float
        Maximum UV in lambda
    nmax : int
        Number of baselines

    Returns
    -------
    list
        List of baselines
    """
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    ant1 = tb.getcol("ANTENNA1")
    ant2 = tb.getcol("ANTENNA2")
    tb.close()
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    wavelength = light_speed / freq
    sun_taper = max_uv * wavelength
    # UV distance
    uvdist = np.hypot(uvw[0], uvw[1])
    # Valid short baselines
    mask = (uvdist > 0) & (uvdist < sun_taper)
    uvdist = uvdist[mask]
    ant1 = ant1[mask]
    ant2 = ant2[mask]
    # Build unique baseline set
    seen = set()
    baselines = []
    for a1, a2, uv in zip(ant1, ant2, uvdist):
        key = (int(a1), int(a2))
        key_inv = (int(a2), int(a1))
        if key not in seen and key_inv not in seen:
            seen.add(key)
            seen.add(key_inv)
            baselines.append([key[0], key[1]])
        if len(baselines) >= nmax:
            break
    return baselines


def calc_dynamic_spectrum(msname, metafits, outdir, nthreads=1):
    """
    Function to calculate MWA dynamic spectrum of the Sun

    Parameters
    ----------
    msname : str
        Measurement set
    metafits : str
        Metafits file
    outdir : str
        Name of the output directory
    nthreads : int, optional
        Number of CPU threads to use

    Returns
    -------
    str
        Output dynamic spectrum file name
    str
        Output normalised cross-correlation file name
    """
    ##################################
    # Determine baseline list
    ##################################
    msmd = msmetadata()
    msmd.open(msname)
    freqs = msmd.chanfreqs(0,unit="MHz")
    npol = int(msmd.ncorrforpol()[0])
    msmd.close()
    highest_freq = np.nanmax(freqs)
    smallest_wavelength = 299792458.0 / (highest_freq*(10**6))
    max_uv = 100 / smallest_wavelength
    baselines = get_short_baselines(msname, max_uv=max_uv, nmax=3)
    sun_radec_string, sun_ra, sun_dec, radeg, decdeg = radec_sun(msname)

    ######################################################
    # Receiver, pickup temperature and solar solid angles
    ######################################################
    T_rec = []
    T_pick = []
    solar_solid_angle = []
    for freq in freqs:
        T_rec.append(calc_T_rec(freq))
        T_pick.append(calc_T_pickup(freq))
        solar_solid_angle.append(cal_sun_solid_angle(freq))
    T_pick = np.array(T_pick)
    T_rec = np.array(T_rec)
    solar_solid_angle = np.array(solar_solid_angle)

    #########################################
    # Per baseline calculations
    # Calculate normalised cross-correlations
    # Sky and fringe temperature
    #########################################
    rn_xx_list = []
    rn_yy_list = []
    if npol == 4:
        rn_xy_list = []
        rn_yx_list = []
    T_sun_xx_list = []
    T_sun_yy_list = []
    S_sun_xx_list = []
    S_sun_yy_list = []
    rn_dic = {}
    for baseline in baselines:
        ###################################
        # Normalised cross-correlation
        ###################################
        print(f"Extracting normalised cross-correlation from ms: {msname} for baseline: {baseline}.")
        result_rn = cal_norm_crosscorr(msname, baseline[0], baseline[1])
        rn_dic[tuple(baseline)] = result_rn
        rn_xx = result_rn[0]
        rn_yy = result_rn[-1]
        rn_xx_list.append(rn_xx)
        rn_yy_list.append(rn_yy)
        if npol == 4:
            rn_xy_list.append(result_rn[1])
            rn_yx_list.append(result_rn[2])

        ########################################s
        # Sun btightness temperature calculation
        ########################################
        T_ant_xx_spectrum = []
        T_ant_yy_spectrum = []
        T_fringe_xx_spectrum = []
        T_fringe_yy_spectrum = []
        sun_beam_xx_spectrum = []
        sun_beam_yy_spectrum = []
        beam_omega_xx_spectrum = []
        beam_omega_yy_spectrum = []
        print(f"Determining system and sky temperatures for ms: {msname} for baseline: {baseline}.")
        for i in range(len(freqs)):
            #################################
            # Each frequency calculations
            #################################
            freq = freqs[i]
            (
                beamsky_sum_xx,
                beam_sum_xx,
                T_ant_xx,
                beam_dOMEGA_sum_xx,
                beamsky_sum_yy,
                beam_sum_yy,
                T_ant_yy,
                beam_dOMEGA_sum_yy,
                T_fringe_xx,
                T_fringe_yy,
            ) = make_primarybeammap(
                msname,
                metafits,
                baseline,
                freq,
                nthreads=nthreads,
                iau_order=False,
                calc_fringe_temp=True,
            )
            ############################
            # Primary beam at the Sun
            ############################
            sun_beam = get_pb_radec(radeg, decdeg, freq, metafits)
            sun_beam_xx = sun_beam[3]
            sun_beam_yy = sun_beam[4]

            ########################################
            # Making spectrum of various quantities
            ########################################
            T_ant_xx_spectrum.append(T_ant_xx)
            T_ant_yy_spectrum.append(T_ant_yy)
            T_fringe_xx_spectrum.append(T_fringe_xx)
            T_fringe_yy_spectrum.append(T_fringe_yy)
            sun_beam_xx_spectrum.append(sun_beam_xx)
            sun_beam_yy_spectrum.append(sun_beam_yy)
            beam_omega_xx_spectrum.append(beam_dOMEGA_sum_xx)
            beam_omega_yy_spectrum.append(beam_dOMEGA_sum_yy)

        T_ant_xx_spectrum = np.array(T_ant_xx_spectrum)
        T_ant_yy_spectrum = np.array(T_ant_yy_spectrum)
        T_fringe_xx_spectrum = np.array(T_fringe_xx_spectrum)
        T_fringe_yy_spectrum = np.array(T_fringe_yy_spectrum)
        sun_beam_xx_spectrum = np.array(sun_beam_xx_spectrum)
        sun_beam_yy_spectrum = np.array(sun_beam_yy_spectrum)
        beam_omega_xx_spectrum = np.array(beam_omega_xx_spectrum)
        beam_omega_yy_spectrum = np.array(beam_omega_yy_spectrum)

        ################################################
        # Temperature and flux of Sun for X polarisation
        #################################################
        T_sun_xx = (
            (rn_xx/ (1 - rn_xx)) * (T_ant_xx_spectrum[:,None] + T_rec[:,None] + T_pick[:,None])
        ) - (T_fringe_xx_spectrum[:,None] / (1 - rn_xx))
        T_sun_xx_avg = T_sun_xx / sun_beam_xx_spectrum[:,None]
        T_sun_xx = T_sun_xx_avg * beam_omega_xx_spectrum[:,None] / solar_solid_angle[:,None]
        S_sun_xx = (
            2
            * kb
            * T_sun_xx_avg
            * beam_omega_xx_spectrum[:,None]
            / (light_speed / (freqs[:,None] * (10**6))) ** 2
        ) / (10 ** (-22))

        ################################################
        # Temperature and flux of  Sun for Y polarisation
        ################################################
        T_sun_yy = ((rn_yy / (1 - rn_yy)) * (T_ant_yy_spectrum[:,None] + T_rec[:,None] + T_pick[:,None])) - (
            T_fringe_yy_spectrum[:,None] / (1 - rn_yy)
        )
        T_sun_yy_avg = T_sun_yy / sun_beam_yy_spectrum[:,None]
        T_sun_yy = T_sun_yy_avg * beam_omega_yy_spectrum[:,None] / solar_solid_angle[:,None]
        S_sun_yy = (
            2
            * kb
            * T_sun_yy_avg
            * beam_omega_yy_spectrum[:,None]
            / (light_speed / (freqs[:,None] * (10**6))) ** 2
        ) / (10 ** (-22))

        ###################################################
        # Making ds per baseline
        ###################################################
        T_sun_xx_list.append(T_sun_xx)
        T_sun_yy_list.append(T_sun_yy)
        S_sun_xx_list.append(S_sun_xx)
        S_sun_yy_list.append(S_sun_yy)

    ########################################
    # Baslines averaged spectrum
    ########################################
    T_sun_xx_list = np.array(T_sun_xx_list)
    T_sun_yy_list = np.array(T_sun_yy_list)
    S_sun_xx_list = np.array(S_sun_xx_list)
    S_sun_yy_list = np.array(S_sun_yy_list)

    T_sun_xx = np.nanmean(T_sun_xx_list, axis=0)
    T_sun_yy = np.nanmean(T_sun_yy_list, axis=0)
    S_sun_xx = np.nanmean(S_sun_xx_list, axis=0)
    S_sun_yy = np.nanmean(S_sun_yy_list, axis=0)

    T_sun = (T_sun_xx + T_sun_yy) / 2.0
    S_sun = (S_sun_xx + S_sun_yy) / 2.0

    ######################################
    # Extracting metadata
    ######################################
    bad_chans = get_bad_chans(msname)
    if bad_chans != "":
        bad_chans = bad_chans.replace("0:", "").split(";")
        for bad_chan in bad_chans:
            s = int(bad_chan.split("~")[0])
            e = int(bad_chan.split("~")[-1]) + 1
            data[s:e, :] = np.nan
    msmd = msmetadata()
    msmd.open(msname)
    freqs = msmd.chanfreqs(0, unit="MHz")
    mid_freq = msmd.meanfreq(0, unit="MHz")
    times = msmd.timesforspws(0)
    timestamps = [mjdsec_to_timestamp(mjdsec, str_format=0) for mjdsec in times]
    t_string = "".join(timestamps[0].split("T")[0].split("-")) + "".join(
        timestamps[0].split("T")[-1].split(".")[0].split(":")
    )
    msmd.close()
    save_file = f"freq_{mid_freq}MHz_time_{t_string}"
    np.save(
        f"{outdir}/{save_file}_ds.npy",
        np.array([freqs, times[1:], timestamps[1:], T_sun[:,1:], S_sun[:,1:]], dtype="object"),
    )

    np.save(f"{outdir}/{save_file}_rn.npy", np.array(rn_dic, dtype="object"))
    return f"{outdir}/{save_file}_ds.npy", f"{outdir}/{save_file}_rn.npy"


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
