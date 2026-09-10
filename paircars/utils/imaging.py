import numpy as np
from casatools import msmetadata, table
from .mwa_utils import get_bad_chans


##################################
# Imaging related
##################################
def is_fft_good(n):
    """
    Whether this number is good for FFTW or not
    """
    for p in [2, 3, 5, 7]:
        while n % p == 0:
            n //= p
    return n == 1


def get_fft_size(n):
    """
    Give the best number larger than the given number for best FFT performance
    """
    while True:
        if is_fft_good(n):
            if n < 128:
                if n <= 1:
                    n = 1
                else:
                    n = 1 << n.bit_length()
            return n + (n % 2)
        n += 1


def calc_sun_dia(freqMHz):
    """
    Function to calculate the diameter of the Sun at a given frequency (White 2016)

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Diameter of the Sun in arcmin
    """
    freqGHz = freqMHz / 10**3  # Convert in GHz
    dia = 32 + (2.2 * (freqGHz) ** (-0.6))
    return round(dia, 2)


def calc_maxuv(msname, chan_number=-1):
    """
    Calculate maximum UV

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
        Maximum UV in meter
    float
        Maximum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    maxuv = np.nanmax(uv)
    return round(float(maxuv), 2), round(float(maxuv / wavelength), 2)


def calc_minuv(msname, chan_number=-1):
    """
    Calculate minimum UV

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
        Minimum UV in meter
    float
        Minimum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    minuv = np.nanmin(uv)
    return round(float(minuv), 2), round(float(minuv / wavelength), 2)


def calc_uvtaper(msname):
    """
    Calculate UV-taper

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    float
        UV-taper in lambda at highest frequency
    """
    tb = table()
    tb.open(msname)
    u, v, w = tb.getcol("UVW")
    tb.close()
    msmd = msmetadata()
    msmd.open(msname)
    freqs = msmd.chanfreqs(0)
    max_freq = np.nanmax(freqs)
    wavelength = (3 * 10**8) / max_freq
    msmd.close()
    sun_dia = np.deg2rad(calc_sun_dia(max_freq / 10**6) / 60.0)
    bin_size_lambda = 1.22 / sun_dia
    bin_size = bin_size_lambda * wavelength
    r = np.sqrt(u**2 + v**2)
    n_bins = int(max(r) / bin_size)
    r_bins = np.linspace(r.min(), r.max(), n_bins)
    counts, edges = np.histogram(r, bins=r_bins)
    max_counts = np.nanmax(counts)
    pos = np.where(counts < 0.01 * max_counts)[0][0]
    uvtaper = edges[pos] / wavelength
    return round(uvtaper, 0)


def calc_field_of_view(msname, FWHM=True):
    """
    Calculate optimum field of view in arcsec.

    Parameters
    ----------
    msname : str
        Measurement set name
    FWHM : bool, optional
        Upto FWHM, otherwise upto first null

    Returns
    -------
    float
        Field of view in arcsec
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[0]
    msmd.close()
    tb = table()
    tb.open(msname + "/ANTENNA")
    dish_dia = np.nanmin(tb.getcol("DISH_DIAMETER"))
    tb.close()
    wavelength = 299792458.0 / freq
    if FWHM:
        FOV = 1.22 * wavelength / dish_dia
    else:
        FOV = 2.04 * wavelength / dish_dia
    fov_arcsec = np.rad2deg(FOV) * 3600  # In arcsecs
    return round(float(fov_arcsec), 2)


def get_optimal_image_interval(
    msname,
    temporal_tol_factor=0.1,
    spectral_tol_factor=0.1,
    chan_range="",
    timestamp_range="",
    flag_central_chan=False,
    max_nchan=-1,
    max_ntime=-1,
):
    """
    Get optimal image spectral temporal interval such that total flux max-median in each chunk is within tolerance limit

    Parameters
    ----------
    msname : str
        Name of the measurement set
    temporal_tol_factor : float, optional
        Tolerance factor for temporal variation (default : 0.1, 10%)
    spectral_tol_factor : float, optional
        Tolerance factor for spectral variation (default : 0.1, 10%)
    chan_range : str, optional
        Channel range
    timestamp_range : str, optional
        Timestamp range
    flag_central_chan : bool, optional
        Flag central channel
    max_nchan : int, optional
        Maxmium number of spectral chunk
    max_ntime : int, optional
        Maximum number of temporal chunk

    Returns
    -------
    int
        Number of time intervals to average
    int
        Number of channels to averages
    """
    from casatools import ms as casamstool

    def is_valid_chunk(chunk, tolerance):
        mean_flux = np.nanmedian(chunk)
        if mean_flux == 0:
            return False
        return (np.nanmax(chunk) - np.nanmin(chunk)) / mean_flux <= tolerance

    def find_max_valid_chunk_length(fluxes, tolerance):
        n = len(fluxes)
        for window in range(n, 1, -1):  # Try from largest to smallest
            valid = True
            for start in range(0, n, window):
                end = min(start + window, n)
                chunk = fluxes[start:end]
                if len(chunk) < window:  # Optionally require full window
                    valid = False
                    break
                if not is_valid_chunk(chunk, tolerance):
                    valid = False
                    break
            if valid:
                return window  # Return the largest valid window
        return 1  # Minimum chunk size is 1 if nothing else is valid

    temporal_tol_factor = max(
        1, temporal_tol_factor
    )  # Tolerance factor is maximum 100%
    spectral_tol_factor = max(1, spectral_tol_factor)

    tb = table()
    mstool = casamstool()
    msmd = msmetadata()
    msmd.open(msname)
    max_nchan = min(max_nchan, msmd.nchan(0))
    times = msmd.timesforspws(0)
    max_ntime = min(max_ntime, len(times))
    msmd.close()
    tb.open(msname)
    u, v, w = tb.getcol("UVW")
    tb.close()
    bad_chans = get_bad_chans(msname, flag_central_chan=flag_central_chan)
    if bad_chans != "":
        bad_chan_blocks = [i for i in bad_chans.split("0:")[1].split(";")]
        bad_chan_list = []
        for bad_chan in bad_chan_blocks:
            if "~" not in bad_chan:
                bad_chan_list.append(int(bad_chan))
            else:
                start_bad_chan = int(bad_chan.split("~")[0])
                end_bad_chan = int(bad_chan.split("~")[-1])
                for b in range(start_bad_chan, end_bad_chan + 1):
                    bad_chan_list.append(int(b))
    else:
        bad_chan_list = []
    uvdist = np.sort(np.unique(np.sqrt(u**2 + v**2)))
    mstool.open(msname)
    if uvdist[0] == 0.0:
        mstool.select({"uvdist": [0.0, 0.0]})
    else:
        mstool.select({"antenna1": 0, "antenna2": 1})
    data = mstool.getdata(["DATA"], ifraxis=True)["data"]
    mstool.close()
    if len(bad_chan_list) > 0:
        for bad_chan in bad_chan_list:
            data[:, bad_chan, ...] = np.nan
    data_shape = data.shape
    if len(data_shape) == 4:
        if chan_range != "":
            start_chan = int(chan_range.split(",")[0])
            end_chan = int(chan_range.split(",")[-1])
            spectra = np.nanmedian(data[:, start_chan:end_chan, ...], axis=(0, 2, 3))
        else:
            spectra = np.nanmedian(data, axis=(0, 2, 3))
        if timestamp_range != "":
            t_start = int(timestamp_range.split(",")[0])
            t_end = int(timestamp_range.split(",")[-1])
            t_series = np.nanmedian(data[..., t_start:t_end], axis=(0, 1, 2))
        else:
            t_series = np.nanmedian(data, axis=(0, 1, 2))
        t_series = t_series[t_series != 0]
        t_chunksize = find_max_valid_chunk_length(t_series, temporal_tol_factor)
        n_time_interval = int(len(t_series) / t_chunksize)
        if max_ntime > 0 and n_time_interval > max_ntime:
            n_time_interval = max_ntime
    else:
        if chan_range != "":
            start_chan = int(chan_range.split(",")[0])
            end_chan = int(chan_range.split(",")[-1])
            spectra = np.nanmedian(data[:, start_chan:end_chan, ...], axis=(0, 2))
        else:
            spectra = np.nanmedian(data, axis=(0, 2))
        n_time_interval = 1

    spectra = spectra[spectra != 0]
    f_chunksize = find_max_valid_chunk_length(spectra, spectral_tol_factor)
    n_spectral_interval = int(len(spectra) / f_chunksize)
    if max_nchan > 0 and n_spectral_interval > max_nchan:
        n_spectral_interval = max_nchan
    return n_time_interval, n_spectral_interval


def calc_psf(msname, chan_number=-1):
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
    return round(float(psf), 2)


def calc_npix_in_psf(weight, robust=0.0):
    """
    Calculate number of pixels in a PSF (could be fractional)

    Parameters
    ----------
    weight : str
        Image weighting scheme
    robust : float, optional
        Briggs weighting robust parameter (-1 to +1)

    Returns
    -------
    float
        Number of pixels in a PSF
    """
    weight = weight.upper()

    if weight == "NATURAL":
        npix = 5.0
    elif weight == "UNIFORM":
        npix = 3.0
    else:
        # robust: -1 (uniform) → 3, +1 (natural) → 5
        robust = np.clip(robust, -1.0, 1.0)
        npix = 3.0 + ((robust + 1.0) / 2.0) * (5.0 - 3.0)
    return round(npix, 0)


def calc_cellsize(msname, num_pixel_in_psf):
    """
    Calculate pixel size in arcsec

    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : float
        Number of pixels in one PSF

    Returns
    -------
    int
        Pixel size in arcsec
    """
    psf = calc_psf(msname)
    pixel = round(psf / num_pixel_in_psf, 1)
    return pixel


def calc_multiscale_scales(msname, num_pixel_in_psf, chan_number=-1, max_scale=8):
    """
    Calculate multiscale scales

    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : float
        Number of pixels in one PSF
    max_scale : float, optional
        Maximum scale in arcmin

    Returns
    -------
    list
        Multiscale scales in pixel units
    """
    psf = calc_psf(msname, chan_number=chan_number)
    minuv, minuv_l = calc_minuv(msname, chan_number=chan_number)
    max_interferometric_scale = (
        0.5 * np.rad2deg(1.0 / minuv_l) * 60.0
    )  # In arcmin, half of maximum scale
    max_interferometric_scale = min(max_scale, max_interferometric_scale)
    max_scale_pixel = int((max_interferometric_scale * 60.0) / (psf / num_pixel_in_psf))
    current_scale = num_pixel_in_psf
    multiscale_scales = [0, int(current_scale)]
    while True:
        current_scale = current_scale * 3
        if current_scale >= max_scale_pixel:
            multiscale_scales.append(int(max_scale_pixel))
            break
        multiscale_scales.append(int(current_scale))
    return multiscale_scales


def get_multiscale_bias(freq, bias_min=0.6, bias_max=0.9, minfreq=100, maxfreq=200):
    """
    Get frequency dependent multiscale bias

    Parameters
    ----------
    freq : float
        Frequency in MHz
    bias_min : float, optional
        Minimum bias at minimum L-band frequency
    bias_max : float, optional
        Maximum bias at maximum L-band frequency
    minfreq : float, optional
        Minimum frequency range in MHz
    maxfreq : float, optional
        Maximum frequency range in MHz

    Returns
    -------
    float
        Multiscale bias patrameter
    """
    if freq <= minfreq:
        return bias_min
    elif freq >= maxfreq:
        return bias_max
    else:
        freq_min = 100
        freq_max = 200
        logf = np.log10(freq)
        logf_min = np.log10(freq_min)
        logf_max = np.log10(freq_max)
        frac = (logf - logf_min) / (logf_max - logf_min)
        return round(
            np.clip(bias_min + frac * (bias_max - bias_min), bias_min, bias_max), 3
        )
