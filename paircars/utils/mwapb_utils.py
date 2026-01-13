import datetime
import glob
import os
import time
import warnings
import types
import argparse
import numpy as np
import astropy.units as u
import astropy.wcs as pywcs
import mwa_hyperbeam
import skyfield.api as si
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from joblib import Parallel, delayed as jobdelayed
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import FITSFixedWarning
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from .basic_utils import *
from .udocker_utils import *

warnings.filterwarnings("ignore")

datadir = get_datadir()
MWA_PB_file_paircars = f"{datadir}/mwa_full_embedded_element_pattern.h5"
sweet_spot_file_paircars = f"{datadir}/MWA_sweet_spots.npy"
haslam_map_paircars = f"{datadir}/haslam_map.fits"
MWALON = 116.67
MWALAT = -26.7
MWAALT = 377.8
MWAPOS = EarthLocation.from_geodetic(
    lon="116:40:14.93", lat="-26:42:11.95", height=377.8
)


def get_azza_from_fits(filename, metafits):
    """
    Get azimuith and zenith angle arrays from fits file

    Parameters
    ----------
    filename : str
        Name of the fits file
    metafits : str
        Metafits file

    Returns
    -------
    dict
        {'za_rad': theta,'astro_az_rad': phi}
    """
    f = fits.open(filename)
    h = f[0].header
    f.close()
    wcs = pywcs.WCS(h)
    naxes = h["NAXIS"]

    x = np.arange(1, h["NAXIS1"] + 1)
    y = np.arange(1, h["NAXIS2"] + 1)
    Y, X = np.meshgrid(y, x)
    Xflat = X.flatten()
    Yflat = Y.flatten()
    FF = np.ones(Xflat.shape)
    if naxes >= 4:
        Tostack = [Xflat, Yflat, FF]
        for i in range(3, naxes):
            Tostack.append(np.ones(Xflat.shape))
    else:
        Tostack = [Xflat, Yflat]
    pixcrd = np.vstack(Tostack).transpose()
    sky = wcs.wcs_pix2world(pixcrd, 1)

    ################
    # Extract RA-DEC
    ################
    ra = sky[:, 0]
    dec = sky[:, 1]
    RA = ra.reshape(X.shape)
    Dec = dec.reshape(Y.shape)

    ##########################################
    # Get the date so we can convert to Az,El
    ##########################################
    metadata = fits.getheader(metafits)
    d = metadata["DATE-OBS"]
    if "." in d:
        d = d.split(".")[0]
    dt = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
    mwatime = Time(dt)

    #############################
    # Transform to Alt-Az
    #############################
    source = SkyCoord(
        ra=RA, dec=Dec, frame="icrs", unit=(u.deg, u.deg)
    )
    source.location = MWAPOS
    source.obstime = mwatime
    s = time.time()
    source_altaz = source.transform_to("altaz")
    Alt, Az = (
        source_altaz.alt.deg,
        source_altaz.az.deg,
    )
    theta = (90 - Alt) * np.pi / 180
    phi = Az * np.pi / 180
    return {"za_rad": theta.transpose(), "astro_az_rad": phi.transpose()}


def get_IQUV(filename, stokesaxis=4):
    """Get IQUV from a fits file"""
    data = fits.getdata(filename)
    stokes = {}
    if stokesaxis == 3:
        stokes["I"] = data[0, 0, :, :]
        stokes["Q"] = data[0, 1, :, :]
        stokes["U"] = data[0, 2, :, :]
        stokes["V"] = data[0, 3, :, :]
    elif stokesaxis == 4:
        stokes["I"] = data[0, 0, :, :]
        stokes["Q"] = data[1, 0, :, :]
        stokes["U"] = data[2, 0, :, :]
        stokes["V"] = data[3, 0, :, :]
    else:
        stokes["I"] = data[0, 0, :, :]
        stokes["Q"] = data[0, 0, :, :] * 0
        stokes["U"] = data[0, 0, :, :] * 0
        stokes["V"] = data[0, 0, :, :] * 0
    return stokes


def get_inst_pols(stokes):
    """Return instrumental polaristations matrix (Vij)"""
    XX = stokes["I"] + stokes["Q"]
    XY = stokes["U"] + stokes["V"] * 1j
    YX = stokes["U"] - stokes["V"] * 1j
    YY = stokes["I"] - stokes["Q"]
    Vij = np.array([[XX, XY], [YX, YY]])
    Vij = np.swapaxes(np.swapaxes(Vij, 0, 2), 1, 3)
    return Vij


def B2IQUV(B, iau_order=False):
    """
    Convert sky brightness matrix to I, Q, U, V

    Parameters
    ----------
    B : numpy.array
        Brightness matrix array
    iau_order : bool, optional
        Whether brightness matrix is in IAU or MWA convention

    Returns
    -------
    dict
        Stokes dictionary
    """
    B11 = B[:, :, 0, 0]
    B12 = B[:, :, 0, 1]
    B21 = B[:, :, 1, 0]
    B22 = B[:, :, 1, 1]
    if iau_order:
        stokes = {}
        stokes["I"] = (B11 + B22) / 2.0
        stokes["Q"] = (B11 - B22) / 2.0
        stokes["U"] = (B12 + B21) / 2.0
        stokes["V"] = 1j * (B21 - B12) / 2.0
    else:
        stokes = {}
        stokes["I"] = (B11 + B22) / 2.0
        stokes["Q"] = (B22 - B11) / 2.0
        stokes["U"] = (B21 + B12) / 2.0
        stokes["V"] = 1j * (B12 - B21) / 2.0
    return stokes


def all_sky_beam_interpolator(
    sweet_spot_num,
    freq,
    resolution,
    ncpu=-1,
    MWA_PB_file="",
    sweet_spot_file="",
    iau_order=False,
):
    """
    Calculate all sky beam interpolation for given sweet spot pointing

    Parameters
    ----------
    sweet_spot_num : int
        Sweet spot number
    freq : float
        Frequency in MHz
    resolution : float
        Spatial resolution in degree
    ncpu : int, optional
        Number of CPU threads to use
    MWA_PB_file : str, optional
        MWA primary beam file
    sweet_spot_file : str, optional
        MWA sweet spot file name
    iau_order : bool, optional
        PB Jones in IAU order or not

    Returns
    -------
    numpy.array
        All sky primary beam Jones array
    """
    if MWA_PB_file == "" or os.path.exists(MWA_PB_file) is False:
        MWA_PB_file = MWA_PB_file_paircars
    if sweet_spot_file == "" or os.path.exists(sweet_spot_file) is False:
        sweet_spot_file = sweet_spot_file_paircars
    ncpu = max(1, ncpu)
    os.environ["RAYON_NUM_THREADS"] = str(ncpu)
    beam = mwa_hyperbeam.FEEBeam(MWA_PB_file)
    az_scale = np.arange(0, 360, resolution)
    alt_scale = np.arange(0, 90, resolution)
    az, alt = np.meshgrid(az_scale, alt_scale)
    za_rad = np.deg2rad(90 - alt.ravel())  # Zenith angle in radian
    az_rad = np.deg2rad(az.ravel())  # Azimuth in radian
    sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
    delay = sweet_spots[int(sweet_spot_num)][-1]
    ##############################################
    # Calculating Jones array in 1 deg alt-az grid
    ##############################################
    jones = beam.calc_jones_array(
        az_rad,
        za_rad,
        freq,
        delay,
        [1] * 16,
        True,
        np.deg2rad(MWALAT),
        iau_order,
    )
    jones = jones.swapaxes(0, 1).reshape(4, alt_scale.shape[0], az_scale.shape[0])
    j00_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[0, ...]))
    )
    j00_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[0, ...]))
    )
    j01_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[1, ...]))
    )
    j01_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[1, ...]))
    )
    j10_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[2, ...]))
    )
    j10_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[2, ...]))
    )
    j11_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[3, ...]))
    )
    j11_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[3, ...]))
    )
    return j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i


def get_jones_array(
    alt_arr,
    az_arr,
    freq,
    gridpoint,
    ncpu=-1,
    interpolated=True,
    MWA_PB_file="",
    sweet_spot_file="",
    iau_order=False,
):
    """
    Get primary beam jones matrix

    Parameters
    ----------
    alt_arr : numpy.array
        Flattened altitude array in degrees
    az_arr : numpy.array
        Flattened azimuth array in degrees
    gridpoint : int
        Gridpoint number
    ncpu : int, optional
        Number of CPU threads to use
    interpolated : bool, optional
        Use spatially interpolated beams or not
    MWA_PB_file : str, optional
        Primary beam file name
    sweet_spot_file : str, optional
        MWA sweet spot file name
    iau_order : bool, optional
        IAU order of the beam

    Returns
    -------
    numpy.array
        Jones array (shape : coordinate_arr_shape, 2 ,2)
    """
    if MWA_PB_file == "" or os.path.exists(MWA_PB_file) is False:
        MWA_PB_file = MWA_PB_file_paircars
    if sweet_spot_file == "" or os.path.exists(sweet_spot_file) is False:
        sweet_spot_file = sweet_spot_file_paircars
    ncpu = min(max(1, ncpu), 8)
    if interpolated:
        j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i = (
            all_sky_beam_interpolator(
                int(gridpoint),
                freq,
                0.25,
                ncpu=ncpu,
                MWA_PB_file=MWA_PB_file,
                sweet_spot_file=sweet_spot_file,
                iau_order=iau_order,
            )
        )
        # Change resolution based on frequency
        with Parallel(n_jobs=ncpu, backend="multiprocessing") as parallel:
            results = parallel(
                [
                    jobdelayed(j00_r)(alt_arr, az_arr, grid=False),
                    jobdelayed(j00_i)(alt_arr, az_arr, grid=False),
                    jobdelayed(j01_r)(alt_arr, az_arr, grid=False),
                    jobdelayed(j01_i)(alt_arr, az_arr, grid=False),
                    jobdelayed(j10_r)(alt_arr, az_arr, grid=False),
                    jobdelayed(j10_i)(alt_arr, az_arr, grid=False),
                    jobdelayed(j11_r)(alt_arr, az_arr, grid=False),
                    jobdelayed(j11_i)(alt_arr, az_arr, grid=False),
                ]
            )
        del parallel
        (
            j00_r_arr,
            j00_i_arr,
            j01_r_arr,
            j01_i_arr,
            j10_r_arr,
            j10_i_arr,
            j11_r_arr,
            j11_i_arr,
        ) = results
        j00 = j00_r_arr + 1j * j00_i_arr
        j01 = j01_r_arr + 1j * j01_i_arr
        j10 = j10_r_arr + 1j * j10_i_arr
        j11 = j11_r_arr + 1j * j11_i_arr
        j00 = j00.reshape(az_arr.shape)
        j01 = j01.reshape(az_arr.shape)
        j10 = j10.reshape(az_arr.shape)
        j11 = j11.reshape(az_arr.shape)
        jones_array = np.array([j00, j01, j10, j11]).T
    else:
        ncpu = max(1, ncpu)
        os.environ["RAYON_NUM_THREADS"] = str(ncpu)
        beam = mwa_hyperbeam.FEEBeam(MWA_PB_file)
        sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
        delay = sweet_spots[int(gridpoint)][-1]
        za_arr = 90 - alt_arr
        jones_array = beam.calc_jones_array(
            np.deg2rad(az_arr),
            np.deg2rad(za_arr),
            freq,
            delay,
            [1] * 16,
            True,
            np.deg2rad(MWALAT),
            iau_order,
        )
    jones_array = jones_array.reshape(jones_array.shape[0], 2, 2)
    return jones_array


def get_pb_radec(
    ra,
    dec,
    freq,
    metafits,
    ncpu=-1,
    MWA_PB_file="",
    sweet_spot_file="",
    iau_order=False,
):
    """
    Function to get MWA primary beam at specific RA, DEC

    Parameters
    ----------
    ra : str
        RA either in degree or 'hh:mm:ss' or '%fh%fm%fs' format
    dec : str
        DEC either in degree or 'dd:mm:ss' or '%fd%fm%fs'format
    freq : float
        Frequency in MHz
    metafits : str
        MWA metafits file
    ncpu : int, optional
        Number of CPU threads
    MWA_PB_file : str, optional
        MWA primary beam file path
    sweet_spot_file : str, optional
        Sweetspot file name
    iau_order : bool, optional
        Beam Jones in IAU order or not

    Returns
    -------
    int
        Success message (0 or 1)
    np.array
        Jones matrix
    float
        Stokes I beam value
    float
        XX power beam value
    float
        YY power beam value
    """
    if MWA_PB_file == "" or os.path.exists(MWA_PB_file) is False:
        MWA_PB_file = MWA_PB_file_paircars
    if sweet_spot_file == "" or os.path.exists(sweet_spot_file) is False:
        sweet_spot_file = sweet_spot_file_paircars
    ncpu = max(1, ncpu)
    os.environ["RAYON_NUM_THREADS"] = str(ncpu)
    beam = mwa_hyperbeam.FEEBeam(MWA_PB_file)
    metadata = fits.getheader(metafits)
    obstime = metadata["DATE-OBS"]
    gridpoint = metadata["GRIDNUM"]
    sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
    delay = sweet_spots[int(gridpoint)][-1]
    observing_time = Time(obstime)
    aa = AltAz(location=MWAPOS, obstime=observing_time)
    try:
        ra = float(ra)
        dec = float(dec)
        coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    except:
        try:
            coord = SkyCoord(ra, dec)
        except:
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    altaz_object = coord.transform_to(aa)
    alt = altaz_object.alt.degree
    az = altaz_object.az.degree

    # Decide whether we have arrays or scalars
    is_array = isinstance(ra, np.ndarray) and isinstance(dec, np.ndarray)

    # Compute Jones matrices
    if is_array:
        jones = beam.calc_jones_array(
            np.deg2rad(az),
            np.deg2rad(90 - alt),
            freq * 1e6,
            delay,
            [1] * 16,
            True,
            np.deg2rad(MWALAT),
            iau_order,
        )
    else:
        jones = beam.calc_jones(
            np.deg2rad(az),
            np.deg2rad(90 - alt),
            freq * 1e6,
            delay,
            [1] * 16,
            True,
            np.deg2rad(MWALAT),
            iau_order,
        )
        # Promote scalar → (1,4) for uniform handling
        jones = np.asarray(jones)[None, :]

    # Jones shape: (N, 4)
    J = np.abs(jones) ** 2

    stokesI_beam = (J[:, 0] + J[:, 1] + J[:, 2] + J[:, 3]) / 2
    power_beam_array_XX = J[:, 0] + J[:, 1]
    power_beam_array_YY = J[:, 2] + J[:, 3]

    # If scalar input, return scalars
    if not is_array:
        stokesI_beam = stokesI_beam[0]
        power_beam_array_XX = power_beam_array_XX[0]
        power_beam_array_YY = power_beam_array_YY[0]
        jones = jones[0]

    return 0, jones, stokesI_beam, power_beam_array_XX, power_beam_array_YY


def get_haslam(freq, scaling=-2.55):
    """
    Get the Haslam 408 MHz all sky map extrapolated to desired frequency.

    Parameters
    ----------
    freq : float
        Frequency in MHz
    scaling : float, optional
        Power law index for extrapolation (default : -2.55)

    Returns
    -------
    dict
        A python dictionary
        i) Haslam map at given frequency in 10xK unit
        ii) RA in degree
        iii) DEC in degree
    """
    if not os.path.exists(haslam_map_paircars):
        print(f"Could not find 408 MHz image: {haslam_map_paircars}.")
        return None
    try:
        f = fits.open(haslam_map_paircars)
    except Exception:
        print(f"Error opening 408 MHz image: {haslam_map_paircars}.")
        return None
    skymap = f[0].data[0] / 10.0  # Haslam map is in 10xK
    skymap = skymap * (freq / 408.0) ** scaling  # Scale to frequency
    RA_1D = f[0].header.get("CRVAL1") + (
        np.arange(1, skymap.shape[1] + 1) - f[0].header.get("CRPIX1")
    ) * f[0].header.get("CDELT1")
    dec_1D = f[0].header.get("CRVAL2") + (
        np.arange(1, skymap.shape[0] + 1) - f[0].header.get("CRPIX2")
    ) * f[0].header.get("CDELT2")
    return {"skymap": skymap, "RA": RA_1D, "dec": dec_1D}  # RA, dec in degs


def map_sky_haslam(skymap, RA, dec, az_grid, za_grid, obstime=""):
    """
    Reprojects Haslam map onto an input az, ZA grid.

    Parameters
    ----------
    skymap : np.array
        Haslam map in RA DEC
    RA : np.array
        1D range of RAs (deg)
    dec : np.array
        1D range of decs (deg)
    az_grid : np.array
        Grid of azes onto which we map sky
    za_grid : np.array
        Grid of ZAs onto which we map sky
    obstime : str, optional
        Time of the observation in 'yyyy-mm-dd hh:mm:ss' format

    Returns
    -------
    numpy.array
        Sky map array
    """
    # Get az, ZA grid transformed to equatorial coords
    grid2eq = horz2eq(az_grid, za_grid, obstime)
    my_interp_fn = RegularGridInterpolator(
        (dec, RA[::-1]), skymap[:, ::-1], bounds_error=False, fill_value=np.nan
    )
    # interpolate map onto az,ZA grid
    # Convert to RA=-180 - 180 format (same as Haslam)
    # We do it this way so RA values are always increasing for RegularGridInterpolator
    grid2eq["RA"][grid2eq["RA"] > 180] = grid2eq["RA"][grid2eq["RA"] > 180] - 360
    my_map = my_interp_fn(np.dstack([grid2eq["DEC"], grid2eq["RA"]]))
    return my_map


def map_sky(skymap, RA, dec, az_grid, za_grid, obstime=""):
    """
    Reprojects Haslam map onto an input az, ZA grid.

    Parameters
    ----------
    skymap : np.array
        Haslam map in RA DEC
    RA : np.array
        1D range of RAs (deg)
    dec : np.array
        1D range of decs (deg)
    az_grid : np.array
        Grid of azes onto which we map sky
    za_grid : np.array
        Grid of ZAs onto which we map sky
    obstime : str, optional
        Time of the observation in 'yyyy-mm-dd hh:mm:ss' format

    Returns
    -------
    numpy.array
        Sky map
    """
    # Get az, ZA grid transformed to equatorial coords
    grid2eq = horz2eq(az_grid, za_grid, obstime)
    my_interp_fn = RegularGridInterpolator(
        (dec, RA[::-1]), skymap[:, ::-1], bounds_error=False, fill_value=np.nan
    )
    # interpolate map onto az,ZA grid
    # We do it this way so RA values are always increasing for RegularGridInterpolator
    my_map = my_interp_fn(np.dstack([grid2eq["DEC"], grid2eq["RA"]]))
    return my_map


def horz2eq(az, ZA, obstime):
    """
    Convert from horizontal (az, ZA) to equatorial (RA, dec)grid2eq = horz2eq(az_grid, za_grid, obstime)

    Parameters
    ----------
    az : np.array
        Grid of azes onto which we map sky
    za : np.array
        Grid of ZAs onto which we map sky
    obstime : str
        Time of the observation in 'yyyy-mm-dd hh:mm:ss' format

    Returns
    -------
    dict
        A python dictionary {'RA' : degress, 'DEC' : degrees}
    """
    MWA_TOPO = si.Topos(
        longitude=(116, 40, 14.93), latitude=(-26, 42, 11.95), elevation_m=377.8
    )
    skyfield_loader = si.Loader(datadir, verbose=False, expire=True)
    PLANETS = skyfield_loader("de421.bsp")
    TIMESCALE = skyfield_loader.timescale(builtin=True)
    S_MWAPOS = PLANETS["earth"] + MWA_TOPO
    observing_time = Time(obstime)
    observer = S_MWAPOS.at(TIMESCALE.from_astropy(observing_time))
    coords = observer.from_altaz(
        alt_degrees=(90 - ZA), az_degrees=az, distance=si.Distance(au=9e90)
    )
    ra_a, dec_a, _ = coords.radec()
    return {"RA": ra_a._degrees, "DEC": dec_a.degrees}


def get_image_info(image):
    """
    Get the image data and its coordinates

    Parameters
    ----------
    image : str
        Name of the image

    Returns
    -------
    dict
        A python dictionary
            i) Map
            ii) RA in degree
            iii) DEC in degree
    """
    if not os.path.exists(image):
        print("Could not find 408 MHz image: %s\n" % (image))
        return None
    try:
        f = fits.open(image)
    except Exception:
        print("Error opening 408 MHz image: %s\n" % (image))
        return None
    skymap = f[0].data[0, 0, ...]
    RA_1D = f[0].header.get("CRVAL1") + (
        np.arange(1, skymap.shape[1] + 1) - f[0].header.get("CRPIX1")
    ) * f[0].header.get(
        "CDELT1"
    )  # /15.0
    dec_1D = f[0].header.get("CRVAL2") + (
        np.arange(1, skymap.shape[0] + 1) - f[0].header.get("CRPIX2")
    ) * f[0].header.get("CDELT2")
    return {"skymap": skymap, "RA": RA_1D, "dec": dec_1D}  # RA, dec in degs


def makeAZZA_dOMEGA(npix, projection="SIN"):
    """
    Make azimuth and zenith angle arrays for a square image of side npix

    Parameters
    ----------
    npix : int
        Number of pixels of the grid
    projection : str, optional
        SIN or ZEA

    Returns
    -------
    list
        Azimuth angles in radians
    list
        Zenith angle in radians
    int
        Total number of pixels above the horizon
    float
        Differential solid angle (dOMEGA)
    """
    # build az and za arrays
    # use linspace to ensure we go to horizon on all sides
    z = np.linspace(-npix / 2.0, npix / 2.0, num=npix, dtype=np.float32)
    x = np.empty((npix, npix), dtype=np.float32)
    y = np.empty((npix, npix), dtype=np.float32)
    dOMEGA = np.empty((npix, npix), dtype=np.float32)
    for i in range(npix):
        y[i, 0:] = z
        x[0:, i] = z
    d = np.sqrt(x * x + y * y) / (npix / 2)
    # only select pixels above horizon
    t = d <= 1.0
    n_total = t.sum()
    dOMEGA.fill(np.pi * 2.00 / n_total)
    za = np.zeros((npix, npix), dtype=np.float32) * np.NaN
    if projection == "SIN":
        za[t] = np.arcsin(d[t])
        dOMEGA = np.cos(za) * np.pi * 2.00 / n_total
    elif projection == "ZEA":
        d = d * 2**0.5  # ZEA requires R to extend beyond 1.
        za[t] = 2 * np.arcsin(d[t] / 2.0)
    else:
        e = "Projection %s not found" % projection
        print(e)
        raise ValueError(e)
    az = np.arctan2(y, x)
    az = az + np.pi  # 0 to 2pi
    az = 2 * np.pi - az  # Change to clockwise from top (when origin is in top-left)
    return az, za, n_total, dOMEGA


def get_fringe(msname, freq, metafits, resolution=1, nthreads=1, baseline=[]):
    """
    Function to calculate all sky fringe of a baseline

    Parameters
    ----------
    msname : str
        Name of the measurement set
    freq : float
        Frequency in MHz
    metafits : str
        Name of the metafits file
    resolution : float, optional
        Beam resolution in degree (default : 1deg)
    nthreads : int, optional
        Number of cpu threads use for parallel computing
    baseline : list, optional
        Antenna list of a baseline

    Returns
    -------
    np.array
        All-sky fringe array in sky coornidinate
    """
    try:
        msname = msname.rstrip("/")
        baseline_str = str(baseline[0]) + "&&" + str(baseline[1])
        bs_ms = (
            os.path.abspath(msname).split(".ms")[0]
            + "_"
            + str(baseline[0])
            + "_"
            + str(baseline[1])
            + "_"
            + str(freq)
            + ".ms"
        )
        if os.path.exists(bs_ms):
            os.system(f"rm -rf {bs_ms}")
        split(
            vis=msname,
            outputvis=bs_ms,
            spw="0:" + str(freq) + "MHz",
            antenna=baseline_str,
            datacolumn="data",
        )
        imsize = 512
        cellsize = int(110 * 3600 / 512)
        imagename_prefix = os.path.abspath(bs_ms).split(".ms")[0] + "_fringe"
        wsclean_args = [
            "-size " + str(imsize) + " " + str(imsize),
            "-scale " + str(cellsize) + "asec",
            "-niter 0",
            "-make-psf-only",
            "-no-fit-beam",
            "-pol i",
            "-j " + str(nthreads),
            "-name " + imagename_prefix,
            "-quiet",
        ]
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + bs_ms
        msg = run_wsclean(wsclean_cmd, "solarwsclean", verbose=False)
        wsclean_image_list = glob.glob(f"{imagename_prefix}*psf*.fits")
        wsclean_psf = wsclean_image_list[0]
        obstime = fits.getheader(metafits)["DATE-OBS"]
        n_pix = int(360 / resolution)
        az_grid, za_grid, n_total, dOMEGA = makeAZZA_dOMEGA(n_pix, "ZEA")
        az_grid = az_grid * 180 / np.pi
        za_grid = za_grid * 180 / np.pi
        fringe_map = get_image_info(wsclean_psf)
        sky_grid = map_sky(
            fringe_map["skymap"],
            fringe_map["RA"],
            fringe_map["dec"],
            az_grid,
            za_grid,
            obstime=obstime,
        )
        os.system(f"rm -rf {imagename_prefix}*")
        os.system("rm -rf {bs_ms}")
        return sky_grid
    except:
        return []


def make_primarybeammap(
    msname,
    metafits,
    baselines=[],
    freq=0,
    obstime="",
    resolution=1,
    iau_order=True,
    MWA_PB_file="",
    sweet_spot_file="",
    nthreads=1,
    calc_fringe_temp=False,
):
    """
    Parameters
    ----------
    msname : str
        Measurement set
    metafits : str
        Metafits file
    freq : float, optional
        Frequency in MHz
    obstime : str, optional
        Time of the observation in 'yyyy-mm-dd hh:mm:ss' format (If not given automatically obtain from metafits file)
    resolution : float, optional
        Beam resolution in degree (default : 1deg)
    iau_order : bool, optional
        Beam in IAU order or not
    MWA_PB_file : str, optional
        MWA primary beam file path
    sweet_spot_file: str, optional
        MWA sweet spot file
    nthreads : int, optional
        Number of cpu threads use for parallel computing
    calc_fringe_temp : bool, optional
        Calculate temperature contribution of the baseline

    Returns
    -------
    float
        Sum of full Beam*Sky (XX)
    float
        Sum of full Beam (XX)
    float
        Antenna temperature (XX)
    float
        Total beam area (XX)
    float
        Sum of full Beam*Sky (YY)
    float
        Sum of full Beam (YY)
    float
        Antenna temperature (YY)
    float
        Total beam area (YY)
    """
    warnings.filterwarnings("ignore")
    if MWA_PB_file == "" or os.path.exists(MWA_PB_file) is False:
        MWA_PB_file = MWA_PB_file_paircars
    if sweet_spot_file == "" or os.path.exists(sweet_spot_file) is False:
        sweet_spot_file = sweet_spot_file_paircars
    nthreads = max(1, nthreads)
    os.environ["RAYON_NUM_THREADS"] = str(nthreads)
    beam = mwa_hyperbeam.FEEBeam(MWA_PB_file)

    ############################
    # Creating sky grid
    ############################
    n_pix = int(360 / resolution)
    az_grid, za_grid, n_total, dOMEGA = makeAZZA_dOMEGA(n_pix, "ZEA")
    az_grid = az_grid * 180 / np.pi
    za_grid = za_grid * 180 / np.pi
    alt_grid = 90 - (za_grid)
    # first go from altitude to zenith angle
    theta = (90 - alt_grid) * np.pi / 180
    phi = az_grid * np.pi / 180

    ###############################
    # Determining beamformer delays
    ###############################
    metadata = fits.getheader(metafits)
    gridpoint = metadata["GRIDNUM"]
    sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
    delay = sweet_spots[int(gridpoint)][-1]
    obstime = metadata["DATE-OBS"]

    #################################
    # Calculating beam array
    #################################
    jones_array = beam.calc_jones_array(
        phi.flatten(),
        theta.flatten(),
        freq * 10**6,
        delay,
        [1] * 16,
        True,
        np.deg2rad(MWALAT),
        iau_order,
    )
    power_beam_array = {}
    power_beam_array["XX"] = np.abs(
        jones_array[:, 0] * jones_array[:, 0].conjugate()
        + jones_array[:, 1] * jones_array[:, 1].conjugate()
    ).reshape(az_grid.shape)
    power_beam_array["YY"] = np.abs(
        jones_array[:, 2] * jones_array[:, 2].conjugate()
        + jones_array[:, 3] * jones_array[:, 3].conjugate()
    ).reshape(az_grid.shape)

    #######################################
    # Get Haslam and interpolate onto grid
    #######################################
    haslam_map = get_haslam(freq)
    mask = np.isnan(za_grid)
    za_grid[np.isnan(za_grid)] = 90.0  # Replace nans as they break the interpolation
    sky_grid = map_sky_haslam(
        haslam_map["skymap"],
        haslam_map["RA"],
        haslam_map["dec"],
        az_grid,
        za_grid,
        obstime=obstime,
    )
    sky_grid[mask] = np.nan  # Remask beyond the horizon

    #######################################
    # Calculate sky fringe
    #######################################
    beamsky_sum_XX = 0
    beam_sum_XX = 0
    Tant_XX = 0
    beam_dOMEGA_sum_XX = 0
    beamsky_sum_YY = 0
    beam_sum_YY = 0
    Tant_YY = 0
    beam_dOMEGA_sum_YY = 0
    pols = ["XX", "YY"]
    fringe_list = []
    param_list = []
    if calc_fringe_temp and len(baselines) > 0:
        for bs in baselines:
            fringe = get_fringe(
                msname,
                freq,
                metafits,
                resolution=resolution,
                nthreads=nthreads,
                baseline=bs,
            )
            time.sleep(0.5)
            if len(fringe) > 0:
                fringe_list.append(fringe)

    ###############################
    # Calculate sky beam
    ###############################
    for pol in pols:
        # Get gridded sky
        beam = power_beam_array[pol]
        beamsky = beam * sky_grid
        beam_dOMEGA = beam * dOMEGA
        beamsky_sum = np.nansum(beamsky)
        beam_sum = np.nansum(beam)
        beam_dOMEGA_sum = np.nansum(beam_dOMEGA)
        Tant = np.nansum(beamsky) / np.nansum(beam)
        if pol == "XX":
            beamsky_sum_XX = beamsky_sum
            beam_sum_XX = beam_sum
            Tant_XX = Tant
            beam_dOMEGA_sum_XX = beam_dOMEGA_sum
            T_fringe_XX = 0
            if len(fringe_list) > 0:
                for i in range(len(fringe_list)):
                    fringe = fringe_list[i]
                    T_fringe_XX += np.abs(np.nansum(fringe * beamsky) / np.nansum(beam))
        if pol == "YY":
            beamsky_sum_YY = beamsky_sum
            beam_sum_YY = beam_sum
            Tant_YY = Tant
            beam_dOMEGA_sum_YY = beam_dOMEGA_sum
            T_fringe_YY = 0
            if len(fringe_list) > 0:
                for i in range(len(fringe_list)):
                    fringe = fringe_list[i]
                    T_fringe_YY += np.abs(np.nansum(fringe * beamsky) / np.nansum(beam))
    return (
        beamsky_sum_XX,
        beam_sum_XX,
        Tant_XX,
        beam_dOMEGA_sum_XX,
        beamsky_sum_YY,
        beam_sum_YY,
        Tant_YY,
        beam_dOMEGA_sum_YY,
        T_fringe_XX,
        T_fringe_YY,
    )


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
