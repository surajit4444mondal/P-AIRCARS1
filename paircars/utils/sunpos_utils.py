import astropy.units as u
import os
import copy
import traceback
import numpy as np
from astropy.time import Time
from astropy.coordinates import (
    EarthLocation,
    AltAz,
    get_sun,
    solar_system_ephemeris,
    SkyCoord,
)
from astropy.io import fits
from astropy.wcs import WCS
from casatools import msmetadata
from .basic_utils import get_datadir, mjdsec_to_timestamp
from .udocker_utils import run_solar_sidereal_cor, run_chgcenter
from .image_utils import create_circular_mask_array
from .imaging import calc_sun_dia

#####################################
# Sun position related
#####################################
datadir = get_datadir()
try:
    solar_system_ephemeris.set(f"{datadir}/de440s")
except Exception:
    solar_system_ephemeris.set("builtin")


def get_solar_elevation(lat, lon, elev, date_time):
    """
    Get solar elevation

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    elev : float
        Elevation in degrees
    date_time : str
        Date time in YYYY-MM-DDThh:mm:ss (ISOT) format, default : present time


    Returns
    -------
    float
        Solar elevation in degree
    """
    latitude = lat * u.deg  # In degree
    longitude = lon * u.deg  # In degree
    elevation = elev * u.m  # In meter
    if date_time == "":
        astro_time = Time.now()
    else:
        astro_time = Time(date_time)
    location = EarthLocation(lat=latitude, lon=longitude, height=elevation)
    sun_coords = get_sun(astro_time)  # In GCRS (geocentric frame)
    altaz_frame = AltAz(obstime=astro_time, location=location)
    sun_altaz = sun_coords.transform_to(altaz_frame)
    solar_elevation = sun_altaz.alt.deg
    return round(solar_elevation, 3)


def radec_sun(msname):
    """
    RA DEC of the Sun at the midpoint of scan (offline version)

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    str
        RA DEC of the Sun in J2000
    str
        RA string
    str
        DEC string
    float
        RA in degree
    float
        DEC in degree
    """
    msmd = msmetadata()
    msmd.open(msname)
    times = msmd.timesforspws(0)
    msmd.close()
    msmd.done()
    mid_time = times[int(len(times) / 2)]
    mid_timestamp = mjdsec_to_timestamp(mid_time)
    astro_time = Time(mid_timestamp, scale="utc")
    sun_coord = get_sun(astro_time)  # In GCRS (geocentric frame)
    sun_ra = (
        f"{int(sun_coord.ra.hms.h)}h"
        f"{int(sun_coord.ra.hms.m)}m"
        f"{round(sun_coord.ra.hms.s, 2)}s"
    )
    sun_dec = (
        f"{int(sun_coord.dec.dms.d)}d"
        f"{abs(int(sun_coord.dec.dms.m))}m"
        f"{abs(round(sun_coord.dec.dms.s, 2))}s"
    )
    sun_radec_string = f"J2000 {sun_ra} {sun_dec}"
    return (
        sun_radec_string,
        sun_ra,
        sun_dec,
        sun_coord.ra.deg,
        sun_coord.dec.deg,
    )


def radec_sun_at_time(timestamp):
    """
    RA DEC of the Sun a given time

    Parameters
    ----------
    timestamp : str
        Time in format dd-mm-yyyyThh:mm:ss

    Returns
    -------
    str
        RA DEC of the Sun in J2000
    str
        RA string
    str
        DEC string
    float
        RA in degree
    float
        DEC in degree
    """
    astro_time = Time(timestamp, scale="utc")
    sun_coord = get_sun(astro_time)  # In GCRS (geocentric frame)
    sun_ra = (
        f"{int(sun_coord.ra.hms.h)}h"
        f"{int(sun_coord.ra.hms.m)}m"
        f"{round(sun_coord.ra.hms.s, 2)}s"
    )
    sun_dec = (
        f"{int(sun_coord.dec.dms.d)}d"
        f"{abs(int(sun_coord.dec.dms.m))}m"
        f"{abs(round(sun_coord.dec.dms.s, 2))}s"
    )
    sun_radec_string = f"J2000 {sun_ra} {sun_dec}"
    return (
        sun_radec_string,
        sun_ra,
        sun_dec,
        sun_coord.ra.deg,
        sun_coord.dec.deg,
    )


def move_to_sun(msname, ncpu=1, only_uvw=False):
    """
    Move the phasecenter of the measurement set at the center of the Sun (Assuming ms has one scan)

    Parameters
    ----------
    msname : str
        Name of the measurement set
    ncpu : int, optional
        Number of CPU threads to use
    only_uvw : bool, optional
        Note: This is required when visibilities are properly phase rotated in correlator to track the Sun,
        but while creating the MS, UVW values are estimated using a wrong phase center at the start of solar center at the start.

    Returns
    -------
    int
        Success message
    """
    msname = msname.rstrip("/")
    os.system(f"rm -rf {msname}/.solarcenter_move_*")
    print(f"Moving phasecenter to solar center for measurement set: {msname}")
    sun_radec_string, sunra, sundec, sunra_deg, sundec_deg = radec_sun(msname)
    msg = run_chgcenter(
        msname,
        sunra,
        sundec,
        ncpu=ncpu,
        only_uvw=only_uvw,
        container_name="paircarswsclean",
    )
    if msg != 0:
        print("Phasecenter could not be shifted.")
        os.system(f"touch {msname}/.solarcenter_move_failed")
    else:
        os.system(f"touch {msname}/.solarcenter_move_succeed")
    return msg


def determine_quiet_disk(imagename, sigma=10):
    """
    Determine whether disk is visible or not

    Parameters
    ----------
    imagename : str
        Imagename
    sigma : float, optional
        Threshold

    Returns
    -------
    bool
        Whether disk is detected or not
    float
        Emission area radius in arcmin
    """
    if os.path.exists(imagename) is False:
        return False, 0.0
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import remove_small_objects, convex_hull_image

    try:
        data = fits.getdata(imagename)
        header = fits.getheader(imagename)
        bmaj = header["BMAJ"] * 3600.0
        bmin = header["BMIN"] * 3600.0
        if header["CTYPE3"] == "FREQ":
            freqMHz = float(header["CRVAL3"]) / 10**6  # In MHz
            sun_dia = calc_sun_dia(freqMHz)  # In arcmin
        elif header["CTYPE4"] == "FREQ":
            freqMHz = float(header["CRVAL4"]) / 10**6  # In MHz
            sun_dia = calc_sun_dia(freqMHz)  # In arcmin
        else:
            sun_dia = 32  # In arcmin
        cellsize = float(abs(header["CDELT1"])) * 3600.0  # In arcsec
        npix_psf = int(min(bmaj, bmin) / cellsize)
        if npix_psf <= 3:
            gauss_filter_sigma = 1
        elif npix_psf > 3 and npix_psf <= 5:
            gauss_filter_sigma = 2
        else:
            gauss_filter_sigma = 3
        if data.ndim == 4:
            data2d = data[0, 0, ...]
        elif data.ndim == 3:
            data2d = data[0, ...]
        else:
            data2d = data
        data2d = gaussian_filter(data2d, sigma=gauss_filter_sigma)
        max_pos = np.where(data2d == np.nanmax(data2d))
        center_x, center_y = max_pos[1][0], max_pos[0][0]
        sun_rad_pix = 2 * sun_dia * 60 / cellsize  # 4 solar radii
        masked_array = create_circular_mask_array(
            data2d, sun_rad_pix, center_x=center_x, center_y=center_y
        )
        masked_data2d = copy.deepcopy(data2d)
        masked_data2d[masked_array] = np.nan
        rms = np.nanstd(masked_data2d)
        mask = data2d < sigma * rms
        min_size = int(min(bmaj, bmin) / cellsize)
        mask_clean = remove_small_objects(~mask, min_size=min_size)
        mask_clean = convex_hull_image(mask_clean)
        data2d[~mask_clean] = False
        data2d[mask_clean] = True
        area = np.nansum(data2d) * cellsize**2
        radius = np.sqrt(area / np.pi) / 60.0
        if radius >= (sun_dia / 2.0):
            disk_detected = True
        else:
            disk_detected = False
        return disk_detected, radius
    except Exception:
        print("Disk detection is failed.")
        return False, 0


def cal_apparent_solarcenter(imagename, sigma=10, use_gaussian=False):
    """
    Calculate the apparent solar center of the image

    Parameters
    ----------
    imagename : str
        Name of the image
    sigma : float
        If Gaussian fitting is not used, threshold for estimating center of mass as solar center (default =10)
    use_gaussian : bool, optional
        Use gaussian fitting or not

    Returns
    -------
    int
        Success message
    float
        RA of the apparent solar center in degree
    float
        DEC of the apparent solarcenter in degree
    int
        Apparent RA pixel
    int
        Apparent DEC pixel
    """

    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
        x, y = xy
        g = offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
        )
        return g.ravel()

    try:
        data = fits.getdata(imagename)
        header = fits.getheader(imagename)
        if header["CTYPE3"] == "FREQ":
            freqMHz = float(header["CRVAL3"]) / 10**6  # In MHz
            sun_dia = calc_sun_dia(freqMHz)  # In arcmin
        elif header["CTYPE4"] == "FREQ":
            freqMHz = float(header["CRVAL4"]) / 10**6  # In MHz
            sun_dia = calc_sun_dia(freqMHz)  # In arcmin
        else:
            sun_dia = 32  # In arcmin
        cellsize = float(abs(header["CDELT1"])) * 3600.0  # In arcsec
        imsize = int(header["NAXIS1"])  # Image size
        pix_radius = min(imsize, int((4 * 16 * 60) / cellsize))  # 4 solar radii
        if data.ndim == 4:
            data2d = data[0, 0, ...]
        elif data.ndim == 3:
            data2d = data[0, ...]
        else:
            data2d = data
        circular_mask = create_circular_mask_array(data2d, pix_radius)
        if use_gaussian:
            from scipy.optimize import curve_fit
            from scipy.ndimage import gaussian_filter

            data2d = gaussian_filter(data2d, sigma=3)
            max_pos = np.where(data2d == np.nanmax(data2d))
            y0, x0 = max_pos[0][0], max_pos[1][0]
            y_min = max(0, y0 - pix_radius)
            y_max = min(data2d.shape[0], y0 + pix_radius)
            x_min = max(0, x0 - pix_radius)
            x_max = min(data2d.shape[1], x0 + pix_radius)
            y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
            subdata = data2d[y_min:y_max, x_min:x_max]
            base_mean = np.nanmean(data2d[~circular_mask])
            gauss_sigma = int((sun_dia / 2) * 60.0 / cellsize)
            p0 = [np.nanmax(subdata), x0, y0, gauss_sigma, gauss_sigma, base_mean]
            popt, pcov = curve_fit(
                gaussian_2d, (x_grid, y_grid), subdata.ravel(), p0=p0, maxfev=5000
            )
            apparent_pix_ra = int(popt[1])
            apparent_pix_dec = int(popt[2])
        else:
            from scipy.ndimage import center_of_mass
            from skimage.morphology import remove_small_objects

            max_pos = np.where(data2d == np.nanmax(data2d))
            center_x, center_y = max_pos[1][0], max_pos[0][0]
            sun_rad_pix = 2 * sun_dia * 60 / cellsize  # 2 solar radii
            masked_array = create_circular_mask_array(
                data2d, sun_rad_pix, center_x=center_x, center_y=center_y
            )
            masked_data2d = copy.deepcopy(data2d)
            masked_data2d[masked_array] = np.nan
            rms = np.nanstd(masked_data2d)
            mask = data2d < sigma * rms
            mask_clean = remove_small_objects(~mask, min_size=100)
            data2d[~mask_clean] = False
            data2d[mask_clean] = True
            apparent_pix_dec, apparent_pix_ra = center_of_mass(data2d)
            apparent_pix_dec = int(apparent_pix_dec)
            apparent_pix_ra = int(apparent_pix_ra)
        w = WCS(imagename).celestial
        result = w.array_index_to_world(apparent_pix_dec, apparent_pix_ra)
        x_cen = result.ra.deg
        y_cen = result.dec.deg
        ra = float(x_cen)
        dec = float(y_cen)
        return 0, ra, dec, apparent_pix_ra, apparent_pix_dec
    except Exception:
        traceback.print_exc()
        return 1, None, None, None, None


def shift_solarcenter_to_imagecenter(
    imagename,
    sigma=10,
    apparent_ra=None,
    apparent_dec=None,
    use_gaussian=False,
    overwrite=True,
):
    """
    Function to shift solar center to image center

    Parameters
    ----------
    imagename : str
        Name of the image
    sigma : float, optional
        Sigma threshold for masking solar disk
    apparent_ra : float, optional
        Apparent solar disk RA in degree
    apparent_dec : float, optional
        Apparent solar disk dec in degree
    use_gaussian : bool, optional
        Use gaussian fitting or not
    overwrite : bool, optional
        Overwrite existing image or not

    Returns
    -------
    int
        Success code 0: Successfully shifted, 1: Not shifted
    str
        Output image name
    """
    if apparent_ra is None or apparent_dec is None:
        msg, apparent_ra, apparent_dec, _, _ = cal_apparent_solarcenter(
            imagename, sigma=sigma, use_gaussian=use_gaussian
        )
        if msg != 0:
            print("Error in estimating apparent solar center.")
            return 1, ""
    try:
        w = WCS(imagename).celestial
        coord = SkyCoord(ra=apparent_ra * u.deg, dec=apparent_dec * u.deg, frame="icrs")
        apparent_pix_dec, apparent_pix_ra = w.world_to_array_index(coord)
        data = fits.getdata(imagename)
        header = fits.getheader(imagename)
        if data.ndim == 4:
            ny, nx = data[0, 0, ...].shape
        elif data.ndim == 3:
            ny, nx = data[0, ...].shape
        else:
            ny, nx = data.shape
        center_ra = nx // 2
        center_dec = ny // 2
        offset_ra = int(center_ra - apparent_pix_ra)
        offset_dec = int(center_dec - apparent_pix_dec)
        try:
            time = header["DATE-OBS"]
            astro_time = Time(time, scale="utc")
            sun_coords = get_sun(astro_time)
            header["CRVAL1"] = sun_coords.ra.deg
            header["CRVAL2"] = sun_coords.dec.deg
        except Exception:
            pass
        new_data = np.roll(np.roll(data, offset_dec, axis=-2), offset_ra, axis=-1)
        if overwrite:
            outfile = imagename
            fits.writeto(imagename, data=new_data, header=header, overwrite=True)
        else:
            outfile = imagename.split(".fits")[0] + "_centered.fits"
            fits.writeto(
                outfile,
                data=new_data,
                header=header,
                overwrite=True,
            )
        msg = 0
    except Exception:
        msg = 1
        outfile = imagename
        traceback.print_exc()
    finally:
        return msg, outfile


def interpolate_apparent_solar_center(
    sun_radeg,
    sun_decdeg,
    target_freq,
    freqlist=[],
    apparent_radeg_list=[],
    apparent_decdeg_list=[],
):
    """
    Estimate apparent solar center at a target frequency assuming
    ionospheric refraction (shift ∝ ν^-2).

    Parameters
    ----------
    sun_radeg : float
        Sun true RA in degree
    sun_decdeg : float
        Sun true DEC in degree
    target_freq : float
        Target frequency (same units as freqlist)
    freqlist : list
        Frequencies
    apparent_radeg_list : list
        Apparent RA at each frequency (deg)
    apparent_decdeg_list : list
        Apparent DEC at each frequency (deg)

    Returns
    -------
    float
        Apparent RA at target frequency (deg)
    float
        Apparent DEC at target frequency (deg)
    """
    if (
        len(freqlist) == 0
        or len(apparent_radeg_list) == 0
        or len(apparent_decdeg_list) == 0
        or len(freqlist) != len(apparent_radeg_list)
        or len(freqlist) != len(apparent_decdeg_list)
    ):
        print("Please provide matching frequency and apparent position lists.")
        return None, None

    freqlist = np.asarray(freqlist, dtype=float)
    apparent_radeg_list = np.asarray(apparent_radeg_list, dtype=float)
    apparent_decdeg_list = np.asarray(apparent_decdeg_list, dtype=float)

    # Measured shifts from true solar position
    dra = apparent_radeg_list - sun_radeg
    ddec = apparent_decdeg_list - sun_decdeg

    # Fit A in shift = A / nu^2
    A_ra = np.mean(dra * freqlist**2)
    A_dec = np.mean(ddec * freqlist**2)

    # Optional: reject obvious outliers if enough frequencies are available
    if len(freqlist) >= 3:
        sigma_ra = np.std(dra * freqlist**2)
        sigma_dec = np.std(ddec * freqlist**2)

        good = (np.abs(dra * freqlist**2 - A_ra) < 3 * sigma_ra) & (
            np.abs(ddec * freqlist**2 - A_dec) < 3 * sigma_dec
        )

        if np.sum(good) >= 2:
            A_ra = np.mean((dra * freqlist**2)[good])
            A_dec = np.mean((ddec * freqlist**2)[good])

    # Predict shift at target frequency
    dra_target = A_ra / target_freq**2
    ddec_target = A_dec / target_freq**2

    apparent_radeg_target = sun_radeg + dra_target
    apparent_decdeg_target = sun_decdeg + ddec_target

    return apparent_radeg_target, apparent_decdeg_target


def correct_solar_sidereal_motion(msname="", ncpu=1, verbose=False):
    """
    Correct sodereal motion of the Sun

    Parameters
    ----------
    msname : str
        Name of the measurement set
    ncpu : int, optional
        Number of CPU threads to use

    Returns
    -------
    int
        Success message
    """
    print(f"Correcting sidereal motion for ms: {msname}\n")
    if not os.path.exists(msname + "/.sidereal_cor"):
        msg = run_solar_sidereal_cor(
            msname=msname, ncpu=ncpu, container_name="paircarswsclean", verbose=verbose
        )
        if msg != 0:
            print("Sidereal motion correction is not successful.")
        else:
            os.system("touch " + msname + "/.sidereal_cor")
        return msg
    else:
        print(f"Sidereal motion correction is already done for ms: {msname}")
        return 0
