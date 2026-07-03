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
)
from astropy.io import fits
from astropy.wcs import WCS
from casatools import msmetadata
from .basic_utils import get_datadir, mjdsec_to_timestamp, angular_separation_equatorial
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


def cal_solar_phaseshift(imagename, sigma=10, use_gaussian=False):
    """
    Calculate the difference between solar center and phase center of the image

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
    float
        RA of true solarcenter in degree
    float
        DEC of true solarcenter in degree
    int
        Apparent RA pixel of solarcenter
    int
        Apparent DEC pixel of solarcenter
    float
        Shift size in arcseconds
    """

    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
        x, y = xy
        g = offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
        )
        return g.ravel()

    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    obstime = header["DATE-OBS"]
    if header["CTYPE3"] == "FREQ":
        freqMHz = float(header["CRVAL3"]) / 10**6  # In MHz
        sun_dia = calc_sun_dia(freqMHz)  # In arcmin
    elif header["CTYPE4"] == "FREQ":
        freqMHz = float(header["CRVAL4"]) / 10**6  # In MHz
        sun_dia = calc_sun_dia(freqMHz)  # In arcmin
    else:
        sun_dia = 32  # In arcmin
    (
        _,
        _,
        _,
        sun_radeg,
        sun_decdeg,
    ) = radec_sun_at_time(obstime)
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
        max_pos = np.where(data2d == np.nanmax(data2d))
        center_x, center_y = max_pos[1][0], max_pos[0][0]
        sun_rad_pix = 2*sun_dia*60/cellsize # 2 solar radii
        masked_array=create_circular_mask_array(data2d,sun_rad_pix,center_x=center_x,center_y=center_y)
        masked_data2d = copy.deepcopy(data2d)
        masked_data2d[masked_array]=np.nan
        rms = np.nanstd(masked_data2d)
        mask = data2d<sigma*rms
        data2d[mask]=False
        data2d[~mask]=True
        apparent_pix_dec, apparent_pix_ra = center_of_mass(data2d)
    try:
        w = WCS(imagename).celestial
        result = w.array_index_to_world(apparent_pix_dec, apparent_pix_ra)
        x_cen = result.ra.deg
        y_cen = result.dec.deg
        ra = float(x_cen)
        dec = float(y_cen)
        seperation_deg = angular_separation_equatorial(ra, dec, sun_radeg, sun_decdeg)
        seperation_arcsec = seperation_deg * 3600.0
        return (
            0,
            ra,
            dec,
            sun_radeg,
            sun_decdeg,
            apparent_pix_ra,
            apparent_pix_dec,
            seperation_arcsec,
        )
    except Exception:
        traceback.print_exc()
        return 1, sun_radeg, sun_decdeg, sun_radeg, sun_decdeg, 0, 0, 0


def shift_solarcenter(
    imagename,
    sigma=10,
    sun_radeg=None,
    sun_decdeg=None,
    apparent_pix_ra=None,
    apparent_pix_dec=None,
    use_gaussian=False,
    overwrite=True,
):
    """
    Function to shift solar center to image phase center

    Parameters
    ----------
    imagename : str
        Name of the image
    sigma : float, optional
        Sigma threshold for masking solar disk
    sun_radeg : float, optional
        Sun RA in degree
    sun_decdeg : float, optional
        Sun DEC in degree
    apparent_pix_ra :int, optional
        Apparent solar center pixel in RA
    apparent_pix_dec : int, optional
        Apparent solar center pixel in DEC
    use_gaussian : bool, optional
        Use gaussian fitting or not
    overwrite : bool, optional
        Overwrite existing image or not

    Returns
    -------
    int
        Success code 0: Successfully shifted, 1: Shifting is not required, 2: Error in shifting
    str
        Output image name
    bool
        Shifted or not
    int
        Maximum pixel offset
    """
    if (
        sun_radeg is None
        or sun_decdeg is None
        or apparent_pix_ra is None
        or apparent_pix_dec is None
    ):
        (
            msg,
            ra,
            dec,
            sun_radeg,
            sun_decdeg,
            apparent_pix_ra,
            apparent_pix_dec,
            seperation_arcsec,
        ) = cal_solar_phaseshift(imagename, sigma=sigma, use_gaussian=use_gaussian)
    shifted = False
    r_offset = 0
    try:
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
        r_offset = max(offset_ra, offset_dec)
        if abs(offset_ra) > 0 or abs(offset_dec) > 0:
            header["CRVAL1"] = float(sun_radeg)
            header["CRVAL2"] = float(sun_decdeg)
            header["CRPIX1"] = float(center_ra + 1)
            header["CRPIX2"] = float(center_dec + 1)
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
            shifted = True
            msg = 0
        else:
            outfile = imagename
            msg = 1
    except Exception:
        msg = 2
        outfile = imagename
        traceback.print_exc()
    finally:
        return msg, outfile, shifted, r_offset


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
