import numpy as np
import numexpr as ne
import traceback
import glob
import os
import subprocess
from casatools import msmetadata
from astropy.io import fits
from functools import partial
from .basic_utils import suppress_output, ra_dec_to_hms_dms, mjdsec_to_timestamp
from .resource_utils import limit_threads
from .flagging import do_flag_backup, flag_quartical_table, get_chans_flag
from .solarflagger import flagger
from .calibration import (
    fluxcal_caltable,
    uvrange_casa_to_quartical,
    quartical_matrix_normalize,
    get_cal_flag_info,
)
from .imaging import (
    calc_sun_dia,
    get_optimal_image_interval,
    calc_multiscale_scales,
    get_multiscale_bias,
)
from .image_utils import (
    create_circular_mask,
    create_circular_mask_array,
    calc_dyn_range,
    generate_tb_map,
    make_timeavg_image,
    make_stokes_wsclean_imagecube,
)
from .udocker_utils import run_wsclean, run_quartical
from .sunpos_utils import cal_solar_phaseshift, shift_solarcenter


def cal_crossphase(imagename):
    """
    Function to calculate Stokes U, V leakage through correlation analysis

    Parameters
    ----------
    imagename : str
        FITS image

    Returns
    -------
    float
        Cross hand phase
    """
    data = fits.getdata(imagename)
    u_data = data[2, 0, ...].astype(np.float64)
    v_data = data[3, 0, ...].astype(np.float64)
    max_pos = np.where(np.abs(v_data) == np.nanmax(np.abs(v_data)))
    peak_v = v_data[max_pos][0]
    crossphase_list = np.arange(-180, 180, 1)
    psi = np.deg2rad(crossphase_list)
    if u_data.size > 1 and v_data.size > 1:
        cospsi = np.cos(psi)[:, None, None]
        sinpsi = np.sin(psi)[:, None, None]
        u = u_data[None, ...]
        v = v_data[None, ...]
        # rotation using numexpr
        new_u = ne.evaluate("cospsi*u + sinpsi*v")
        new_v = ne.evaluate("-sinpsi*u + cospsi*v")
        # compute correlation coefficient for each psi
        cc_list = []
        for i in range(len(crossphase_list)):
            cc = abs(np.corrcoef(new_u[i].ravel(), new_v[i].ravel())[1, 0])
            cc_list.append(cc)
        cc_list = np.array(cc_list)
    pos = np.argsort(cc_list)
    cross_phases = crossphase_list[pos[0:4]]
    psi = np.deg2rad(cross_phases)
    cospsi = np.cos(psi)[:, None, None]
    sinpsi = np.sin(psi)[:, None, None]
    u = u_data[None, ...]
    v = v_data[None, ...]
    new_u = ne.evaluate("cospsi*u + sinpsi*v")
    new_v = ne.evaluate("-sinpsi*u + cospsi*v")
    x = np.nanmax(np.abs(new_u), axis=(-2, -1))
    pos = np.argsort(x)
    for i in pos[0:2]:
        peak_new_v = new_v[i][max_pos]
        if np.sign(peak_new_v) == np.sign(peak_v):
            cross_phase = cross_phases[i]
            return cross_phase


def do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=1):
    """
    Perform uv-sub flags

    Parameters
    ----------
    msname : str
        Measurement set
    threshold_list: list, optional
        Threshold list
    ncpu: int, optional
        Number of CPU threads to use
    """
    for threshold in threshold_list:
        count = 0
        while count < 5:
            result, n_final_flagged, n_additional_flagged = flagger(
                msname,
                "residual",
                threshold=threshold,
                num_processes=ncpu,
                flagbackup=False,
            )
            if n_additional_flagged == 0:
                break
            else:
                count += 1


def determine_disk_visibility(msname,chan=-1):
    """
    Determine whether solar disk is visible or not

    Parameters
    ----------
    msname : str
        Measurement set
    chan : int, optional
        Channel number to use
        
    Returns
    -------
    numpy.array
        Channel list where disk may not be detected
    numpy.array
        Timestamp list where disk may not be detected
    numpy.array
        Timestamps where disk is detected at least in one channel
    """
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from casatools import ms as casamstool, table
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.nchan(0)
    msmd.close()
    wavelength = (3 * 10**8) / freq
    tb = table()
    tb.open(msname)
    colnames = tb.colnames()
    tb.close()
    if "CORRECTED_DATA" in colnames:
        datacolumn = "corrected"
    else:
        datacolumn = "data"
    mstool = casamstool()
    uvdist = 150.0 * wavelength
    mstool.open(msname)
    if chan>0:
        mstool.selectchannel(nchan=1,start=chan, width=1)
        print(f"Using channel: {chan}")
    print(f"UV range: {uvdist-10.0}, {uvdist+10.0}")
    selection_ok = mstool.select({"uvdist": [uvdist - 10.0, uvdist + 10.0]})
    if not selection_ok:
        mstool.close()
        mstool.open(msname)
        if chan>0:
            mstool.selectchannel(nchan=1,start=chan, width=1)
        mstool.select({"uvdist": [uvdist - 50.0, uvdist + 50.0]})
    mstool.selectpolarization("I")
    if datacolumn == "corrected":
        data_first_lobe = np.abs(mstool.getdata("CORRECTED_DATA", ifraxis=True)["corrected_data"])
    else:
        data_first_lobe = np.abs(mstool.getdata("DATA", ifraxis=True)["data"])
    data_first_lobe_flag = mstool.getdata("FLAG", ifraxis=True)["flag"]
    mstool.close()
    data_first_lobe_flag = np.any(data_first_lobe_flag,axis=0)
    data_first_lobe[0,...][data_first_lobe_flag]=np.nan
    data_first_lobe = np.nanmedian(data_first_lobe,axis=2)
    mstool.open(msname)
    mstool.selectpolarization("I")
    if chan>0:
        mstool.selectchannel(nchan=1,start=chan, width=1)
    mstool.select({"uvdist": [0.0,0.0]})
    if datacolumn == "corrected":
        data_autocorr = np.abs(mstool.getdata("CORRECTED_DATA", ifraxis=True)["corrected_data"])
    else:
        data_autocorr = np.abs(mstool.getdata("DATA", ifraxis=True)["data"])
    data_autocorr_flag = mstool.getdata("FLAG", ifraxis=True)["flag"]
    data_autocorr_flag = np.any(data_autocorr_flag,axis=0)
    mstool.close()
    data_autocorr[0,...][data_autocorr_flag]=np.nan
    data_autocorr = np.nanmedian(data_autocorr,axis=2)
    r_I = data_first_lobe[0, ...]/data_autocorr[0,...]
    detected = r_I < 0.1
    n_detected_per_time = np.nansum(detected, axis=0)
    detected_timestamps = np.where(n_detected_per_time > 0)[0]
    pos = np.where(r_I >= 0.1)
    if len(pos) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), detected_timestamps
    elif len(pos) == 1:
        if chan>1:
            chans = np.array([chan])
        else:
            chans = pos[0]
        timestamps = np.zeros_like(chans)
        return chans, timestamps, detected_timestamps
    else:
        if chan>1:
            chans = np.array([chan])
        else:
            chans = pos[0]
        timestamps = pos[1]
        return chans, timestamps, detected_timestamps


def flag_non_disk(msname):
    """
    Flag spectro-temporal blocks when solar disk is not visible

    Parameters
    ----------
    msname : str
        Measurement set
    """
    from casatasks import flagdata

    try:
        chans, timestamps, detected_timestamps = determine_disk_visibility(msname)
        if len(detected_timestamps) == 0:
            return 1
        else:
            msmd = msmetadata()
            msmd.open(msname)
            times = msmd.timesforspws(0)
            msmd.close()
            for c, t in zip(chans, timestamps):
                spw = f"0:{c}"
                timerange = f"{mjdsec_to_timestamp(times[t], str_format=1)}"
                try:
                    flagdata(
                        vis=msname,
                        mode="manual",
                        spw=spw,
                        timerange=timerange,
                        flagbackup=False,
                    )
                except Exception:
                    continue
            unflag_chans, flag_chans = get_chans_flag(msname)
            if len(unflag_chans) == 0:
                return 1
            else:
                return 0
    except Exception:
        traceback.print_exc()
        return 1


def get_quiet_sun_flux(freq):
    """
    Get quiet Sun flux density in Jy.

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Flux density in Jy
    """
    p = np.poly1d([-1.93715165e-06, 7.84627718e-04, -3.15744433e-02, 2.32834400e-01])
    flux = p(freq) * 10**4  # Polynomial return in SFU
    return flux


def make_qs_model(msname, clname="quiet_sun.cl"):
    """
    Make CASA component list of quiet Sun model

    Parameters
    ----------
    msname : str
        Name of the measurement set
    clname : str, optional
        Name of the component list

    Returns
    -------
    str
        Name of the component list file
    """
    from casatools import componentlist

    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0, unit="MHz")
    phasecenter = msmd.phasecenter(0)
    msmd.close()

    radeg = np.rad2deg(phasecenter["m0"]["value"])
    decdeg = np.rad2deg(phasecenter["m1"]["value"])
    rahms, decdms = ra_dec_to_hms_dms(radeg, decdeg)
    radec_str = f"J2000 {rahms} {decdms}"
    sun_size = calc_sun_dia(freq)  # In arcmin
    QS_flux = get_quiet_sun_flux(freq)  # In Jy

    # Make sure the component list does not already exist. The tool will complain otherwise.
    os.system("rm -rf " + clname)
    cl = componentlist()
    cl.addcomponent(
        dir=radec_str,
        flux=QS_flux,  # For a gaussian, this is the integrated area.
        fluxunit="Jy",
        freq=f"{freq}MHz",
        shape="gaussian",  ## Gaussian
        majoraxis=f"{sun_size}arcmin",
        minoraxis=f"{sun_size}arcmin",
        positionangle="0deg",
        spectrumtype="spectral index",
        index=0.0,
    )
    # Save the file
    cl.rename(filename=clname)
    cl.done()
    return clname


def quiet_sun_selfcal(msname, logger, selfcaldir, refant="1", solint="60s"):
    """
    Perform quiet Sun Gaussian model based self-calibration

    Parameters
    ----------
    msname : str
        Measurement set
    logger : str
        Python logger
    selfcaldir : str
        Self-calibration directory
    refant : str, optional
        Reference antenna
    solint : str, optional
        Solution interval

    Returns
    -------
    int
        Success message
    str
        Caltable name
    """
    from casatasks import ft, delmod, gaincal, applycal, flagmanager

    prefix = (
        selfcaldir + "/" + os.path.basename(msname).split(".ms")[0] + "_selfcal_present"
    )
    bpass_caltable = prefix.replace("present", f"{0}") + ".gcal"
    if os.path.exists(bpass_caltable):
        os.system("rm -rf " + bpass_caltable)
    do_flag_backup(msname, flagtype="qs_selfcal")

    try:
        result = flag_non_disk(msname)
        if result != 0:
            logger.warning("Could not flag non-disk time properly.")
        ###################################
        # Import simulated QS model
        ###################################
        qs_model = make_qs_model(
            msname, clname=f"{os.path.basename(msname).split('.ms')[0]}_qs.cl"
        )
        delmod(vis=msname, otf=True, scr=True)
        ft(vis=msname, complist=qs_model, usescratch=True)
        os.system(f"rm -rf {qs_model}")

        #####################
        # Perform calibration
        #####################
        logger.info(
            f"gaincal(vis='{msname}',caltable='{bpass_caltable}',uvrange='<100lambda',refant='{refant}',solint='{solint}',minsnr=3,calmode='p')\n"
        )
        with suppress_output():
            gaincal(
                vis=msname,
                caltable=bpass_caltable,
                uvrange="<100lambda",
                refant=refant,
                minsnr=3,
                solint=f"{solint}",
                solnorm=True,
                calmode="p",
            )
        if not os.path.exists(bpass_caltable):
            logger.info("No gain solutions are found.\n")
            msg = 1
            bpass_caltable = ""
        else:
            ########################
            # Applying solutions
            ########################

            logger.info(
                f"applycal(vis={msname},gaintable=[{bpass_caltable}],interp=['linear'],applymode='calonly',calwt=[False])\n"
            )
            with suppress_output():
                applycal(
                    vis=msname,
                    gaintable=[bpass_caltable],
                    interp=["linear"],
                    applymode="calonly",
                    calwt=[False],
                )
            msg = 0
    except Exception:
        logger.exception(traceback.print_exc())
        msg = 2
        bpass_caltable = ""
    finally:
        with suppress_output():
            flagmanager(vis=msname, mode="restore", versionname="qs_selfcal_1")
            flagmanager(vis=msname, mode="delete", versionname="qs_selfcal_1")
        return msg, bpass_caltable


def check_valid_image(imagename):
    """
    Check whether the image is valid or not

    Parameters
    ----------
    imagename : str
        Image name

    Returns
    -------
    bool
        Whether valid image or not
    """
    data = fits.getdata(imagename)
    if np.nansum(data) == 0:
        return False
    else:
        return True


def single_image_update_phasecenter(
    wsclean_images,
    wsclean_models,
    image_cube,
    model_cube,
    cellsize,
    imsize,
    stokes,
    logger,
):
    """
    Update phase center of a single set of polarisation image

    Parameters
    ----------
    wsclean_images : list
        List of wsclean Stokes images
    wsclean_models : list
        List of wsclean Stokes models
    image_cube : str
        Stokes image cube name
    model_cube : str
        Stokes model cube name
    cellsize : float
        Pixel size in arcseconds
    imsize : int
        Image size
    stokes : str
        Stokes planes of image cube

    Returns
    -------
    int
        Success message
    bool
        Whether phase shift needed or not
    float
        Maximum offset in pixel
    """
    try:
        (
            msg,
            ra,
            dec,
            sun_radeg,
            sun_decdeg,
            apparent_pix_ra,
            apparent_pix_dec,
            seperation_arcsec,
        ) = cal_solar_phaseshift(image_cube)
        if msg != 0:
            return msg, False
        shift_needed = False
        r_offset_list = []
        logger.info(f"Shift {seperation_arcsec}arcsec for {image_cube}.")
        shift_func = partial(
            shift_solarcenter,
            sun_radeg=sun_radeg,
            sun_decdeg=sun_decdeg,
            apparent_pix_ra=apparent_pix_ra,
            apparent_pix_dec=apparent_pix_dec,
            overwrite=True,
        )
        msg, outfile, shifted, r_offset = shift_func(image_cube)
        shift_needed = bool(shift_needed + shifted)
        r_offset_list.append(r_offset)
        for imagename in wsclean_images:
            msg, outfile, shifted, r_offset = shift_func(imagename)
            shift_needed = bool(shift_needed + shifted)
            r_offset_list.append(r_offset)
        msg, outfile, shifted, r_offset = shift_func(model_cube)
        shift_needed = bool(shift_needed + shifted)
        r_offset_list.append(r_offset)
        for modelname in wsclean_models:
            msg, outfile, shifted, r_offset = shift_func(modelname)
            shift_needed = bool(shift_needed + shifted)
            r_offset_list.append(r_offset)
        r_offset = max(r_offset_list)
        return 0, shift_needed, r_offset
    except Exception:
        traceback.print_exc()
        return 1, False, 0


def correct_spectrosnap_phaseshift(
    image_dic,
    model_dic,
    cellsize,
    imsize,
    stokes,
    logger,
):
    """
    Correct spectrocopic snapshot images for phase shift

    Parameters
    ----------
    image_dic : dict
        Image dictionary
    model_dic : dict
        Model dictionary
    cellsize : float
        Pixel size in arcsecond
    imsize : int
        Image size
    stokes : str
        Stokes planes of image list
    logger : logger, optional
        Python logger

    Returns
    -------
    int
        Success message
    bool
        If shift needed for any image
    int
        Maximum pixel offset
    """
    shifted = False
    max_pixel_offset_list = []
    try:
        images = list(image_dic.keys())
        models = list(model_dic.keys())
        for i in range(len(images)):
            imagename = images[i]
            modelname = models[i]
            wsclean_images = image_dic[imagename]
            wsclean_models = model_dic[modelname]
            valid_image = check_valid_image(imagename)
            if valid_image:
                success_msg, shift_needed, max_pixel_offset = (
                    single_image_update_phasecenter(
                        wsclean_images,
                        wsclean_models,
                        imagename,
                        modelname,
                        cellsize,
                        imsize,
                        stokes,
                        logger,
                    )
                )
                max_pixel_offset_list.append(max_pixel_offset)
                if shift_needed:
                    shifted = True
        return 0, shifted, max(max_pixel_offset_list)
    except Exception:
        traceback.print_exc()
        return 1, shifted, 0


def calc_leakage(imagename, threshold=5, disc_size=50):
    """
    Calculate Stokes I to Q, U, V leakages

    Parameters
    ----------
    imagename : str
        Image name
    threshold : float
        Threshold to choose region with Stokes I detection
    disc_size : float
        Solar disc area in arcminute to mask for calculating rms
        N.B.: Chosen slightly larger to avoid any off-coronal emission from CMEs

    Returns
    -------
    float
        Stokes I to Q leakage
    float
        Stokes I to U leakage
    float
        Stokes I to V leakage
    float
        Stokes I to Q leakage error
    float
        Stokes I to U leakage error
    float
        Stokes I to V leakage error
    """
    valid_image = check_valid_image(imagename)
    if valid_image is False:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    tb_map = generate_tb_map(imagename)
    tb_data = fits.getdata(tb_map)[0, 0, ...] / 10**6  # in MK
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    pix_size = abs(header["CDELT1"]) * 3600.0  # In arcsec
    radius = int((disc_size * 60) / pix_size)
    i_data = data[0, 0, ...]
    q_data = data[1, 0, ...]
    u_data = data[2, 0, ...]
    v_data = data[3, 0, ...]
    #############################
    # Calculating image rms
    #############################
    mask = create_circular_mask_array(i_data, radius)
    i_rms = np.nanstd(i_data[~mask])
    i_thresh = threshold * i_rms
    ##############################################
    # Estimating regions for leakage calculation
    ##############################################
    pos = np.where((i_data < i_thresh) | (tb_data > 1.0))
    q_data[pos] = np.nan
    u_data[pos] = np.nan
    v_data[pos] = np.nan
    q_by_i = q_data / i_data
    u_by_i = u_data / i_data
    v_by_i = v_data / i_data
    q_by_i = q_by_i[~np.isnan(q_by_i)].flatten()
    u_by_i = u_by_i[~np.isnan(u_by_i)].flatten()
    v_by_i = v_by_i[~np.isnan(v_by_i)].flatten()

    #########################################
    # Estimating leakage and leakage errors
    #########################################
    q_leakage = round(np.nanmedian(q_by_i), 4)
    u_leakage = round(np.nanmedian(u_by_i), 4)
    v_leakage = round(np.nanmedian(v_by_i), 4)

    q_leakage_err = round(
        1.253 * np.nanmedian(abs(q_by_i - q_leakage)) / np.sqrt(q_by_i.size), 6
    )
    u_leakage_err = round(
        1.253 * np.nanmedian(abs(u_by_i - u_leakage)) / np.sqrt(u_by_i.size), 6
    )
    v_leakage_err = round(
        1.253 * np.nanmedian(abs(v_by_i - v_leakage)) / np.sqrt(v_by_i.size), 6
    )

    os.system(f"rm -rf {tb_map}")
    return q_leakage, u_leakage, v_leakage, q_leakage_err, u_leakage_err, v_leakage_err


def correct_image_leakage(
    imagename,
    modelname="",
    q_leakage=0.0,
    u_leakage=0.0,
    v_leakage=0.0,
    threshold=5,
    disc_size=50,
):
    """
    Correct leakages in image plane

    Parameters
    ----------
    imagename : str
        Image name
    modelname : str, optional
        Model name
    q_leakage : float, optional
        Q leakage
    u_leakage : float, optional
        U leakage
    v_leakage : float, optional
        V leakage
    threshold : float
        Threshold to choose region with Stokes I detection
    disc_size : float
        Solar disc area in arcminute to mask for calculating rms
        N.B.: Chosen slightly larger to avoid any off-coronal emission from CMEs

    Returns
    -------
    str
        Leakage corrected imagename
    str
        Leakage corrected modelname
    """
    #######################
    # Read image data
    #######################
    imagedata = fits.getdata(imagename)
    image_I = imagedata[0, 0, ...]
    image_Q = imagedata[1, 0, ...]
    image_U = imagedata[2, 0, ...]
    image_V = imagedata[3, 0, ...]

    if os.path.exists(modelname):
        correct_model = True
    else:
        correct_model = False
    if correct_model:
        ##########################
        # Read model data
        ##########################
        modeldata = fits.getdata(modelname)
        model_I = modeldata[0, 0, ...]
        model_Q = modeldata[1, 0, ...]
        model_U = modeldata[2, 0, ...]
        model_V = modeldata[3, 0, ...]
        modelheader = fits.getheader(modelname)

    ###################################
    # Creating mask
    ####################################
    imageheader = fits.getheader(imagename)

    pix_size = abs(imageheader["CDELT1"]) * 3600.0  # In arcsec
    radius = int((disc_size * 60) / pix_size)
    mask = create_circular_mask_array(image_I, radius)

    ####################################
    # Calculate rms
    ####################################
    q_rms = np.nanstd(image_Q[~mask])
    u_rms = np.nanstd(image_U[~mask])
    v_rms = np.nanstd(image_V[~mask])

    ###################################
    # Correcting images
    ###################################
    image_Q = image_Q - (q_leakage * image_I)
    image_U = image_U - (u_leakage * image_I)
    image_V = image_V - (v_leakage * image_I)
    posq = np.where(abs(image_Q) < threshold * q_rms)
    posu = np.where(abs(image_U) < threshold * u_rms)
    posv = np.where(abs(image_V) < threshold * v_rms)
    imagedata[1, 0, ...] = image_Q
    imagedata[2, 0, ...] = image_U
    imagedata[3, 0, ...] = image_V
    fits.writeto(
        imagename.split(".fits")[0] + "_leakagecor.fits",
        data=imagedata,
        header=imageheader,
        overwrite=True,
    )

    if correct_model:
        ####################################
        # Correcting model images
        ####################################
        model_Q = model_Q - (q_leakage * model_I)
        model_U = model_U - (u_leakage * model_I)
        model_V = model_V - (v_leakage * model_I)
        model_Q[posq] = 0.0
        model_U[posu] = 0.0
        model_V[posv] = 0.0
        modeldata[1, 0, ...] = model_Q
        modeldata[2, 0, ...] = model_U
        modeldata[3, 0, ...] = model_V
        modeldata[np.isinf(modeldata)] = 0.0
        modeldata[np.isnan(modeldata)] = 0.0
        fits.writeto(
            modelname.split(".fits")[0] + "_leakagecor.fits",
            data=modeldata,
            header=modelheader,
            overwrite=True,
        )

    if correct_model:
        return (
            imagename.split(".fits")[0] + "_leakagecor.fits",
            modelname.split(".fits")[0] + "_leakagecor.fits",
        )
    else:
        return (
            imagename.split(".fits")[0] + "_leakagecor.fits",
            None,
        )


def correct_pbcor_leakage(
    imagename,
    modelname,
    metafits,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    ncpu=1,
):
    """
    Perform primary beam and leakage correction

    Parameters
    ----------
    imagename : str
        Image name
    modelname : str
        Model image name
    metafits : str
        Metafits file
    pbcor : bool, optional
        Perform primary beam correction
    leakagecor : bool, optional
        Perform image based residual leakage correction
    pbuncor : bool, optional
        Undo primary beam correction
    ncpu : int, optional
        Number of CPU threads

    Returns
    -------
    str
        Final image
    str
        Final model
    list
        Leakage and leakage error list
    """
    ncpu = max(1, ncpu)
    leakage_info = []
    freq = fits.getheader(imagename)["CRVAL3"]
    pbfile = f"freq_{freq}_pb.npy"
    if pbcor is False:
        pbcor_image = imagename
        pbcor_model = modelname
    else:
        ####################################
        # Correcting image
        ####################################
        pbcor_image = imagename.split(".fits")[0] + "_pbcor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            imagename,
            metafits,
            pbcor_image,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
        ]
        if os.path.exists(pbfile) is False:
            pbcor_cmds.append("--save_pb")
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        #########################################
        # Correcting model
        #########################################
        pbcor_model = modelname.split(".fits")[0] + "_pbcor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            modelname,
            metafits,
            pbcor_model,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
        ]
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if leakagecor is False:
        leakagecor_image = pbcor_image
        leakagecor_model = pbcor_model
    else:
        ########################################
        # Estimating and correcting leakage
        ########################################
        (
            q_leakage,
            u_leakage,
            v_leakage,
            q_leakage_err,
            u_leakage_err,
            v_leakage_err,
        ) = calc_leakage(pbcor_image)
        leakage_info = [
            q_leakage,
            u_leakage,
            v_leakage,
            q_leakage_err,
            u_leakage_err,
            v_leakage_err,
        ]
        leakagecor_image, leakagecor_model = correct_image_leakage(
            pbcor_image,
            modelname=pbcor_model,
            q_leakage=q_leakage,
            u_leakage=u_leakage,
            v_leakage=v_leakage,
        )

    if pbuncor is False:
        final_image = leakagecor_image
        final_model = leakagecor_model
    else:
        ##########################################
        # Restore primary beam corrections
        ###########################################
        # For image
        ###############
        final_image = imagename.split(".fits")[0] + "_pbuncor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            leakagecor_image,
            metafits,
            final_image,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
            "--restore",
        ]
        if os.path.exists(pbfile) is False:
            pbcor_cmds.append("--save_pb")
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        #################
        # For model
        #################
        final_model = modelname.split(".fits")[0] + "_pbuncor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            leakagecor_model,
            metafits,
            final_model,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
            "--restore",
        ]
        if os.path.exists(pbfile) is False:
            pbcor_cmds.append("--save_pb")
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return final_image, final_model, leakage_info


def single_image_update_leakage(
    wsclean_images,
    wsclean_models,
    image_cube,
    model_cube,
    metafits,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    ncpu=-1,
):
    """
    Update leakage of a single set pf polarisation image

    Parameters
    ----------
    wsclean_images : list
        List of wsclean Stokes images
    wsclean_models : list
        List of wsclean Stokes models
    image_cube : str
        Stokes image cube name
    model_cube : str
        Stokes model cube name
    metafits : str
        Metafits file
    pbcor : bool, optional
        Perform primary beam correction or not
    leakagecor : bool, optional
        Perform leakage correction or not
    pbuncor : bool, optional
        Undo primary beam correction or not
    ncpu : int, optional
        NUmber of CPU threads to use

    Returns
    -------
    list
        Leakage informations
    """
    ncpu = max(1, ncpu)
    valid_image = check_valid_image(image_cube)
    if valid_image:
        cor_imagename, cor_modelname, leakage_info = correct_pbcor_leakage(
            image_cube,
            model_cube,
            metafits,
            pbcor=pbcor,
            leakagecor=leakagecor,
            pbuncor=pbuncor,
            ncpu=ncpu,
        )
        image_data = fits.getdata(cor_imagename)
        image_data[np.isinf(image_data)] = 0.0
        image_data[np.isnan(image_data)] = 0.0
        model_data = fits.getdata(cor_modelname)
        model_data[np.isinf(model_data)] = 0.0
        model_data[np.isnan(model_data)] = 0.0
        for i in range(len(wsclean_images)):
            spectro_image_header = fits.getheader(wsclean_images[i])
            spectro_image_data = fits.getdata(wsclean_images[i])
            spectro_image_data[0, 0, ...] = image_data[i, 0, ...]
            fits.writeto(
                wsclean_images[i],
                data=spectro_image_data,
                header=spectro_image_header,
                overwrite=True,
            )
            spectro_model_header = fits.getheader(wsclean_models[i])
            spectro_model_data = fits.getdata(wsclean_models[i])
            spectro_model_data[0, 0, ...] = model_data[i, 0, ...]
            fits.writeto(
                wsclean_models[i],
                data=spectro_model_data,
                header=spectro_model_header,
                overwrite=True,
            )
        return leakage_info
    else:
        return


def correct_spectrosnap_pbleak(
    image_dic,
    model_dic,
    metafits,
    logger=None,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    ncpu=-1,
):
    """
    Correct spectrocopic snapshot images for leakage

    Parameters
    ----------
    image_dic : dict
        Image dictionary
    model_dic : dict
        Model dictionary
    metafits : str
        Metafits file
    logger : logger, optional
        Python logger
    pbcor : bool, optional
        Perform primary beam correction
    leakagecor : bool, optional
        Leakage correction
    pbuncor : bool, optional
        Undo primary beam correction
    ncpu : int, optional
        Number of CPU threads to use

    Returns
    -------
    list
        Leakage information list
    """
    ncpu = max(1, ncpu)
    images = list(image_dic.keys())
    models = list(model_dic.keys())
    leakage_info_list = []
    for i in range(len(images)):
        imagename = images[i]
        modelname = models[i]
        fits.getheader(imagename)
        if "MFS" not in imagename:
            wsclean_images = image_dic[imagename]
            wsclean_models = model_dic[modelname]
            valid_image = check_valid_image(imagename)
            if valid_image:
                if logger is not None:
                    logger.info(f"Leakage correction for: {imagename}.")
                else:
                    print(f"Leakage correction for: {imagename}.")
                leakage_info = single_image_update_leakage(
                    wsclean_images,
                    wsclean_models,
                    imagename,
                    modelname,
                    metafits,
                    pbcor=pbcor,
                    leakagecor=leakagecor,
                    pbuncor=pbuncor,
                    ncpu=ncpu,
                )
                if leakage_info is not None:
                    leakage_info_list.append(leakage_info)
    os.system("rm -rf *_pbcor.fits *_leakagecor.fits *_pbuncor.fits *pb.npy")
    return leakage_info_list


def selfcal_round(
    msname,
    metafits,
    logger,
    selfcaldir,
    cellsize,
    imsize,
    round_number=0,
    uvrange="",
    minuv=0,
    calmode="ap",
    solint="60s",
    solnorm=True,
    refant="1",
    do_bandpass=True,
    applymode="calonly",
    threshold=3,
    weight="briggs",
    robust=0.0,
    use_previous_model=False,
    use_solar_mask=True,
    mask_radius=40,
    min_tol_factor=-1,
    calc_chunks=True,
    fluxscale_mwa=False,
    solar_attn=10,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    do_intensity_cal=False,
    do_polcal=False,
    solve_array_leakage=False,
    pol_solnorm=False,
    do_flag=False,
    restore_flag=True,
    ncpu=-1,
    mem=-1,
):
    """
    A single self-calibration round

    Parameters
    ----------
    msname : str
        Name of the measurement set
    metafits : str
        Metafits file
    logger : logger
        Python logger
    selfcaldir : str
        Self-calibration directory
    cellsize : float
        Cellsize in arcsec
    imsize :  int
        Image pixel size
    round_number : int, optional
        Selfcal iteration number
    uvrange : float, optional
       UV range for calibration
    minuv : float, optional
        Minimum uv in lambda
    calmode : str, optional
        Calibration mode ('p' or 'ap')
    solint : str, optional
        Solution intervals
    solnorm : bool, optional
        Solution normalisation
    refant : str, optional
        Reference antenna
    do_bandpass: bool, optional
        Perform bandpass calibration
    applymode : str, optional
        Solution apply mode (calonly or calflag)
    threshold : float, optional
        Imaging and auto-masking threshold
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting
    use_previous_model : bool, optional
        Use previous model
    use_solar_mask : bool, optional
        Use solar disk mask or not
    mask_radius : float, optional
        Mask radius in arcminute
    min_tol_factor : float, optional
        Minimum tolerance factor
    calc_chunks : bool, optional
        Whether calculate spectro-temporal chunks or not
    fluxscale_mwa : bool, optional
        Fluxscale caltable using reference bandpass
    solar_attn : float, optional
        Solar attenuation in dB (only used if fluxscale_mwa is True)
    pbcor : bool, optional
        Primary beam correction
    leakagecor : bool, optional
        Leakage correction
    pbuncor : bool, optional
        Undo primary beam correction
    do_intensity_cal : bool, optional
        Perform intensity self-calibration
    do_polcal : bool, optional
        Perform polarisation calibration or not
    solve_array_leakage : bool, optional
        Perform a single leakage correction over the entire array
    pol_solnorm : bool, optional
        Normalise quartical solutions or not
    do_flag : bool, optional
        Perform UVsub flagging
    restore_flag : bool, optional
        Restore last round flags or not
    ncpu : int, optional
        Number of CPUs to use in WSClean
    mem : float, optional
        Memory usage limit in WSClean

    Returns
    -------
    int
        Success message
    list
        Caltable name list
    float
        RMS based dynamic range
    float
        RMS of the image
    str
        Image name
    str
        Model image name
    str
        Residual image name
    list
        Leakage informations [Q_leakage, U_leakage, V_leakage, Q_leakage_error, U_leakage_error, V_leakage_error]
    int
        Maximum pixel offset
    """
    ncpu = max(1, ncpu)
    mem = max(1, mem)

    limit_threads(n_threads=ncpu)
    from casatasks import gaincal, bandpass, applycal, flagdata, delmod, flagmanager
    from casatools import table

    cwd = os.getcwd()
    msname = msname.rstrip("/")
    msname = os.path.abspath(msname)
    os.chdir(selfcaldir)

    if minuv > 0:
        if uvrange == "" or uvrange.startswith(">"):
            uvrange = f">{minuv}lambda"
        elif uvrange.startswith("<"):
            maxuv = uvrange.split("lambda")[0].split("<")[1]
            uvrange = f"{minuv}~{maxuv}lambda"
        elif "~" in uvrange:
            maxuv = uvrange.split("lambda")[0].split("~")[-1]
            uvrange = f"{minuv}~{maxuv}lambda"

    if not use_previous_model:
        delmod(vis=msname, otf=True, scr=True)
    prefix = (
        selfcaldir + "/" + os.path.basename(msname).split(".ms")[0] + "_selfcal_present"
    )
    applycal_gaintable = []
    interp = []
    leakage_info_list = []
    try:
        ####################################
        # Determining ms metadata
        ####################################
        msmd = msmetadata()
        msmd.open(msname)

        freq = msmd.meanfreq(0, unit="MHz")
        num_chan = msmd.nchan(0)
        freqres = msmd.chanres(0, unit="MHz")[0]
        bw = num_chan * freqres
        msmd.close()

        #####################################
        # Get column names
        #####################################
        tb = table()
        tb.open(msname)
        tb.colnames()
        tb.close()

        ######################################
        # Determine spectro-temporal chunking
        ######################################
        if calc_chunks:
            header = fits.getheader(metafits)
            mode = header["MODE"]
            if "MWAX" in mode:
                flag_central_chan = False
            else:
                flag_central_chan = True
            if calmode == "ap" or do_polcal:
                nchans = max(1, int(bw / 0.32))  # Fixed to 320 kHz
            else:
                nchans = 1
            if min_tol_factor <= 0:
                min_tol_factor = 1.0  # In percentage
            msmd = msmetadata()
            msmd.open(msname)
            times = msmd.timesforspws(0)
            msmd.close()
            diff = np.diff(times)
            change_idx = np.where(np.diff(diff) != 0)[0]
            max_ntime = int(len(change_idx) / 2) + 1
            nintervals, nchans_variablity = get_optimal_image_interval(
                msname,
                temporal_tol_factor=float(min_tol_factor / 100.0),
                spectral_tol_factor=float(min_tol_factor / 100.0),
                flag_central_chan=flag_central_chan,
                max_ntime=max_ntime,
            )
            nchans = min(nchans, nchans_variablity)
        else:
            nchans = 1
            nintervals = 1
            max_ntime = 1

        os.system(f"rm -rf {prefix}*image.fits {prefix}*residual.fits")

        if weight == "briggs":
            weight += " " + str(robust)

        wsclean_args = [
            "-quiet",
            f"-scale {cellsize}asec",
            f"-size {imsize} {imsize}",
            "-no-dirty",
            "-gridder wgridder",
            f"-weight {weight}",
            "-niter 10000",
            "-mgain 0.85",
            "-nmiter 5",
            "-gain 0.1",
            f"-minuv-l {minuv}",
            f"-j {ncpu}",
            f"-abs-mem {mem}",
            f"-auto-mask {threshold + 0.1}",
            f"-auto-threshold {threshold}",
        ]
        if do_intensity_cal:
            wsclean_args.append("-pol I")
            pol = "I"
            if calmode == "p":
                wsclean_args.append("-no-negative")
        else:
            wsclean_args.append("-pol IQUV")
            pol = "IQUV"

        ngrid = max(1, int(ncpu / 2))
        if ngrid > 1:
            wsclean_args.append(f"-parallel-gridding {ngrid}")

        ################################################
        # Creating and using solar mask
        ################################################
        fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
        if not os.path.exists(fits_mask):
            logger.info(f"Creating solar mask of size: {mask_radius} arcmin.\n")
            fits_mask = create_circular_mask(
                msname, cellsize, imsize, mask_radius=mask_radius
            )
        if fits_mask is not None and os.path.exists(fits_mask) and use_solar_mask:
            wsclean_args.append(f"-fits-mask {fits_mask}")

        ######################################
        # Determining multiscale parameter
        ######################################
        sun_dia = calc_sun_dia(freq)  # Sun diameter in arcmin
        sun_rad = sun_dia / 2.0
        multiscale_scales = calc_multiscale_scales(msname, 3, max_scale=sun_rad)
        scale_bias = round(get_multiscale_bias(freq), 2)
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
        wsclean_args.append(
            "-multiscale-scales " + ",".join([str(s) for s in multiscale_scales])
        )
        wsclean_args.append(f"-multiscale-scale-bias {scale_bias}")
        if imsize >= 1024 and 4 * max(multiscale_scales) < 512:
            wsclean_args.append("-parallel-deconvolution 512")

        #####################################
        # Temporal imaging configuration
        #####################################
        logger.info(f"Temporal chunks: {nintervals}, Spectral chunks: {nchans}.")
        if nintervals > 1:
            wsclean_args.append(f"-intervals-out {nintervals}")
        if nchans > 1:
            wsclean_args.append(f"-channels-out {nchans}")
            wsclean_args.append("-gap-channel-division")
            wsclean_args.append("-no-mf-weighting")

        #####################################
        # Image naming
        #####################################
        wsclean_args.append(f"-name {prefix}")
        if use_previous_model and do_polcal is False:
            previous_models = glob.glob(f"{prefix}*model.fits")
            total_models_expected = nintervals * nchans
            if len(previous_models) == total_models_expected:
                wsclean_args.append("-continue")
            else:
                os.system(f"rm -rf {prefix}*")

        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        logger.info(f"\nWSClean command: {wsclean_cmd}\n")
        msg = run_wsclean(wsclean_cmd, "paircarswsclean", verbose=False)
        if msg != 0:
            logger.error("Imaging is not successful.\n")
            return 1, applycal_gaintable, 0, 0, "", "", "", [], 0

        #######################################
        # Making stokes cube
        #######################################
        pollist = list(pol)
        wsclean_images_dic = {}
        wsclean_models_dic = {}
        wsclean_residuals_dic = {}
        for suffix in ["image", "model", "residual"]:
            stokeslist = []
            for p in pollist:
                if pollist == ["I"]:
                    stokeslist.append(
                        sorted(glob.glob(prefix + "*" + f"-{suffix}.fits"))
                    )
                else:
                    stokeslist.append(
                        sorted(glob.glob(prefix + "*" + p + f"-{suffix}.fits"))
                    )
            for i in range(len(stokeslist[0])):
                wsclean_images = sorted([stokeslist[k][i] for k in range(len(pollist))])
                image_prefix = (
                    selfcaldir
                    + "/"
                    + os.path.basename(wsclean_images[0])
                    .split(f"-{suffix}")[0]
                    .split("-I")[0]
                )
                image_cube = make_stokes_wsclean_imagecube(
                    wsclean_images,
                    image_prefix + f"-IQUV-{suffix}.fits",
                    keep_wsclean_images=True,
                )
                if suffix == "image":
                    wsclean_images_dic[image_cube] = wsclean_images
                elif suffix == "model":
                    wsclean_models_dic[image_cube] = wsclean_images
                elif suffix == "residual":
                    wsclean_residuals_dic[image_cube] = wsclean_images

        #########################################
        # Shifting solar center to phase center
        #########################################
        logger.info("Shifting images...")
        shifting_msg, shifted, max_pixel_offset = correct_spectrosnap_phaseshift(
            wsclean_images_dic,
            wsclean_models_dic,
            cellsize,
            imsize,
            pol,
            logger,
        )
        if shifting_msg != 0:
            logger.warning("Error occured in phase shift correction.")
        else:
            logger.info("Image shift is done.")

        if do_polcal:
            ################################
            # Leakage correction
            ################################
            if pbcor is True or leakagecor is True or pbuncor is True:
                leakage_info_list = correct_spectrosnap_pbleak(
                    wsclean_images_dic,
                    wsclean_models_dic,
                    metafits,
                    logger=logger,
                    pbcor=pbcor,
                    leakagecor=leakagecor,
                    pbuncor=pbuncor,
                    ncpu=ncpu,
                )

        ####################################
        # Predict models
        ####################################
        if do_polcal or shifted:
            prediction_failed = False
            delmod(vis=msname, otf=True, scr=True)
            wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " -predict " + msname
            logger.info(f"\nWSClean command: {wsclean_cmd}\n")
            prediction_msg = run_wsclean(wsclean_cmd, "paircarswsclean", verbose=False)
            if prediction_msg != 0:
                prediction_failed = True

        if do_polcal:
            #######################################
            # Remove chunk files
            #######################################
            images = list(wsclean_images_dic.keys())
            models = list(wsclean_models_dic.keys())
            residuals = list(wsclean_residuals_dic.keys())
            for i in range(len(images)):
                imagename = images[i]
                modelname = models[i]
                residualname = residuals[i]
                wsclean_images = wsclean_images_dic[imagename]
                wsclean_models = wsclean_models_dic[modelname]
                wsclean_residuals = wsclean_residuals_dic[residualname]
                for img in wsclean_images:
                    os.system(f"rm -rf {img}")
                for mod in wsclean_models:
                    os.system(f"rm -rf {mod}")
                for res in wsclean_residuals:
                    os.system(f"rm -rf {res}")

        #####################################
        # Analyzing images
        #####################################
        wsclean_files = {}
        for suffix in ["image", "model", "residual"]:
            files = glob.glob(prefix + f"*MFS-{suffix}.fits")
            if not files:
                files = glob.glob(prefix + f"*{suffix}.fits")
            wsclean_files[suffix] = files

        wsclean_images = wsclean_files["image"]
        wsclean_models = wsclean_files["model"]
        wsclean_residuals = wsclean_files["residual"]

        #######################################################################
        # Final frequency averaged images for backup or calculating dynamic ranges
        #######################################################################
        if do_polcal:
            keep_wsclean_images = False
        else:
            keep_wsclean_images = True
        final_image = (
            prefix.replace("present", f"{round_number}") + f"_{pol}_image.fits"
        )
        final_model = (
            prefix.replace("present", f"{round_number}") + f"_{pol}_model.fits"
        )
        final_residual = (
            prefix.replace("present", f"{round_number}") + f"_{pol}_residual.fits"
        )

        if len(wsclean_images) == 0:
            logger.error("No image is made.")
            return 1, applycal_gaintable, 0, 0, "", "", "", [], 0
        elif len(wsclean_images) == 1:
            os.system(f"cp -r {wsclean_images[0]} {final_image}")
        else:
            final_image = make_timeavg_image(
                wsclean_images, final_image, keep_wsclean_images=keep_wsclean_images
            )
        if len(wsclean_models) == 1:
            os.system(f"cp -r {wsclean_models[0]} {final_model}")
        else:
            final_model = make_timeavg_image(
                wsclean_models, final_model, keep_wsclean_images=keep_wsclean_images
            )
        if len(wsclean_residuals) == 1:
            os.system(f"cp -r {wsclean_residuals[0]} {final_residual}")
        else:
            final_residual = make_timeavg_image(
                wsclean_residuals,
                final_residual,
                keep_wsclean_images=keep_wsclean_images,
            )
        os.system("rm -rf *psf.fits")

        #########################################
        # Restoring previous round flags
        #########################################
        with suppress_output():
            flags = flagmanager(vis=msname, mode="list")
        keys = flags.keys()
        for k in keys:
            if k == "MS":
                pass
            else:
                version = flags[0]["name"]
                if "selfcal" in version:
                    try:
                        with suppress_output():
                            if restore_flag:
                                flagmanager(
                                    vis=msname, mode="restore", versionname=version
                                )
                            flagmanager(vis=msname, mode="delete", versionname=version)
                    except BaseException:
                        pass

        #####################################
        # Calculating dynamic ranges
        ######################################
        model_flux, rms_DR, rms = calc_dyn_range(
            final_image,
            final_model,
            final_residual,
            fits_mask=fits_mask,
        )
        if model_flux == 0:
            logger.error("No model flux.\n")
            return 1, applycal_gaintable, 0, 0, "", "", "", [], 0

        ############################
        # Flag backup before selfcal
        ############################
        do_flag_backup(msname, flagtype="selfcal")

        ########################################
        # Check if any calibration is requested
        ########################################
        if do_intensity_cal is False and do_polcal is False:
            logger.info("No calibration is requested. Returing only previous state.")
            return (
                2,
                applycal_gaintable,
                rms_DR,
                rms,
                final_image,
                final_model,
                final_residual,
                [],
                max_pixel_offset,
            )

        #########################################
        # If model prediction failed in polcal
        #########################################
        if (do_polcal or shifted) and prediction_failed:
            logger.error("Error in predicting model.")
            return (
                2,
                applycal_gaintable,
                rms_DR,
                rms,
                final_image,
                final_model,
                final_residual,
                [],
                max_pixel_offset,
            )

        ##############################
        # Perform intensity selfcal
        ##############################
        if do_intensity_cal:
            if fluxscale_mwa:
                solnorm = True
            ##########################
            # Perform gain calibration
            ##########################
            gain_caltable = prefix.replace("present", f"{round_number}") + ".gcal"
            if os.path.exists(gain_caltable):
                os.system("rm -rf " + gain_caltable)

            logger.info(
                f"gaincal(vis='{msname}',caltable='{gain_caltable}',uvrange='{uvrange}',refant='{refant}',solint='{solint}',calmode='{calmode}',minsnr=3,solnorm={solnorm})\n"
            )
            with suppress_output():
                gaincal(
                    vis=msname,
                    caltable=gain_caltable,
                    uvrange=uvrange,
                    refant=refant,
                    minsnr=3,
                    calmode=calmode,
                    solint=f"{solint}",
                    solnorm=solnorm,
                )

            if not os.path.exists(gain_caltable):
                logger.error("No gain solutions are found.\n")
                return 3, applycal_gaintable, 0, 0, "", "", "", [], max_pixel_offset
            applycal_gaintable.append(gain_caltable)
            interp.append("linear")

            #################################
            # Gaincal flagging
            #################################
            (
                _,
                _,
                _,
                pre_flag_frac,
                pre_chan_flag_frac,
                pre_ant_flag_frac,
                pre_time_flag_frac,
            ) = get_cal_flag_info(gain_caltable)
            do_flag_backup(gain_caltable, flagtype="gainflag")
            with suppress_output():
                flagdata(
                    vis=gain_caltable,
                    mode="rflag",
                    datacolumn="CPARAM",
                    timedevscale=5.0,
                    freqdevscale=5.0,
                    flagbackup=False,
                )
            (
                _,
                _,
                _,
                flag_frac,
                chan_flag_frac,
                ant_flag_frac,
                time_flag_frac,
            ) = get_cal_flag_info(gain_caltable)
            if (
                flag_frac - pre_flag_frac > 0.5
                or ant_flag_frac - pre_ant_flag_frac > 0.5
                or time_flag_frac - pre_time_flag_frac > 0.5
            ):
                logger.info("Restoring flags of gaincal solutions.")
                flagmanager(
                    vis=gain_caltable,
                    mode="restore",
                    versionname="gainflag_1",
                )
            else:
                tb = table()
                tb.open(gain_caltable)
                gain = tb.getcol("CPARAM")
                flag = tb.getcol("FLAG")
                tb.close()
                gain[flag] = np.nan
                tb.open(gain_caltable, nomodify=False)
                new_gain = tb.getcol("CPARAM")
                shape = new_gain.shape
                for i in range(shape[0]):
                    avg = np.nanmedian(np.abs(gain[i, ...]))
                    new_gain[i, ...] = new_gain[i, ...] / avg
                tb.putcol("CPARAM", new_gain)
                tb.flush()
                tb.close()
            with suppress_output():
                flagmanager(vis=gain_caltable, mode="delete", versionname="gainflag_1")
            if not do_bandpass and fluxscale_mwa:
                logger.info("Flux scaled gain caltable using MWA reference bandpass.")
                fluxcal_caltable(gain_caltable, attn=solar_attn)

            ##################################
            # Perform bandpass calibration
            ##################################
            if calmode == "ap" and do_bandpass:
                bpass_caltable = prefix.replace("present", f"{round_number}") + ".bcal"
                if os.path.exists(bpass_caltable):
                    os.system("rm -rf " + bpass_caltable)
                logger.info(
                    f"bandpass(vis='{msname}',caltable='{bpass_caltable}',uvrange='{uvrange}',refant='{refant}',solint='inf',gaintable=['{gain_caltable}'],interp={interp},minsnr=3,solnorm=True)\n"
                )
                with suppress_output():
                    bandpass(
                        vis=msname,
                        caltable=bpass_caltable,
                        uvrange=uvrange,
                        refant=refant,
                        minsnr=3,
                        solint="inf",
                        interp=interp,
                        gaintable=[gain_caltable],
                        solnorm=True,
                    )
                if not os.path.exists(bpass_caltable):
                    logger.error("No bandpass solutions are found.\n")
                    if fluxscale_mwa:
                        logger.info(
                            "Flux scaled gain caltable using MWA reference bandpass."
                        )
                        fluxcal_caltable(gain_caltable, attn=solar_attn)
                else:
                    applycal_gaintable.append(bpass_caltable)
                    interp.append("linear,linear")

                    #############################
                    # Bandpass flagging
                    #############################
                    (
                        _,
                        _,
                        _,
                        pre_flag_frac,
                        pre_chan_flag_frac,
                        pre_ant_flag_frac,
                        pre_time_flag_frac,
                    ) = get_cal_flag_info(bpass_caltable)
                    do_flag_backup(bpass_caltable, flagtype="bpassflag")
                    with suppress_output():
                        flagdata(
                            vis=bpass_caltable,
                            mode="rflag",
                            datacolumn="CPARAM",
                            timedevscale=5.0,
                            freqdevscale=5.0,
                            flagbackup=False,
                        )
                    if (
                        flag_frac - pre_flag_frac > 0.5
                        or ant_flag_frac - pre_ant_flag_frac > 0.5
                        or chan_flag_frac - pre_chan_flag_frac > 0.5
                    ):
                        logger.info("Restoring flags of bandpass solutions.")
                        flagmanager(
                            vis=bpass_caltable,
                            mode="restore",
                            versionname="bpassflag_1",
                        )
                    else:
                        tb = table()
                        tb.open(bpass_caltable)
                        gain = tb.getcol("CPARAM")
                        flag = tb.getcol("FLAG")
                        tb.close()
                        gain[flag] = np.nan
                        tb.open(bpass_caltable, nomodify=False)
                        new_gain = tb.getcol("CPARAM")
                        shape = new_gain.shape
                        for i in range(shape[0]):
                            avg = np.nanmedian(np.abs(gain[i, ...]))
                            new_gain[i, ...] = new_gain[i, ...] / avg
                        tb.putcol("CPARAM", new_gain)
                        tb.flush()
                        tb.close()
                    with suppress_output():
                        flagmanager(
                            vis=bpass_caltable, mode="delete", versionname="bpassflag_1"
                        )
                    if fluxscale_mwa:
                        logger.info(
                            "Flux scaled bandpass caltable using MWA reference bandpass."
                        )
                        fluxcal_caltable(bpass_caltable, attn=solar_attn)

            logger.info(
                f"applycal(vis='{msname}',gaintable={applycal_gaintable},interp={interp},applymode='{applymode}',calwt=[False],flagbackup=False)\n"
            )
            with suppress_output():
                applycal(
                    vis=msname,
                    gaintable=applycal_gaintable,
                    interp=interp,
                    applymode=applymode,
                    calwt=[False],
                    flagbackup=False,
                )

        ###################################################
        # Perform polarisation calibration using quartical
        ###################################################
        if do_polcal:
            pol_caltable = prefix.replace("present", f"{round_number}") + ".dcal"
            quartical_log = prefix.replace("present", f"{round_number}") + ".qclog"
            if os.path.exists(pol_caltable):
                os.system(f"rm -rf {pol_caltable}")
            minuv, maxuv = uvrange_casa_to_quartical(msname, uvrange)
            quartical_args = [
                "goquartical",
                f"input_ms.path={msname}",
                "input_ms.data_column=DATA",
                f"input_ms.select_uv_range=[{minuv},{maxuv}]",
                "input_model.recipe=MODEL_DATA",
                f"output.gain_directory={pol_caltable}",
                f"solver.reference_antenna={refant}",
                "output.overwrite=True",
                "output.log_to_terminal=True",
                f"output.log_directory={quartical_log}",
                "solver.terms=[D]",
                "solver.iter_recipe=[200]",
                "solver.propagate_flags=True",
                f"solver.threads={ncpu}",
                "dask.threads=1",
                "D.type=complex",
            ]
            if solint == "inf":
                quartical_args.append("D.time_interval=1")
            elif solint != "int":
                quartical_args.append(f"D.time_interval={solint}")
            else:
                quartical_args.append(f"D.time_interval={max_ntime}")
            if do_bandpass:
                quartical_args.append(f"D.freq_interval={int(freqres*1000.0)}kHz")
            else:
                quartical_args.append("D.freq_interval=1")
            if solve_array_leakage:
                quartical_args.append("D.solve_per=array")
            quartical_cmd = " ".join(quartical_args)
            logger.info(f"\nQuartical cmd: {quartical_cmd}\n")
            quartical_msg = run_quartical(
                quartical_cmd, "paircarsquartical", verbose=False
            )
            os.system(f"rm -rf {quartical_log}")
            if quartical_msg != 0 or os.path.exists(pol_caltable) is False:
                logger.error("Quartical calibration is not successful.\n")
                return 3, [], 0, 0, "", "", "", [], max_pixel_offset
            applycal_gaintable.append(pol_caltable)

            ######################################
            # Flagging quartical table
            ######################################
            pol_caltable = flag_quartical_table(pol_caltable)

            ######################################
            # Caltable normalisation
            ######################################
            if pol_solnorm:
                pol_caltable = quartical_matrix_normalize(pol_caltable, overwrite=True)

            ######################################
            # Applying quartical solutions
            ######################################
            temp_pol_caltable = (
                prefix.replace("present", f"{round_number}") + "_temp.dcal"
            )
            quartical_args = [
                "goquartical",
                f"input_ms.path={msname}",
                "input_ms.data_column=DATA",
                "output.log_to_terminal=True",
                f"output.log_directory={quartical_log}",
                f"output.gain_directory={temp_pol_caltable}",
                "output.overwrite=True",
                "output.products=[corrected_data]",
                "output.columns=[CORRECTED_DATA]",
                "output.flags=True",
                "solver.terms=[D]",
                "solver.iter_recipe=[0]",
                "solver.propagate_flags=True",
                f"solver.threads={ncpu}",
                "dask.threads=1",
                "D.type=complex",
                f"D.load_from={pol_caltable}/D",
            ]
            quartical_cmd = " ".join(quartical_args)
            logger.info(f"\nQuartical cmd: {quartical_cmd}\n")
            quartical_msg = run_quartical(
                quartical_cmd, "paircarsquartical", verbose=False
            )
            os.system(f"rm -rf {quartical_log} {temp_pol_caltable}")
            if quartical_msg != 0:
                logger.error(
                    "Quartical calibration applying solutions is not successful.\n"
                )
                return 3, [], 0, 0, "", "", "", [], max_pixel_offset

        #####################################
        # Flag zeros
        #####################################
        with suppress_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="corrected",
                flagbackup=False,
            )

        ######################################
        # UVsub flagging
        ######################################
        try:
            if do_flag:
                logger.info("Flagging in uv-domain data.\n")
                do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=max(1, ncpu))
        except Exception:
            logger.exception(traceback.print_exc())
        return (
            0,
            applycal_gaintable,
            rms_DR,
            rms,
            final_image,
            final_model,
            final_residual,
            leakage_info_list,
            max_pixel_offset,
        )
    except Exception:
        logger.exception(traceback.print_exc())
        return 4, applycal_gaintable, 0, 0, "", "", "", [], 0
    finally:
        os.chdir(cwd)
