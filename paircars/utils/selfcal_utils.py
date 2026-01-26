import psutil
import numpy as np
import traceback
import copy
import glob
import os
import subprocess
import copy
import types
import time
from casatools import msmetadata, table, componentlist, ms as casamstool
from casatasks import ft, importfits, imhead
from astropy.io import fits
from .basic_utils import *
from .resource_utils import *
from .proc_manage_utils import *
from .ms_metadata import *
from .casatasks import *
from .flagging import *
from .calibration import *
from .imaging import *
from .image_utils import *
from .udocker_utils import *


def determine_disk_visibility(msname):
    """
    Determine whether solar disk is visible or not

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    numpy.array
        Channel list where disk may not be detected
    numpy.array
        Timestamp list where disk may not be detected
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    wavelength = (3 * 10**8) / freq
    uvdist = 10.0 * wavelength
    mstool = casamstool()
    mstool.open(msname)
    mstool.select({"uvdist": [0.0, uvdist]})
    data_short = np.nanmean(
        np.abs(mstool.getdata("DATA", ifraxis=True)["data"]), axis=2
    )
    mstool.close()
    uvdist = 150.0 * wavelength
    mstool.open(msname)
    mstool.select({"uvdist": [uvdist - 10.0, uvdist + 10.0]})
    data_first_lobe = np.nanmean(
        np.abs(mstool.getdata("DATA", ifraxis=True)["data"]), axis=2
    )
    mstool.close()
    r = data_first_lobe / data_short
    r_I = (r[0, ...] + r[-1, ...]) / 2.0
    pos = np.where(r_I >= 0.03)
    chans = pos[0]
    timestamps = pos[1]
    r_I[pos] = np.nan
    return chans, timestamps


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
        chans, timestamps = determine_disk_visibility(msname)
        msmd = msmetadata()
        msmd.open(msname)
        times = msmd.timesforspws(0)
        msmd.close()

        for i in range(len(chans)):
            spw = f"0:{chans[i]}"
            timerange = f"{mjdsec_to_timestamp(times[timestamps[i]], str_format=1)}"
            flagdata(
                vis=msname,
                mode="manual",
                spw=spw,
                timerange=timerange,
                flagbackup=False,
            )

        return 0
    except Exception as e:
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
    from casatasks import ft, delmod, gaincal, applycal

    prefix = (
        selfcaldir + "/" + os.path.basename(msname).split(".ms")[0] + "_selfcal_present"
    )
    bpass_caltable = prefix.replace("present", f"{0}") + ".gcal"
    if os.path.exists(bpass_caltable):
        os.system("rm -rf " + bpass_caltable)

    try:
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
            f"gaincal(vis='{msname}',caltable='{bpass_caltable}',uvrange='<100lambda',refant='{refant}',solint='{solint}',minsnr=1,calmode='p')\n"
        )
        with suppress_output():
            gaincal(
                vis=msname,
                caltable=bpass_caltable,
                uvrange="<100lambda",
                refant=refant,
                minsnr=1,
                solint=f"{solint}",
                solnorm=True,
                calmode="p",
            )
        if os.path.exists(bpass_caltable) == False:
            logger.info(f"No gain solutions are found.\n")
            return 2, ""

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
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
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

    ###################################
    # Creating mask
    ####################################
    imageheader = fits.getheader(imagename)
    modelheader = fits.getheader(modelname)
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
    logger,
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
    logger : logger
        Python logger
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
    leakage_info = []
    pbfile = imagename.split(".fits")[0] + "_pb.npy"
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
            "--save_pb",
            "--pb_jones_file",
            f"{pbfile}",
        ]
        logger.info(f"Correcting primary beam: {imagename} to {pbcor_image}.\n")
        result = subprocess.run(
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
        logger.info(f"Correcting primary beam: {modelname} to {pbcor_model}.\n")
        result = subprocess.run(
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
        logger.info(f"Estimating leakage on : {pbcor_image}.\n")
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
        logger.info(
            f"Undo leakage corrected image: {leakagecor_image} to {imagename}.\n"
        )
        result = subprocess.run(
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
        logger.info(
            f"Undo leakage corrected model: {leakagecor_model} to {modelname}.\n"
        )
        result = subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return final_image, final_model, leakage_info


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
    applymode="calonly",
    threshold=3,
    weight="briggs",
    robust=0.0,
    use_previous_model=False,
    use_solar_mask=True,
    mask_radius=40,
    min_tol_factor=-1,
    fluxscale_mwa=False,
    solar_attn=10,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    do_intensity_cal=False,
    do_polcal=False,
    solve_array_leakage=False,
    pol_solnorm=False,
    do_uvsub_flag=False,
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
    calmode : str, optional
        Calibration mode ('p' or 'ap')
    solint : str, optional
        Solution intervals
    solnorm : bool, optional
        Solution normalisation
    refant : str, optional
        Reference antenna
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
    do_uvsub_flag : bool, optional
        Perform UVsub flagging
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
    """
    limit_threads(n_threads=ncpu)
    from casatasks import gaincal, bandpass, applycal, flagdata, delmod, flagmanager

    cwd = os.getcwd()
    ##################################
    # Setup wsclean params
    ##################################
    if ncpu < 1:
        ncpu = psutil.cpu_count()
    if mem < 0:
        mem = round(psutil.virtual_memory().available / (1024**3), 2)
    msname = msname.rstrip("/")
    msname = os.path.abspath(msname)
    os.chdir(selfcaldir)

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
        nchan = msmd.nchan(0)
        freqres = msmd.chanres(0, unit="MHz")[0]
        bw = nchan * freqres

        times = msmd.timesforspws(0)
        ntime = len(times)
        total_time = max(times) - min(times)

        msmd.close()

        #####################################
        # Get column names
        #####################################
        tb = table()
        tb.open(msname)
        colnames = tb.colnames()
        tb.close()

        ######################################
        # Determine spectro-temporal chunking
        ######################################
        if do_intensity_cal:
            if calmode == "ap":
                nchans = max(
                    1, round(bw / 0.32)
                )  # Fixed to 320 kHz, 4 channels per coarse channels
            else:
                nchans = 1
            if min_tol_factor <= 0:
                min_tol_factor = 1.0  # In percentage
            nintervals, _ = get_optimal_image_interval(
                msname,
                temporal_tol_factor=float(min_tol_factor / 100.0),
                spectral_tol_factor=0.1,
            )
        else:
            nchans = max(
                    1, round(bw / 0.32)
                ) 
            nintervals = 1

        os.system(f"rm -rf {prefix}*image.fits {prefix}*residual.fits")

        if weight == "briggs":
            weight += " " + str(robust)

        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-no-dirty",
            "-gridder wgridder",
            "-weight " + weight,
            "-niter 10000",
            "-mgain 0.85",
            "-nmiter 5",
            "-gain 0.1",
            "-minuv-l " + str(minuv),
            "-j " + str(ncpu),
            "-abs-mem " + str(mem),
            "-auto-mask " + str(threshold + 0.1),
            "-auto-threshold " + str(threshold),
        ]
        if do_intensity_cal:
            wsclean_args.append("-pol I")
            if calmode == "p":
                wsclean_args.append("-no-negative")
        else:
            wsclean_args.append("-pol IQUV")

        ngrid = int(ncpu / 2)
        if ngrid > 1:
            wsclean_args.append("-parallel-gridding " + str(ngrid))

        ################################################
        # Creating and using solar mask
        ################################################
        fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
        if os.path.exists(fits_mask) == False:
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
            wsclean_args.append("-no-mf-weighting")

        #####################################
        # Image naming
        #####################################
        wsclean_args.append(f"-name {prefix}")
        if use_previous_model:
            previous_models = glob.glob(f"{prefix}*model.fits")
            total_models_expected = nintervals * nchans
            if len(previous_models) == total_models_expected:
                wsclean_args.append("-continue")
            else:
                os.system(f"rm -rf {prefix}*")

        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        logger.info(f"WSClean command: {wsclean_cmd}\n")
        msg = run_wsclean(wsclean_cmd, "solarwsclean", verbose=False)
        if msg != 0:
            logger.error(f"Imaging is not successful.\n")
            return 1, applycal_gaintable, 0, 0, "", "", "", []

        if do_polcal:
            #######################################
            # Making stokes cube
            #######################################
            imagelist = []
            modellist = []
            pollist = ["I", "Q", "U", "V"]
            for suffix in ["image", "model", "residual"]:
                stokeslist = []
                for p in pollist:
                    stokeslist.append(
                        sorted(glob.glob(prefix + "*" + p + f"-{suffix}.fits"))
                    )
                for i in range(len(stokeslist[0])):
                    wsclean_images = sorted(
                        [stokeslist[k][i] for k in range(len(pollist))]
                    )
                    image_prefix = (
                        selfcaldir
                        + "/"
                        + os.path.basename(wsclean_images[0]).split(f"-{suffix}")[0]
                    )
                    image_cube = make_stokes_wsclean_imagecube(
                        wsclean_images,
                        image_prefix + f"_IQUV_{suffix}.fits",
                        keep_wsclean_images=False,
                    )
                    if suffix == "image":
                        imagelist.append(image_cube)
                    if suffix == "model":
                        modellist.append(image_cube)

            #####################################
            # PB correction and residual leakages
            #####################################
            if pbcor is True or leakagecor is True or pbuncor is True and do_polcal:
                delmod(vis=msname, otf=True, scr=True)
                for count in range(len(imagelist)):
                    imagename = imagelist[count]
                    modelname = modellist[count]
                    if "MFS" not in imagename:
                        valid_image = check_valid_image(imagename)
                        if valid_image:
                            imagename, modelname, leakage_info = correct_pbcor_leakage(
                                imagename,
                                modelname,
                                metafits,
                                logger,
                                pbcor=pbcor,
                                leakagecor=leakagecor,
                                pbuncor=pbuncor,
                                ncpu=ncpu,
                            )
                            casa_modelname = f"{modelname.split('.fits')[0]}.model"
                            importfits(
                                fitsimage=modelname,
                                imagename=casa_modelname,
                                defaultaxes=True,
                                defaultaxesvalues=["ra", "dec", "stokes", "freq"],
                                overwrite=True,
                            )
                            header = imhead(imagename=casa_modelname,mode="list")
                            cent_freq = header["crval4"]/10**6
                            bw = header["cdelt4"]/10**6
                            spw=f"0:{cent_freq-(bw/2)}~{cent_freq+(bw/2)}MHz"
                            ft(vis=msname, model=casa_modelname, spw=spw, incremental=True, usescratch=True)
                            os.system(f"rm -rf {casa_modelname}")
                            leakage_info_list.append(leakage_info)

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
        final_image = prefix.replace("present", f"{round_number}") + "_I_image.fits"
        final_model = prefix.replace("present", f"{round_number}") + "_I_model.fits"
        final_residual = (
            prefix.replace("present", f"{round_number}") + "_I_residual.fits"
        )

        if len(wsclean_images) == 0:
            logger.error("No image is made.")
            return 1, applycal_gaintable, 0, 0, "", "", "", []
        elif len(wsclean_images) == 1:
            os.system(f"cp -r {wsclean_images[0]} {final_image}")
        else:
            final_image = make_timeavg_image(
                wsclean_images, final_image, keep_wsclean_images=True
            )
        if len(wsclean_models) == 1:
            os.system(f"cp -r {wsclean_models[0]} {final_model}")
        else:
            final_model = make_timeavg_image(
                wsclean_models, final_model, keep_wsclean_images=True
            )
        if len(wsclean_residuals) == 1:
            os.system(f"cp -r {wsclean_residuals[0]} {final_residual}")
        else:
            final_residual = make_timeavg_image(
                wsclean_residuals, final_residual, keep_wsclean_images=True
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
                            flagmanager(vis=msname, mode="restore", versionname=version)
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
            logger.error(f"No model flux.\n")
            return 1, applycal_gaintable, 0, 0, "", "", "", []

        ############################
        # Flag backup before selfcal
        ############################
        do_flag_backup(msname, flagtype="selfcal")

        ########################################
        # Check if any calibration is requested
        ########################################
        if do_intensity_cal is False and do_polcal is False:
            logger.info("No calibration is requested. Returing only previous state.")
            return 2, [], rms_DR, rms, final_image, final_model, final_residual, []

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
                f"gaincal(vis='{msname}',caltable='{gain_caltable}',uvrange='{uvrange}',refant='{refant}',solint='{solint}',calmode='{calmode}',minsnr=1,solnorm={solnorm})\n"
            )
            with suppress_output():
                gaincal(
                    vis=msname,
                    caltable=gain_caltable,
                    uvrange=uvrange,
                    refant=refant,
                    minsnr=1,
                    calmode=calmode,
                    solint=f"{solint}",
                    solnorm=solnorm,
                )

            if os.path.exists(gain_caltable) == False:
                logger.error(f"No gain solutions are found.\n")
                return 3, applycal_gaintable, 0, 0, "", "", "", []
            applycal_gaintable.append(gain_caltable)
            interp.append("linear")

            ##################################
            # Perform bandpass calibration
            ##################################
            if calmode == "ap":
                bpass_caltable = prefix.replace("present", f"{round_number}") + ".bcal"
                if os.path.exists(bpass_caltable):
                    os.system("rm -rf " + bpass_caltable)

                logger.info(
                    f"bandpass(vis='{msname}',caltable='{bpass_caltable}',uvrange='{uvrange}',refant='{refant}',solint='inf',gaintable=['{gain_caltable}'],minsnr=1,solnorm=True)\n"
                )
                with suppress_output():
                    bandpass(
                        vis=msname,
                        caltable=bpass_caltable,
                        uvrange=uvrange,
                        refant=refant,
                        minsnr=1,
                        solint="inf",
                        gaintable=[gain_caltable],
                        solnorm=True,
                    )
                if os.path.exists(bpass_caltable) == False:
                    logger.error(f"No bandpass solutions are found.\n")
                    return 3, applycal_gaintable, 0, 0, "", "", "", []

                applycal_gaintable.append(bpass_caltable)
                interp.append("linear,linear")

            #########################################
            # Flagging bad gains
            #########################################
            if calmode == "ap":
                with suppress_output():
                    flagdata(
                        vis=gain_caltable,
                        mode="rflag",
                        datacolumn="CPARAM",
                        flagbackup=False,
                    )
                    flagdata(
                        vis=bpass_caltable,
                        mode="rflag",
                        datacolumn="CPARAM",
                        flagbackup=False,
                    )
                if fluxscale_mwa:
                    logger.info("Flux scaled caltable using MWA reference bandpass.")
                    fluxcal_caltable(bpass_caltable, attn=solar_attn)

            logger.info(
                f"applycal(vis={msname},gaintable={applycal_gaintable},interp={interp},applymode='{applymode}',calwt=[False],flagbackup=False)\n"
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
                f"input_ms.data_column=DATA",
                f"input_ms.select_uv_range=[{minuv},{maxuv}]",
                "input_model.recipe=MODEL_DATA",
                f"output.gain_directory={pol_caltable}",
                f"solver.reference_antenna={refant}",
                "output.overwrite=True",
                "output.log_to_terminal=True",
                f"output.log_directory={quartical_log}",
                "solver.terms=[D]",
                "solver.iter_recipe=[200]",
                "solver.propagate_flags=False",
                f"solver.threads={ncpu}",
                "dask.threads=1",
                "D.type=complex",
                f"D.time_interval={solint}",
                f"D.freq_interval=1",
            ]
            if solve_array_leakage:
                quartical_args.append("D.solve_per=array")
            quartical_cmd = " ".join(quartical_args)
            logger.info(f"Quartical cmd: {quartical_cmd}\n")
            quartical_msg = run_quartical(
                quartical_cmd, "solarquartical", verbose=False
            )
            os.system(f"rm -rf {quartical_log}")
            if quartical_msg != 0 or os.path.exists(pol_caltable) is False:
                logger.error(f"Quartical calibration is not successful.\n")
                return 3, [], 0, 0, "", "", "", []
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
            if applymode == "calonly":
                calflag = False
            else:
                calflag = True
            temp_pol_caltable = (
                prefix.replace("present", f"{round_number}") + "_temp.dcal"
            )
            quartical_args = [
                "goquartical",
                f"input_ms.path={msname}",
                f"input_ms.data_column=DATA",
                "output.log_to_terminal=True",
                f"output.log_directory={quartical_log}",
                f"output.gain_directory={temp_pol_caltable}",
                "output.overwrite=True",
                "output.products=[corrected_data]",
                "output.columns=[CORRECTED_DATA]",
                f"output.flags={calflag}",
                "solver.terms=[D]",
                "solver.iter_recipe=[0]",
                "solver.propagate_flags=False",
                f"solver.threads={ncpu}",
                "dask.threads=1",
                "D.type=complex",
                f"D.load_from={pol_caltable}/D",
            ]
            quartical_cmd = " ".join(quartical_args)
            logger.info(f"Quartical cmd: {quartical_cmd}\n")
            quartical_msg = run_quartical(
                quartical_cmd, "solarquartical", verbose=False
            )
            os.system(f"rm -rf {quartical_log} {temp_pol_caltable}")
            if quartical_msg != 0:
                logger.error(
                    f"Quartical calibration applying solutions is not successful.\n"
                )
                return 3, [], 0, 0, "", "", "", []

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
        if do_uvsub_flag:
            logger.info("UVsub flagging on residual data. Threshold: 5.0.\n")
            uvbin_flag(
                msname,
                uvbin_size=50,
                datacolumn="residual",
                mode="rflag",
                threshold=5.0,
                flagbackup=False,
            )

        return (
            0,
            applycal_gaintable,
            rms_DR,
            rms,
            final_image,
            final_model,
            final_residual,
            leakage_info_list,
        )
    except Exception as e:
        traceback.print_exc()
        return 4, applycal_gaintable, 0, 0, "", "", "", []
    finally:
        os.chdir(cwd)


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
