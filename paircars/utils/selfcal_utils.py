import psutil
import numpy as np
import traceback
import copy
import glob
import os
from casatools import msmetadata, table
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


def intensity_selfcal(
    msname,
    logger,
    selfcaldir,
    cellsize,
    imsize,
    round_number=0,
    uvrange="",
    minuv=0,
    calmode="ap",
    solint="60s",
    refant="1",
    solmode="",
    gaintype="G",
    applymode="calonly",
    threshold=3,
    weight="briggs",
    robust=0.0,
    use_previous_model=False,
    use_solar_mask=True,
    mask_radius=32,
    min_tol_factor=-1,
    ncpu=-1,
    mem=-1,
):
    """
    A single self-calibration round

    Parameters
    ----------
    msname : str
        Name of the measurement set
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
    ncpu : int, optional
        Number of CPUs to use in WSClean
    mem : float, optional
        Memory usage limit in WSClean

    Returns
    -------
    int
        Success message
    str
        Caltable name
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
    """
    limit_threads(n_threads=ncpu)
    from casatasks import gaincal, bandpass, applycal, flagdata, delmod, flagmanager

    try:
        ##################################
        # Setup wsclean params
        ##################################
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = round(psutil.virtual_memory().available / (1024**3), 2)
        msname = msname.rstrip("/")
        if not use_previous_model:
            delmod(vis=msname, otf=True, scr=True)
        prefix = (
            selfcaldir
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_selfcal_present"
        )

        ####################################
        # Determining ms metadata
        ####################################
        msmd = msmetadata()
        msmd.open(msname)

        freq = msmd.meanfreq(0, unit="MHz")
        nchan = msmd.nchan(0)
        freqres = msmd.chanres(0, unit="MHz")

        times = msmd.timesforspws(0)
        ntime = len(times)
        total_time = max(times) - min(times)
        max_intervals = min(1, int(total_time // 10))  # Minimum 10s chunk

        msmd.close()

        ########################################
        # Scale bias determination
        ########################################
        scale_bias = round(get_multiscale_bias(freq), 2)

        ###############################
        # Determine temporal chunking
        ###############################
        if min_tol_factor <= 0:
            min_tol_factor = 1.0  # In percentage
        nintervals, _ = get_optimal_image_interval(
            msname,
            temporal_tol_factor=float(min_tol_factor / 100.0),
            spectral_tol_factor=0.1,
            max_nchan=-1,
            max_ntime=max_intervals,
        )

        os.system(f"rm -rf {prefix}*image.fits {prefix}*residual.fits")

        if weight == "briggs":
            weight += " " + str(robust)

        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-no-dirty",
            "-gridder tuned-wgridder",
            "-weight " + weight,
            "-niter 10000",
            "-mgain 0.85",
            "-nmiter 5",
            "-gain 0.1",
            "-minuv-l " + str(minuv),
            "-j " + str(ncpu),
            "-abs-mem " + str(mem),
            "-no-negative",
            "-auto-mask " + str(threshold + 0.1),
            "-auto-threshold " + str(threshold),
        ]

        ngrid = int(ncpu / 2)
        if ngrid > 1:
            wsclean_args.append("-parallel-gridding " + str(ngrid))

        ################################################
        # Creating and using solar mask
        ################################################
        if use_solar_mask:
            fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
            if os.path.exists(fits_mask) == False:
                logger.info(f"Creating solar mask of size: {mask_radius} arcmin.\n")
                fits_mask = create_circular_mask(
                    msname, cellsize, imsize, mask_radius=mask_radius
                )

        if fits_mask is not None and os.path.exists(fits_mask):
            wsclean_args.append("-fits-mask " + fits_mask)

        ######################################
        # Determining multiscale parameter
        ######################################
        sun_dia = calc_sun_dia(freq)  # Sun diameter in arcmin
        sun_rad = sun_dia / 2
        multiscale_scales = calc_multiscale_scales(msname, 3, max_scale=sun_rad)
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
        wsclean_args.append(
            "-multiscale-scales " + ",".join([str(s) for s in multiscale_scales])
        )
        wsclean_args.append(f"-multiscale-scale-bias {scale_bias_list[i]}")
        if imsize >= 2048 and 4 * max(multiscale_scales) < 1024:
            wsclean_args.append("-parallel-deconvolution 1024")

        #####################################
        # Temporal imaging configuration
        #####################################
        if nintervals > 1:
            wsclean_args.append(f"-intervals-out {nintervals}")
        logger.info(f"Temporal chunks: {nintervals}.")
        wsclean_args.append(f"-name {prefix}")

        if use_previous_model:
            previous_models = glob.glob(f"{prefix}*model.fits")
            total_models_expected = nintervals
            if len(previous_models) == total_models_expected:
                wsclean_args.append("-continue")
            else:
                os.system(f"rm -rf {prefix}*")

        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        logger.info(f"WSClean command: {wsclean_cmd}\n")
        msg = run_wsclean(wsclean_cmd, "solarwsclean", verbose=False)
        if msg != 0:
            logger.info(f"Imaging is not successful.\n")
            return 1, "", 0, 0, "", "", ""

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
            print("No image is made.")
            return 1, "", 0, 0, "", "", ""
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
        # Restoring flags if applymode is calflag
        #########################################
        if applymode == "calflag":
            with suppress_output():
                flags = flagmanager(vis=msname, mode="list")
            keys = flags.keys()
            for k in keys:
                if k == "MS":
                    pass
                else:
                    version = flags[0]["name"]
                    if "applycal" in version:
                        try:
                            with suppress_output():
                                flagmanager(
                                    vis=msname, mode="restore", versionname=version
                                )
                                flagmanager(
                                    vis=msname, mode="delete", versionname=version
                                )
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
            logger.info(f"No model flux.\n")
            return 1, "", 0, 0, "", "", ""

        #####################
        # Perform calibration
        #####################
        bpass_caltable = prefix.replace("present", f"{round_number}") + ".gcal"
        if os.path.exists(bpass_caltable):
            os.system("rm -rf " + bpass_caltable)

        logger.info(
            f"bandpass(vis='{msname}',caltable='{bpass_caltable}',uvrange='{uvrange}',refant='{refant}',solint='{solint},{freqres}MHz',minsnr=1,solnorm=True)\n"
        )
        with suppress_output():
            bandpass(
                vis=msname,
                caltable=bpass_caltable,
                uvrange=uvrange,
                refant=refant,
                minsnr=1,
                solint=f"{solint},{freqres}MHz",
                solnorm=True,
            )
        if os.path.exists(bpass_caltable) == False:
            logger.info(f"No gain solutions are found.\n")
            return 2, "", 0, 0, "", "", ""

        #########################################
        # Flagging bad gains
        #########################################
        with suppress_output():
            flagdata(
                vis=bpass_caltable, mode="rflag", datacolumn="CPARAM", flagbackup=False
            )
        tb = table()
        tb.open(bpass_caltable, nomodify=False)
        gain = tb.getcol("CPARAM")
        if calmode == "p":
            gain /= np.abs(gain)
        flag = tb.getcol("FLAG")
        gain[flag] = 1.0
        pos = np.where(np.abs(gain) == 0.0)
        gain[pos] = 1.0
        flag *= False
        tb.putcol("CPARAM", gain)
        tb.putcol("FLAG", flag)
        tb.flush()
        tb.close()

        logger.info(
            f"applycal(vis={msname},gaintable=[{bpass_caltable}],interp=['linear,nearestflag'],applymode='{applymode}',calwt=[False])\n"
        )
        with suppress_output():
            applycal(
                vis=msname,
                gaintable=[bpass_caltable],
                interp=["linear,nearestflag"],
                applymode=applymode,
                calwt=[False],
            )

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
        return (
            0,
            bpass_caltable,
            rms_DR,
            rms,
            final_image,
            final_model,
            final_residual,
        )
    except Exception as e:
        print(e)
        traceback.print_exc()
        return 3, "", 0, 0, "", "", ""
