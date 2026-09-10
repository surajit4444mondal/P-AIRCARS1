import logging
import numpy as np
import argparse
import time
import sys
import os
from casatools import msmetadata
from dask import delayed
from functools import partial
from astropy.io import fits
from paircars.utils.basic_utils import (
    suppress_output,
    weighted_mean,
    print_banner,
)
from paircars.utils.calibration import (
    get_caltable_metadata,
    get_quartical_table_metadata,
)
from paircars.utils.flagging import (
    get_unflagged_antennas,
    get_chans_flag,
    get_times_flag,
)
from paircars.utils.imaging import (
    calc_field_of_view,
    calc_cellsize,
    get_fft_size,
    calc_sun_dia,
    get_optimal_image_interval,
    calc_multiscale_scales,
    get_multiscale_bias,
)
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    create_logger,
    init_logger,
    get_logger_safe,
)
from paircars.utils.ms_metadata import (
    check_datacolumn_valid,
)
from paircars.utils.mwa_utils import (
    freq_to_MWA_coarse,
    get_MWA_OBSID,
    get_MWA_coarse_chan,
    get_selfcal_uvrange,
)
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.resource_utils import drop_cache, limit_threads
from paircars.utils.selfcal_utils import (
    quiet_sun_selfcal,
    selfcal_round,
    leakage_fitting,
)
from paircars.utils.udocker_utils import (
    check_udocker_container,
    initialize_wsclean_container,
    initialize_quartical_container,
)

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def do_selfcal(
    msname="",
    workdir="",
    selfcaldir="",
    metafits="",
    cal_applied=True,
    refant="",
    start_threshold=5,
    end_threshold=3,
    max_iter=30,
    max_DR=100000,
    min_iter=3,
    DR_convergence_frac=0.1,
    min_tol_factor=10.0,
    uvrange="",
    minuv_l=0,
    solint="60s",
    weight="briggs",
    robust=0.0,
    do_apcal=True,
    applymode="calonly",
    ncpu=1,
    mem=1,
    logfile="intselfcal.log",
):
    """
    Do selfcal iterations and use convergence rules to stop

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory
    selfcaldir : str
        Working directory
    metafits : str
        Metafits file
    cal_applied : bool, optional
        Basic calibration applied or not
    refant : str, optional
        Reference antenna
    start_threshold : int, optional
        Start CLEAN threhold
    end_threshold : int, optional
        End CLEAN threshold
    max_iter : int, optional
        Maximum numbers of selfcal iterations (In each selfcal mode)
    max_DR : float, optional
        Maximum dynamic range
    min_iter : int, optional
        Minimum numbers of seflcal iterations at different stages
    DR_convergence_frac : float, optional
        Dynamic range fractional change to consider as converged
    min_tol_factor : float, optional
         Minimum tolerable variation in temporal direction in percentage
    uvrange : str, optional
        UV-range for calibration
    minuv_l : float, optionial
        Minimum UV-lambda to use in imaging
    solint : str, optional
        Solutions interval
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    do_apcal : bool, optional
        Perform ap-selfcal or not
    applymode : str, optional
        Solution apply mode
    ncpu : int, optional
        Number of CPU threads to use
    mem : float, optional
        Memory in GB to use
    logfile : str, optional
        Log file name

    Returns
    -------
    int
        Success message
    str
        Self-calibrated measurement set
    str
        Final caltable
    bool
        Whether disk detected or not
    float
        Final dynamic range
    """
    ncpu = max(1, ncpu)
    mem = abs(mem)

    with limit_threads(n_threads=ncpu):
        from casatasks import split, flagmanager

    sub_observer = None
    intlogger, logfile = create_logger(
        os.path.basename(logfile).split(".log")[0],
        logfile,
    )
    if os.path.exists(f"{workdir}/.jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            sub_observer = init_logger(
                "remotelogger_intselfcal_{os.path.basename(msname).split('.ms')[0]}",
                logfile,
                jobname=jobname,
                password=password,
            )
    try:
        msname = os.path.abspath(msname.rstrip("/"))
        selfcaldir = selfcaldir.rstrip("/")
        if os.path.exists(selfcaldir):
            intlogger.info(
                f"Removing pre-existing intensity selfcal directory: {selfcaldir}.\n"
            )
        os.system(f"rm -rf {selfcaldir}")
        os.makedirs(selfcaldir, exist_ok=True)
        os.chdir(selfcaldir)

        selfcalms = selfcaldir + "/intselfcal_" + os.path.basename(msname)
        if os.path.exists(selfcalms):
            os.system("rm -rf " + selfcalms)
        if os.path.exists(selfcalms + ".flagversions"):
            os.system("rm -rf " + selfcalms + ".flagversions")

        ##############################
        # Restoring any previous flags
        ##############################
        with suppress_output():
            flags = flagmanager(vis=msname, mode="list")
        keys = flags.keys()
        for k in keys:
            if k == "MS":
                pass
            else:
                version = flags[0]["name"]
                try:
                    with suppress_output():
                        flagmanager(vis=msname, mode="restore", versionname=version)
                        flagmanager(vis=msname, mode="delete", versionname=version)
                except BaseException:
                    pass
        if os.path.exists(msname + ".flagversions"):
            os.system("rm -rf " + msname + ".flagversions")

        #########################################
        # Determining calibration applied or not
        #########################################
        fluxscale_mwa = False
        solar_attn = 1
        if cal_applied and os.path.exists(f"{msname}/.applied_sol") is False:
            cal_applied = False
        if cal_applied is False:
            fluxscale_mwa = True
            if os.path.exists(metafits) is False:
                intlogger.error(
                    "Calibration solutions were not applied and target metafits is also not supplied. Provide any one of them.\n"
                )
                return 1, msname, [], False, 0, []
            solar_attn = float(fits.getheader(metafits)["ATTEN_DB"])
            applymode = "calflag"

        ##############################
        # Spliting corrected data
        ##############################
        hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        msmd = msmetadata()
        msmd.open(msname)
        scan = int(msmd.scannumbers()[0])
        field = int(msmd.fieldsforscan(scan)[0])
        msmd.close()
        if hascor:
            intlogger.info(f"Spliting corrected data to ms : {selfcalms}.\n")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="corrected",
                )
        else:
            intlogger.info(f"Spliting data to ms : {selfcalms}.\n")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="data",
                )
        msname = selfcalms

        ################################################################
        # Initial flagging -- zeros, extreme bad data, and non-disk data
        ################################################################
        intlogger.info("Checking initial flagging.\n")
        unflag_chans, flag_chans = get_chans_flag(msname)
        unflag_times, flag_times = get_times_flag(msname)
        if len(unflag_chans) > 0:
            temp_ms = f"{msname}.tempsplit"
            unflag_chans = [f"{i}" for i in unflag_chans]
            unflag_spw = f"0:{';'.join(unflag_chans)}"
            unflag_timerange = ",".join(unflag_times)
            intlogger.info(
                f"Spliting only unflagged spectral window: {unflag_spw}, and timestamps: {unflag_timerange}\n"
            )
            split(
                vis=msname,
                outputvis=temp_ms,
                datacolumn="all",
                spw=unflag_spw,
                timerange=unflag_timerange,
            )
            os.system(f"rm -rf {msname} {msname}.flagversions")
            os.system(f"mv {temp_ms} {msname}")

        ############################################
        # Imaging and calibration parameters
        ############################################
        intlogger.info("Estimating imaging Parameters.\n")
        cellsize = calc_cellsize(msname, 3)
        instrument_fov = calc_field_of_view(msname, FWHM=False)
        cutout_rsun_arcsec = 10 * 16 * 60  # 10 solar radii
        fov = min(instrument_fov, 2 * cutout_rsun_arcsec)
        imsize = int(fov / cellsize)
        imsize = get_fft_size(imsize)
        if refant == "":
            unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
            msmd = msmetadata()
            msmd.open(msname)
            refant_ids = sorted(
                [msmd.antennaids(antname)[0] for antname in unflagged_antenna_names]
            )[0]
            refant = str(refant_ids)
            msmd.close()

        ######################################
        # Determining multiscale parameter
        ######################################
        msmd.open(msname)
        freq = msmd.meanfreq(0, unit="MHz")
        num_chan = msmd.nchan(0)
        msmd.close()
        sun_dia = calc_sun_dia(freq)  # Sun diameter in arcmin
        sun_rad = sun_dia / 2.0
        multiscale_scales = calc_multiscale_scales(msname, 3, max_scale=sun_rad)
        scale_bias = round(get_multiscale_bias(freq), 2)

        ###########################################
        # No bandpass selfcal for single channel ms
        ###########################################
        if num_chan == 1:
            do_bandpass = False
        else:
            do_bandpass = True

        ################################################################
        # Calculating temporal chunks based on tolerance factor
        ################################################################
        if min_tol_factor <= 0:
            min_tol_factor = 10.0  # In percentage
        nintervals, _ = get_optimal_image_interval(
            msname,
            temporal_tol_factor=float(min_tol_factor / 100.0),
            spectral_tol_factor=float(min_tol_factor / 100.0),
        )
        intlogger.info(f"Temporal chunks: {nintervals}.\n")

        ############################################
        # Initiating selfcal Parameters
        ############################################
        intlogger.info("Estimating self-calibration parameters.\n")
        DR1 = 0.0
        DR2 = 0.0
        DR3 = 0.0
        RMS1 = -1.0
        RMS2 = -1.0
        RMS3 = -1.0
        num_iter = 0
        num_iter_after_ap = 0
        num_iter_fixed_sigma = 0
        last_sigma_DR1 = 0
        sigma_reduced_count = 0
        calmode = "p"
        threshold = start_threshold
        last_round_gaintable = []
        last_round_ms = ""
        use_previous_model = False
        disk_detected = False
        min_DR = 0
        issue_occured = False
        min_iter = max(3, min_iter)  # Minimum 3 iterations
        os.system("rm -rf *_selfcal_present*")
        selfcal_minuv_l, selfcal_maxuv_l, selfcal_uvrange = get_selfcal_uvrange(msname)
        if uvrange == "":
            uvrange = selfcal_uvrange
        if minuv_l == 0:
            minuv_l = selfcal_minuv_l

        ##########################################################################
        # Starting using Gaussian model, if calibrator soutions were not applied
        ##########################################################################
        intlogger.info("Starting self-calibration using Gaussian source model.\n")
        msg, _ = quiet_sun_selfcal(
            msname, intlogger, selfcaldir, refant=str(refant), solint="inf"
        )
        if msg == 0:
            intlogger.info(
                "Starting self-calibration using Gaussian model is successful.\n"
            )
        else:
            intlogger.warning(
                "Starting self-calibration using Gaussian model is not successful.\n"
            )

        ##########################################
        # Starting selfcal loops
        ##########################################
        while True:
            ##################################
            # Selfcal round parameters
            ##################################
            issue_occured = False  # Resetting it in every round
            intlogger.info("######################################")
            intlogger.info(
                "Selfcal iteration : "
                + str(num_iter)
                + ", Threshold: "
                + str(threshold)
                + ", Calibration mode: "
                + str(calmode)
            )
            intlogger.info("######################################\n")

            #################################
            # Flagging operators
            #################################
            if num_iter_after_ap > 1 and threshold < start_threshold:
                do_flag = True
                restore_flag = True
            else:
                do_flag = False
                restore_flag = False

            ##################################
            # Mask option
            ##################################
            (
                msg,
                gaintable,
                dyn,
                rms,
                final_image,
                final_model,
                final_residual,
                _,
                disk_detected,
            ) = selfcal_round(
                msname,
                metafits,
                intlogger,
                selfcaldir,
                cellsize,
                imsize,
                round_number=num_iter,
                uvrange=uvrange,
                minuv_l=minuv_l,
                calmode=calmode,
                solint=solint,
                refant=str(refant),
                applymode=applymode,
                threshold=threshold,
                use_previous_model=use_previous_model,
                weight=weight,
                robust=robust,
                nintervals=nintervals,
                multiscale_scales=multiscale_scales,
                scale_bias=scale_bias,
                fluxscale_mwa=fluxscale_mwa,
                do_intensity_cal=True,
                do_bandpass=do_bandpass,
                do_polcal=False,
                solar_attn=solar_attn,
                do_flag=do_flag,
                restore_flag=restore_flag,
                ncpu=ncpu,
                mem=round(mem, 2),
            )
            if msg == 1:
                if num_iter == 0:
                    intlogger.warning(
                        "No model flux is picked up in first round. Trying with lowest threshold and without solar mask.\n"
                    )
                    (
                        msg,
                        gaintable,
                        dyn,
                        rms,
                        final_image,
                        final_model,
                        final_residual,
                        _,
                        disk_detected,
                    ) = selfcal_round(
                        msname,
                        metafits,
                        intlogger,
                        selfcaldir,
                        cellsize,
                        imsize,
                        round_number=num_iter,
                        uvrange=uvrange,
                        minuv_l=minuv_l,
                        calmode=calmode,
                        solint=solint,
                        refant=str(refant),
                        applymode=applymode,
                        threshold=end_threshold,
                        use_previous_model=False,
                        weight=weight,
                        robust=robust,
                        nintervals=nintervals,
                        multiscale_scales=multiscale_scales,
                        scale_bias=scale_bias,
                        fluxscale_mwa=fluxscale_mwa,
                        do_intensity_cal=True,
                        do_bandpass=do_bandpass,
                        do_polcal=False,
                        solar_attn=solar_attn,
                        do_flag=False,
                        restore_flag=True,
                        ncpu=ncpu,
                        mem=round(mem, 2),
                    )
                    if msg == 1:
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        if sub_observer is not None:
                            clean_shutdown(sub_observer)
                        return msg, msname, [], disk_detected, 0
                    else:
                        threshold = end_threshold
                else:
                    os.system("rm -rf *_selfcal_present*")
                    return msg, msname, [], disk_detected, 0
            elif msg > 1:
                intlogger.error("Self-calibration failed.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                if sub_observer is not None:
                    clean_shutdown(sub_observer)
                return msg, msname, [], disk_detected, 0
            if num_iter == 0:
                DR1 = DR3 = DR2 = dyn
                RMS1 = RMS3 = RMS2 = rms
            elif num_iter == 1:
                DR3 = dyn
                RMS3 = rms
                min_DR = dyn
            else:
                DR1 = DR2
                DR2 = DR3
                DR3 = dyn
                RMS1 = RMS2
                RMS2 = RMS3
                RMS3 = rms
            intlogger.info(f"RMS based dynamic ranges: {DR1}, {DR2}, {DR3}.")
            intlogger.info(f"RMS of the images: {RMS1}, {RMS2}, {RMS3}.\n")
            if DR3 >= DR2 and (
                (calmode == "p" and num_iter != 0)
                or (calmode == "ap" and num_iter_after_ap != 1)
            ):
                use_previous_model = True
            else:
                use_previous_model = False

            #################################
            # Checking DR decrease conditions
            #################################
            # Major condition: If DR suddenly drops below starting DR
            #################################
            cond0 = DR3 < 0.9 * min_DR and num_iter > min_iter
            if cond0:
                intlogger.warning(
                    "Dynamic range dropped suddenly below starting dynamic range.\n"
                )
                if os.path.exists(last_round_ms):
                    os.system(f"rm -rf {msname}")
                    os.system(f"cp -r {last_round_ms} {msname}")
                    return (
                        0,
                        msname,
                        last_round_gaintable,
                        disk_detected,
                        DR2,
                    )
                else:
                    return 1, msname, [], False, 0

            ######################################################################
            # Condition 1: If DR is decreasing (DR decrease in phase-only selfcal)
            # Condition 2: If DR suddenly decreased or decreased below starting DR after apcal
            # Condition 3: If DR is decreasing, DR decrease in amplitude-phase selfcal
            ######################################################################
            cond1 = (
                (DR3 < 0.85 * DR2 and DR3 < 0.9 * DR1 and DR2 > DR1)
                and calmode == "p"
                and num_iter > min_iter
            )
            cond2 = DR3 < 0.7 * DR2 and calmode == "ap" and num_iter_after_ap > 1
            cond3 = (
                DR3 < 0.9 * DR2
                and DR2 > 1.1 * DR1
                and calmode == "ap"
                and num_iter_after_ap > min_iter
            )
            if cond1 or cond2 or cond3:
                issue_occured = True
                ################################
                # Replacing with previous ms
                #################################
                if os.path.exists(last_round_ms):
                    os.system(f"rm -rf {msname}")
                    os.system(f"cp -r {last_round_ms} {msname}")
                #############################################
                # Printing condition message
                ##############################################
                if cond1:
                    intlogger.warning(
                        "Dynamic range decreasing in phase-only self-cal.\n"
                    )
                if cond2:
                    intlogger.warning(
                        "Dynamic range dropped suddenly after 'ap' round started.\n"
                    )
                if cond3:
                    intlogger.warning(
                        "Dynamic range is decreasing after minimum numbers of 'ap' round.\n"
                    )
                ##################################################
                # Performing steps
                ##################################################
                if do_apcal and calmode == "p":
                    intlogger.info("Changed calmode to 'ap'.\n")
                    calmode = "ap"
                    use_previous_model = True
                    threshold -= 1
                    intlogger.info(f"Reducing threshold to : {threshold}.\n")
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                    if last_sigma_DR1 > 0:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                    else:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                elif calmode == "ap" and threshold > end_threshold:
                    threshold -= 1
                    intlogger.info(f"Reducing threshold to: {threshold}.\n")
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                    if last_sigma_DR1 > 0:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                    else:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                else:
                    intlogger.warning(
                        "Stopping self-calibration. Using last round caltable as final.\n"
                    )
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return (
                        0,
                        msname,
                        last_round_gaintable,
                        disk_detected,
                        DR2,
                    )

            ###########################
            # If maximum DR has reached
            ###########################
            if DR3 > max_DR and num_iter_after_ap > 1:
                intlogger.info("Maximum dynamic range is reached.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                if sub_observer is not None:
                    clean_shutdown(sub_observer)
                return 0, msname, gaintable, disk_detected, DR3

            ###########################
            # Checking DR convergence
            ###########################
            # Condition 1
            # (If DR did not increase after one round of sigma reduction, do not reduce sigma further and exit)
            ###########################
            if (
                ((do_apcal and calmode == "ap") or not do_apcal)
                and num_iter_fixed_sigma > min_iter
                and (
                    last_sigma_DR1 > 0
                    and abs(round(np.nanmedian([DR1, DR2, DR3]), 0) - last_sigma_DR1)
                    / last_sigma_DR1
                    < DR_convergence_frac
                )
                and sigma_reduced_count > 1
            ):
                if threshold > end_threshold:
                    intlogger.info(
                        "DR does not increase over last two changes in threshold, but minimum threshold has not reached yet.\n"
                    )
                    intlogger.info(
                        f"Starting final self-calibration rounds with threshold = {end_threshold}sigma.\n"
                    )
                    threshold = end_threshold
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                else:
                    intlogger.info("Selfcal calibration has converged.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return 0, msname, gaintable, disk_detected, DR3
            else:
                ################################################################
                # Condition 2
                # If DR does not increase a certain percentage
                # If threshold not reached to end threshold, reducing threshold
                ################################################################
                if (
                    abs(DR1 - DR2) / DR2 < DR_convergence_frac
                    and num_iter > min_iter
                    and num_iter_fixed_sigma > min_iter
                    and threshold > end_threshold
                ):
                    #####################################
                    # Change from phase only selfcal to amplitude-phase selfcal
                    #####################################
                    if do_apcal and calmode == "p":
                        intlogger.info(
                            "Dynamic range converged. Changing calmode to 'ap'.\n"
                        )
                        calmode = "ap"
                        use_previous_model = True
                        threshold -= 1
                        intlogger.info(f"Reducing threshold to : {threshold}.\n")
                        sigma_reduced_count += 1
                        num_iter_fixed_sigma = 0
                        if last_sigma_DR1 > 0:
                            last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                        else:
                            last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                    ######################################
                    # Reducing threshold if already in apcal
                    ######################################
                    elif (do_apcal and num_iter_after_ap > min_iter) or not do_apcal:
                        threshold -= 1
                        intlogger.info(f"Reducing threshold to : {threshold}.\n")
                        sigma_reduced_count += 1
                        num_iter_fixed_sigma = 0
                        if last_sigma_DR1 > 0:
                            last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                        else:
                            last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                ######################################
                # Condition 3
                # If threshold reached, converged
                ######################################
                elif (
                    abs(DR1 - DR2) / DR2 < DR_convergence_frac
                    and num_iter > min_iter
                    and num_iter_fixed_sigma > min_iter
                    and threshold == end_threshold
                ):
                    intlogger.info("Self-calibration has converged.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return 0, msname, gaintable, disk_detected, DR3
                #########################################
                # In apcal and maximum iteration has reached
                #########################################
                elif num_iter > min_iter and (
                    (not do_apcal and num_iter == max_iter)
                    or (do_apcal and calmode == "ap" and num_iter_after_ap == max_iter)
                ):
                    intlogger.info(
                        "Self-calibration is finished. Maximum iteration is reached.\n"
                    )
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return 0, msname, gaintable, disk_detected, DR3
            num_iter += 1
            os.system(f"cp -r {msname} {msname}.round{num_iter}")
            if calmode == "ap":
                num_iter_after_ap += 1
            num_iter_fixed_sigma += 1
            if not issue_occured:
                last_round_gaintable = gaintable
                last_round_ms = f"{msname}.lastround"
                if os.path.exists(last_round_ms):
                    os.system(f"rm -rf {last_round_ms}")
                os.system(f"cp -r {msname} {last_round_ms}")
    except Exception:
        intlogger.exception(
            "Exception occured in intensity self-calibration", exc_info=True
        )
        os.system("rm -rf *_selfcal_present*")
        time.sleep(5)
        if sub_observer is not None:
            clean_shutdown(sub_observer)
        return 1, msname, [], False, 0


def do_polselfcal(
    msname="",
    workdir="",
    selfcaldir="",
    metafits="",
    refant="",
    max_iter=10,
    max_DR=100000,
    min_iter=3,
    threshold=3.0,
    solint="240s",
    DR_convergence_frac=0.1,
    min_tol_factor=10.0,
    uvrange="",
    minuv_l=0,
    weight="briggs",
    robust=0.0,
    disk_present=True,
    leakage_info_polynomial=[],
    ncpu=1,
    mem=1,
    logfile="polselfcal.log",
):
    """
    Do selfcal iterations and use convergence rules to stop

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory
    selfcaldir : str
        Working directory
    metafits : str
        Metafits file
    refant : str, optional
        Reference antenna
    max_iter : int, optional
        Maximum numbers of selfcal iterations
    max_DR : float, optional
        Maximum dynamic range
    min_iter : int, optional
        Minimum numbers of seflcal iterations at different stages
    threshold: float, optional
        Threshold of CLEANing
    solint : str, optional
        Solution interval
    DR_convergence_frac : float, optional
        Dynamic range fractional change to consider as converged
    min_tol_factor : float, optional
         Minimum tolerable variation in temporal direction in percentage
    uvrange : str, optional
        UV-range for calibration
    minuv_l : float, optionial
        Minimum UV-lambda to use in imaging
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    disk_present : bool, optional
        Whether disk is present or not
    leakage_info_polynomial : list, optional
        Leakage info polynomial provided by use [q_leakage poly, u_leakage poly, v_leakage poly]
    ncpu : int, optional
        Number of CPU threads to use
    mem : float, optional
        Memory in GB to use
    logfile : str, optional
        Log file name

    Returns
    -------
    int
        Success message
    str
        Polarisation self-calibrated measurement set
    str
        Final caltable
    str
        Leakage file
    float
        Final image dynamic range
    """
    ncpu = max(1, ncpu)
    mem = abs(mem)

    with limit_threads(n_threads=ncpu):
        from casatasks import split, flagdata

    sub_observer = None
    pollogger, logfile = create_logger(
        os.path.basename(logfile).split(".log")[0],
        logfile,
    )
    if os.path.exists(f"{workdir}/.jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            sub_observer = init_logger(
                "remotelogger_polselfcal_{os.path.basename(msname).split('.ms')[0]}",
                logfile,
                jobname=jobname,
                password=password,
            )
    try:
        msname = os.path.abspath(msname.rstrip("/"))
        selfcaldir = selfcaldir.rstrip("/")
        if os.path.exists(selfcaldir):
            pollogger.info(
                f"Removing pre-existing polarisation selfcal directory: {selfcaldir}.\n"
            )
            os.system(f"rm -rf {selfcaldir}")
        os.makedirs(selfcaldir, exist_ok=True)
        os.chdir(selfcaldir)
        selfcalms = selfcaldir + "/polselfcal_" + os.path.basename(msname)
        if os.path.exists(selfcalms):
            os.system(f"rm -rf {selfcalms}")
        if os.path.exists(f"{selfcalms}.flagversions"):
            os.system(f"rm -rf {selfcalms}.flagversions")

        ##############################
        # Spliting corrected data
        ##############################
        hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        msmd = msmetadata()
        msmd.open(msname)
        scan = int(msmd.scannumbers()[0])
        field = int(msmd.fieldsforscan(scan)[0])
        msmd.close()
        if hascor:
            pollogger.info(f"Spliting corrected data to ms : {selfcalms}.\n")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="corrected",
                )
        else:
            pollogger.warning("Corrected data column is not present.\n")
            pollogger.info(f"Spliting data to ms : {selfcalms}.\n")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="data",
                )
        msname = selfcalms

        ################################################################
        # Initial flagging -- zeros, extreme bad data
        ################################################################
        pollogger.info("Checking initial flagging.\n")
        with suppress_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="data",
                flagbackup=False,
            )
        unflag_chans, flag_chans = get_chans_flag(msname)
        unflag_times, flag_times = get_times_flag(msname)
        if len(unflag_chans) > 0:
            temp_ms = f"{msname}.tempsplit"
            unflag_chans = [f"{i}" for i in unflag_chans]
            unflag_spw = f"0:{';'.join(unflag_chans)}"
            unflag_timerange = ",".join(unflag_times)
            pollogger.info(
                f"Spliting only unflagged spectral window: {unflag_spw} and timestamps: {unflag_timerange}.\n"
            )
            split(
                vis=msname,
                outputvis=temp_ms,
                datacolumn="all",
                spw=unflag_spw,
                timerange=unflag_timerange,
            )
            os.system(f"rm -rf {msname} {msname}.flagversions")
            os.system(f"mv {temp_ms} {msname}")

        ############################################
        # Imaging and calibration parameters
        ############################################
        pollogger.info("Estimating imaging Parameters.\n")
        cellsize = calc_cellsize(msname, 3)
        instrument_fov = calc_field_of_view(msname, FWHM=False)
        cutout_rsun_arcsec = 10 * 16 * 60  # 10 solar radii
        fov = min(instrument_fov, 2 * cutout_rsun_arcsec)
        imsize = int(fov / cellsize)
        imsize = get_fft_size(imsize)
        if refant == "":
            unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
            msmd = msmetadata()
            msmd.open(msname)
            refant_ids = sorted(
                [msmd.antennaids(antname)[0] for antname in unflagged_antenna_names]
            )[0]
            refant = str(refant_ids)
            msmd.close()

        ######################################
        # Determining multiscale parameter
        ######################################
        msmd.open(msname)
        freq = msmd.meanfreq(0, unit="MHz")
        msmd.close()
        sun_dia = calc_sun_dia(freq)  # Sun diameter in arcmin
        sun_rad = sun_dia / 2.0
        multiscale_scales = calc_multiscale_scales(msname, 3, max_scale=sun_rad)
        scale_bias = round(get_multiscale_bias(freq), 2)

        ################################################################
        # Calculating temporal chunks based on tolerance factor
        ################################################################
        if min_tol_factor <= 0:
            min_tol_factor = 10.0  # In percentage
        nintervals, _ = get_optimal_image_interval(
            msname,
            temporal_tol_factor=float(min_tol_factor / 100.0),
            spectral_tol_factor=float(min_tol_factor / 100.0),
        )
        pollogger.info(f"Temporal chunks: {nintervals}.\n")

        ############################################
        # Initiating selfcal Parameters
        ############################################
        pollogger.info("Estimating self-calibration parameters.\n")
        DR1 = 0.0
        DR2 = 0.0
        DR3 = 0.0
        RMS1 = -1.0
        RMS2 = -1.0
        RMS3 = -1.0
        QL1 = QL2 = QL3 = 1.0
        UL1 = UL2 = UL3 = 1.0
        VL1 = VL2 = VL3 = 1.0
        num_iter = 0
        last_round_gaintable = []
        last_leakage_file = ""
        last_round_ms = ""
        solve_array_leakage = True
        issue_occured = False
        num_iter_after_reset = 0
        min_iter = max(3, min_iter)  # Minimum 3 iterations
        leakage_info_dic = {}
        os.system("rm -rf *_selfcal_present*")
        selfcal_minuv_l, selfcal_maxuv_l, selfcal_uvrange = get_selfcal_uvrange(msname)
        if uvrange == "":
            uvrange = selfcal_uvrange
        if minuv_l == 0:
            minuv_l = selfcal_minuv_l

        ##########################################
        # Starting selfcal loops
        ##########################################
        while True:
            issue_occured = False  # Reseting in every round
            ##################################
            # Selfcal round parameters
            ##################################
            pollogger.info("######################################")
            pollogger.info(f"Selfcal iteration : {num_iter}")
            pollogger.info("######################################")
            if not disk_present and len(leakage_info_polynomial) == 0:
                pbcor = False
                leakagecor = False
                pbuncor = False
                min_iter = 1
                if num_iter > 1 and DR3 > DR2:
                    use_previous_model = True
                else:
                    use_previous_model = False
            else:
                if num_iter == 0:
                    pbcor = True
                    leakagecor = True
                    pbuncor = False
                    use_previous_model = False
                elif num_iter < min_iter:
                    pbcor = False
                    leakagecor = True
                    pbuncor = False
                    if num_iter > 1 and DR3 > DR2:
                        use_previous_model = True
                    else:
                        use_previous_model = False
                elif num_iter == min_iter:
                    pbcor = False
                    leakagecor = True
                    pbuncor = True
                    use_previous_model = False
                else:
                    pbcor = True
                    leakagecor = True
                    pbuncor = True
                    if num_iter > min_iter + 1 and DR3 > DR2:
                        use_previous_model = True
                    else:
                        use_previous_model = False

            if (
                num_iter == 0
            ):  # Only corrected at the very first stage by user provided leakage informations, then reset
                leakage_poly = leakage_info_polynomial
            else:
                leakage_poly = []

            if num_iter == min_iter:
                solve_array_leakage = False  # This is to make sure if it failed, last round ms has same state of polcal

            (
                msg,
                gaintable,
                dyn,
                rms,
                final_image,
                final_model,
                final_residual,
                leakage_info,
                _,
            ) = selfcal_round(
                msname,
                metafits,
                pollogger,
                selfcaldir,
                cellsize,
                imsize,
                round_number=num_iter,
                uvrange=uvrange,
                minuv_l=minuv_l,
                refant=str(refant),
                solint=str(solint),
                threshold=threshold,
                weight=weight,
                robust=robust,
                nintervals=nintervals,
                multiscale_scales=multiscale_scales,
                scale_bias=scale_bias,
                do_polcal=True,
                do_intensity_cal=False,
                use_previous_model=use_previous_model,
                pbcor=pbcor,
                leakagecor=leakagecor,
                pbuncor=pbuncor,
                do_flag=True,
                restore_flag=True,
                solve_array_leakage=solve_array_leakage,
                leakage_info_polynomial=leakage_poly,
                ncpu=ncpu,
                mem=round(mem, 2),
            )
            if msg == 1:
                pollogger.error("No model flux is picked up.\n")
                os.system("rm -rf *_selfcal_present*")
                return msg, msname, [], "", 0
            elif msg > 2:
                pollogger.error("Polarisation self-calibration failed.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                if sub_observer is not None:
                    clean_shutdown(sub_observer)
                return msg, msname, [], "", 0
            elif msg == 2:
                if nintervals > 1:
                    if num_iter > min_iter:
                        pollogger.warning(
                            "Minor issues in polarisation self-calibration model prediction. Stopped at previous round.\n"
                        )
                        if os.path.exists(last_round_ms):
                            os.system(f"rm -rf {msname}")
                            os.system(f"cp -r {last_round_ms} {msname}")
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        if sub_observer is not None:
                            clean_shutdown(sub_observer)
                        return 0, msname, last_round_gaintable, last_leakage_file, DR2
                    else:
                        issue_occured = True
                        pollogger.error(
                            "Minor issues in polarisation self-calibration model prediction. Minimum iteration has not covered.\n"
                        )
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        if sub_observer is not None:
                            clean_shutdown(sub_observer)
                        return msg, msname, [], "", 0
                else:
                    issue_occured = True
                    pollogger.warning(
                        "Minor issues in polarisation self-calibration model prediction. Retrying with entire spectro-temporal chunks.\n"
                    )
                    nintervals = 1
            else:
                try:
                    leakage_info = np.array(leakage_info)
                    Q = leakage_info[:, 0]
                    U = leakage_info[:, 1]
                    V = leakage_info[:, 2]
                    Qe = leakage_info[:, 3]
                    Ue = leakage_info[:, 4]
                    Ve = leakage_info[:, 5]
                    q_leakage, q_err = weighted_mean(Q, Qe)
                    u_leakage, u_err = weighted_mean(U, Ue)
                    v_leakage, v_err = weighted_mean(V, Ve)
                except Exception:
                    q_leakage = u_leakage = v_leakage = q_err = u_err = v_err = 0.0
                leakage_info_dic[num_iter] = [
                    q_leakage,
                    u_leakage,
                    v_leakage,
                    q_err,
                    u_err,
                    v_err,
                ]
                leakage_file = f"{gaintable[0].split('.dcal')[0]}.leakage.npy"
                np.save(leakage_file, [freq, leakage_info_dic])
                if num_iter == 0:
                    DR1 = DR3 = DR2 = dyn
                    RMS1 = RMS2 = RMS3 = rms
                    QL1 = QL2 = QL3 = q_leakage
                    UL1 = UL2 = UL3 = u_leakage
                    VL1 = VL2 = VL3 = v_leakage
                    min_DR = dyn
                elif num_iter == 1:
                    DR3 = dyn
                    RMS3 = rms
                    QL3 = q_leakage
                    UL3 = u_leakage
                    VL3 = v_leakage
                else:
                    DR1 = DR2
                    DR2 = DR3
                    DR3 = dyn
                    RMS1 = RMS2
                    RMS2 = RMS3
                    RMS3 = rms
                    QL1 = QL2
                    UL1 = UL2
                    VL1 = VL2
                    QL2 = QL3
                    UL2 = UL3
                    VL2 = VL3
                    QL3 = q_leakage
                    UL3 = u_leakage
                    VL3 = v_leakage
                pollogger.info(f"RMS based dynamic ranges: {DR1}, {DR2}, {DR3}")
                pollogger.info(f"RMS of the images: {RMS1}, {RMS2}, {RMS3}")
                pollogger.info(
                    f"Stokes I to Q leakage: {round(QL1*100.0,3)}, {round(QL2*100.0,3)}, {round(QL3*100.0,3)}%."
                )
                pollogger.info(
                    f"Stokes I to U leakage: {round(UL1*100.0,3)}, {round(UL2*100.0,3)}, {round(UL3*100.0,3)}%."
                )
                pollogger.info(
                    f"Stokes I to V leakage: {round(VL1*100.0,3)}, {round(VL2*100.0,3)}, {round(VL3*100.0,3)}%.\n"
                )

                #################################################
                # Leakage convergence
                #################################################
                leakage_converged = (
                    (QL3 == 0.0 and UL3 == 0.0 and VL3 == 0.0)
                    or (
                        abs(QL2 - QL3) <= 0.01
                        and abs(UL2 - UL3) <= 0.01
                        and abs(VL2 - VL3) <= 0.01
                    )
                    or (abs(QL3) >= q_err and abs(UL3) >= u_err and abs(VL3) >= v_err)
                )

                ########################################
                # Leakage or big DR related issues
                #########################################
                ###################################################################
                # Condition 1: If solving per antenna decrease DR, solve per array
                ###################################################################
                if not solve_array_leakage and DR3 < 0.9 * DR2:
                    if num_iter > min_iter and leakage_converged:
                        pollogger.info(
                            "DR decreasing, but has leakage converged. Self-calibration is finished.\n"
                        )
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")
                        return 0, msname, last_round_gaintable, last_leakage_file, DR2
                    else:
                        pollogger.warning(
                            "Solving over array instead of antenna, as DR decreases.\n"
                        )
                        solve_array_leakage = True
                        issue_occured = True
                        num_iter_after_reset = 0
                        if os.path.exists(last_round_ms):
                            pollogger.info("Replacing with previous measurement set.\n")
                            os.system(f"rm -rf {msname}")
                            os.system(f"cp -r {last_round_ms} {msname}")

                ##########################################
                # Condition 2: If leakage increased
                ##########################################
                if (num_iter == 2 or num_iter > min_iter) and (
                    abs(QL3 - QL2) > 0.1 or abs(UL3 - UL2) > 0.1 or abs(VL3 - VL2) > 0.1
                ):
                    issue_occured = True
                    pollogger.warning("Leakage increased by 10%.\n")
                    if os.path.exists(last_round_ms):
                        pollogger.info("Replacing with previous measurement set.\n")
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")
                        return 0, msname, last_round_gaintable, last_leakage_file, DR2
                    else:
                        return 1, msname, [], "", 0

                #########################################
                # Condition 3: If leakage becomes nan
                #########################################
                if np.isnan(QL3) or np.isnan(UL3) or np.isnan(VL3):
                    pollogger.error(
                        "Leakages become nan. Serious calibration issue occured at the first round.\n"
                    )
                    return 1, msname, [], "", 0

                ################################
                # DR decraeses
                ################################
                # Condition 1: If DR decreased below starting DR
                # Condition 2: If DR is decreasing (DR decrease in pol selfcal)
                # Condition 3: If DR suddenly decreased
                ###############################################################
                cond1 = DR3 < 0.9 * min_DR and num_iter_after_reset > 1
                cond2 = (
                    (DR3 < 0.9 * DR2 and DR2 > 1.5 * DR1)
                    and num_iter > min_iter
                    and num_iter_after_reset > 1
                    and leakage_converged
                )
                cond3 = (
                    DR3 < 0.7 * DR2
                    and num_iter > min_iter
                    and num_iter_after_reset > 1
                    and leakage_converged
                )
                if cond1 or cond2 or cond3:
                    ##############################
                    # Printing condition messages
                    ##############################
                    if cond1:
                        pollogger.warning(
                            f"Dynamic range decreased below start dynamic range: {min_DR}.\n"
                        )
                    if cond2:
                        pollogger.warning(
                            "Dynamic range is decreasing after minimum numbers of rounds.\n"
                        )
                    if cond3:
                        pollogger.warning(
                            "Dynamic range dropped suddenly. Using last round caltable as final.\n"
                        )
                    ###################################
                    # Replacing previous ms
                    ###################################
                    issue_occured = True
                    if os.path.exists(last_round_ms):
                        pollogger.info("Replacing with previous measurement set.\n")
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")
                    if not solve_array_leakage:
                        num_iter_after_reset = 0
                        pollogger.info("Solving over array instead of antenna.\n")
                        solve_array_leakage = True
                    else:
                        if num_iter > min_iter:
                            pollogger.warning(
                                "Stopping self-calibration. Using last round caltables.\n"
                            )
                            os.system("rm -rf *_selfcal_present*")
                            time.sleep(5)
                            if sub_observer is not None:
                                clean_shutdown(sub_observer)
                            return (
                                0,
                                msname,
                                last_round_gaintable,
                                last_leakage_file,
                                DR2,
                            )
                        else:
                            pollogger.error(
                                "Encountered this error before minimum number of rounds.\n"
                            )
                            return 1, msname, [], "", 0

                ###########################
                # If maximum DR has reached
                ###########################
                if (
                    DR3 > max_DR
                    and num_iter > min_iter
                    and num_iter_after_reset > 1
                    and leakage_converged
                ):
                    pollogger.info("Maximum dynamic range is reached.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return 0, msname, gaintable, leakage_file, DR3

                ###########################
                # Checking DR convergence
                ###########################
                ########################################
                # Condition 1
                # If DR does not increase a certain percentage
                # Leakage becomes zero or did not reduce
                ########################################
                if (
                    abs(DR1 - DR2) / DR2 < DR_convergence_frac
                    and num_iter > min_iter
                    and num_iter_after_reset > 1
                    and leakage_converged
                ):
                    pollogger.info("Self-calibration has converged.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return 0, msname, gaintable, leakage_file, DR3
                #########################################
                # If maximum iteration has reached
                #########################################
                elif (
                    num_iter > min_iter
                    and num_iter_after_reset > 1
                    and num_iter == max_iter
                ):
                    pollogger.info(
                        "Self-calibration is finished. Maximum iteration is reached.\n"
                    )
                    if leakage_converged is False:
                        pollogger.warning("Leakage did not converge.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    if sub_observer is not None:
                        clean_shutdown(sub_observer)
                    return 0, msname, gaintable, leakage_file, DR3
                num_iter += 1
                num_iter_after_reset += 1
                os.system(f"cp -r {msname} {msname}.round{num_iter}")
                if not issue_occured:
                    last_round_gaintable = gaintable
                    last_leakage_file = leakage_file
                    last_round_ms = f"{msname}.lastround"
                    if os.path.exists(last_round_ms):
                        os.system(f"rm -rf {last_round_ms}")
                    os.system(f"cp -r {msname} {last_round_ms}")
    except Exception:
        pollogger.exception(
            "Exception occured in polarisation self-calibration.", exc_info=True
        )
        os.system("rm -rf *_selfcal_present*")
        time.sleep(5)
        if sub_observer is not None:
            clean_shutdown(sub_observer)
        return 1, msname, [], "", 0


def main(
    mslist,
    metafits,
    workdir,
    caldir,
    cal_applied=True,
    start_thresh=5,
    stop_thresh=3,
    max_iter=30,
    max_DR=100000,
    intselfcal_min_iter=3,
    polselfcal_min_iter=3,
    conv_frac=0.1,
    int_solint="60s",
    pol_solint="240s",
    uvrange="",
    minuv_l=0,
    weight="briggs",
    robust=0.0,
    applymode="calonly",
    min_tol_factor=10.0,
    do_polcal=True,
    do_apcal=True,
    keep_backup=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Perform iterative self-calibration on a list of measurement sets.

    Parameters
    ----------
    mslist : str
        Comma-separated list of target measurement sets to be self-calibrated.
    metafits : str
        Metafits file
    workdir : str
        Path to the working directory for outputs, intermediate files, and logs.
    caldir : str
        Directory containing calibration tables (e.g., from flux or phase calibrators).
    cal_applied : bool, optional
        Basic initial calibration applied or not.
    start_thresh : float, optional
        Initial image dynamic range threshold to start self-calibration. Default is 5.
    stop_thresh : float, optional
        Target dynamic range at which to stop iterative self-calibration. Default is 3.
    max_iter : int, optional
        Maximum number of self-calibration iterations. Default is 30.
    max_DR : float, optional
        Maximum dynamic range allowed before halting iterations. Default is 100000.
    intselfcal_min_iter : int, optional
        Minimum number of iterations before checking for convergence for intensity selfcal. Default is 3.
    polselfcal_min_iter : int, optional
        Minimum number of iterations before checking for convergence for polarisation selfcal. Default is 3.
    conv_frac : float, optional
        Convergence criterion: fractional change in dynamic range below which iteration stops. Default is 0.1.
    int_solint : str, optional
        Solution interval for gain calibration (e.g., "inf", "30s", "int"). Default is "60s".
    pol_solint : str, optional
        Solution interval for polarisation calibration (e.g., "inf", "30s", "int"). Default is "240s".
    uvrange : str, optional
        UV range to be used for imaging and calibration, in CASA format. Default is "" (all baselines).
    minuv_l : float, optional
        Minimum baseline length (in wavelengths) to include. Default is 10.
    weight : str, optional
        Weighting scheme for imaging (e.g., "natural", "uniform", "briggs"). Default is "briggs".
    robust : float, optional
        Robustness parameter for Briggs weighting (ignored if not using "briggs"). Default is 0.0.
    applymode : str, optional
        Apply mode for calibration tables ("calonly", "calflag", etc.). Default is "calonly".
    min_tol_factor : float, optional
        Minimum factor for tolerance comparison during convergence checks. Default is 10.0.
    keep_backup : bool, optional
        If True, keeps backup MS before applying selfcal solutions. Default is False.
    cpu_frac : float, optional
        Fraction of available CPUs to use per job. Default is 0.8.
    mem_frac : float, optional
        Fraction of available system memory to use per job. Default is 0.8.
    logfile : str, optional
        Log file name
    jobid : int, optional
        Identifier for job tracking and logging. Default is 0.
    verbose : bool, optional
        Verbose logs
    start_remote_log : bool, optional
        Whether to initiate remote logging via job credentials. Default is False.
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    int
        Intensity selfcal success number
    int
        Intensity selfcal failed number
    int
        Polarisation selfcal success number
    int
        Polarisation selfcal failed nunber
    float
        Average intensity selfcal dynamic range
    float
        Average polarisation selfcal dynamic range
    float
        Maximum intensity selfcal dynamic range
    float
        Maximum polarisation selfcal dynamic range
    int
        Total disk detected measurement sets
    int
        Total non-disk detected measurement sets
    """
    logger = get_logger_safe()
    if verbose:
        logger.setLevel(logging.DEBUG)

    cpu_frac = min(0.8, abs(cpu_frac))
    mem_frac = min(0.8, abs(mem_frac))

    mslist = mslist.split(",")

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    logger.debug(f"Current working directory: {os.getcwd()}.\n")

    if caldir == "" or not os.path.exists(caldir):
        caldir = f"{workdir}/caltables"
    os.makedirs(caldir, exist_ok=True)
    logger.debug(f"Output caltables directory: {caldir}.\n")

    ############
    # Logger
    ############
    observer = None
    if (
        start_remote_log
        and os.path.exists(f"{workdir}/.jobname_password.npy")
        and logfile is not None
    ):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "all_selfcal", logfile, jobname=jobname, password=password
            )

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.\n")
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        int_succeed = 0
        int_failed = len(mslist)
        pol_succeed = 0
        pol_failed = len(mslist)
        avg_int_DR = 0
        avg_pol_DR = 0
        max_int_DR = 0
        max_pol_DR = 0
        total_disk_detected_ms = 0
        total_non_disk_detected_ms = 0

    ###########################
    # WSClean container
    ###########################
    container_name = "paircarswsclean"
    container_present = check_udocker_container(container_name)
    if not container_present:
        logger.debug(f"Initializing {container_name}.\n")
        container_name = initialize_wsclean_container(name=container_name, verbose=True)
        if container_name is None:
            logger.critical(
                f"Container {container_name} is not initiated. First initiate container and then run.\n"
            )
            return 1, int_succeed, int_failed, pol_succeed, pol_failed, 0, 0, 0, 0, 0, 0

    #############################
    # Quartical container
    #############################
    container_name = "paircarsquartical"
    container_present = check_udocker_container(container_name)
    if not container_present:
        logger.debug(f"Initializing {container_name}.\n")
        container_name = initialize_quartical_container(
            name=container_name, verbose=True
        )
        if container_name is None:
            logger.critical(
                f"Container {container_name} is not initiated. First initiate container and then run.\n"
            )
            return 1, int_succeed, int_failed, pol_succeed, pol_failed, 0, 0, 0, 0, 0, 0

    try:
        for banner in print_banner(
            "Starting self-calibrations.", no_print=True
        ).splitlines():
            logger.info(banner)
        header = fits.getheader(metafits)
        obsid = header["GPSTIME"]

        logger.debug("Determining reference antenna.\n")
        unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(mslist[0])
        msmd = msmetadata()
        msmd.open(mslist[0])
        refant_ids = sorted(
            [msmd.antennaids(antname)[0] for antname in unflagged_antenna_names]
        )[0]
        refant = str(refant_ids)
        msmd.close()
        logger.debug(f"Reference antenna: {refant}.\n")

        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                logger.warning(f"Issue in : {ms}.\n")
                os.system(f"rm -rf {ms}")
        mslist = filtered_mslist
        if len(mslist) == 0:
            logger.critical("No filtered ms to continue.\n")
            return 1, int_succeed, int_failed, pol_succeed, pol_failed, 0, 0, 0, 0, 0, 0

        ##########################################
        # Creating local dask cluster if needed
        ##########################################
        dask_cluster = None
        if dask_client is None:
            dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
                workdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                max_worker=len(mslist) + 1,
            )
            if dask_client is None:
                logger.critical("Error occured in creating local cluster.\n")
                return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            scale_worker_and_wait(dask_cluster, dask_client, nworker)

        #####################################
        client_info = dask_client.scheduler_info()["workers"]
        njobs = len(client_info)
        worker_mem_list = []
        for addr, w in client_info.items():
            worker_mem_list.append(w["memory_limit"] / 1024**3)
        if len(worker_mem_list) > 0:
            mem_limit = round(min(worker_mem_list), 3)
        else:
            mem_limit = 1
        n_threads = os.environ.get("OMP_NUM_THREADS")
        if n_threads is not None:
            n_threads = int(n_threads)
        else:
            n_threads = 1

        logger.info("#################################")
        logger.info(f"Total dask worker: {njobs}")
        logger.info(f"CPU per worker: {n_threads}")
        logger.info(f"Memory per worker: {mem_limit} GB")
        logger.info("#################################\n")

        os.makedirs(f"{workdir}/logs", exist_ok=True)

        succeed_intselfcal = 0
        failed_intselfcal = 0
        succeed_polselfcal = 0
        failed_polselfcal = 0
        gcal_list = []
        bpass_list = []
        int_DR_list = []
        disk_detected_ms = []
        disk_detected_selfcaldir = []
        disk_non_detected_ms = []
        disk_non_detected_selfcaldir = []
        leakage_file_list = []
        pol_DR_list = []
        dcal_list = []

        ################################
        # Intensity and bandpass selfcal
        ################################
        partial_do_selfcal = partial(
            do_selfcal,
            metafits=str(metafits),
            cal_applied=bool(cal_applied),
            refant=str(refant),
            start_threshold=float(start_thresh),
            end_threshold=float(stop_thresh),
            max_iter=int(max_iter),
            max_DR=float(max_DR),
            min_iter=max(3, int(intselfcal_min_iter)),
            DR_convergence_frac=float(conv_frac),
            uvrange=str(uvrange),
            minuv_l=float(minuv_l),
            solint=str(int_solint),
            weight=str(weight),
            robust=float(robust),
            do_apcal=do_apcal,
            applymode=applymode,
        )

        tasks = []
        selfcaldir_list = []
        for ms in mslist:
            obsid = get_MWA_OBSID(ms)
            coarse_chan = get_MWA_coarse_chan(ms)
            coarse_chan = f"{min(coarse_chan)}"
            logfile_prefix = f"{workdir}/logs/selfcal_{obsid}_ch_{coarse_chan}"
            logger.info(f"Measurement set name: {ms}.")
            logger.info(f"Intensity self-cal log file: {logfile_prefix}_int.log")
            selfcaldir = f"{workdir}/{os.path.basename(ms).split('.ms')[0]}_selfcal_int"
            if os.path.exists(selfcaldir):
                os.system(f"rm -rf {selfcaldir}")
            tasks.append(
                delayed(partial_do_selfcal)(
                    msname=ms,
                    workdir=workdir,
                    selfcaldir=selfcaldir,
                    ncpu=n_threads,
                    mem=mem_limit,
                    logfile=f"{logfile_prefix}_int.log",
                )
            )
            selfcaldir_list.append(selfcaldir)
        logger.info("Starting all intensity and bandpass self-calibration.\n")
        results = list(dask_client.gather(dask_client.compute(tasks)))

        for i in range(len(results)):
            r = results[i]
            int_msg = r[0]
            int_ms = r[1]
            gaintables = r[2]
            disk_detected = r[3]
            int_DR = r[4]
            int_DR_list.append(int_DR)
            selfcaldir = selfcaldir_list[i]
            if int_msg != 0:
                logger.error(
                    f"Intensity self-calibration was not successful for ms: {mslist[i]}."
                )
                os.system(
                    f"rm -rf {workdir}/.intselfcal_*_{os.path.basename(mslist[i])}"
                )
                os.system(
                    f"touch {workdir}/.intselfcal_failed_{os.path.basename(mslist[i])}"
                )
                failed_intselfcal += 1
            else:
                if disk_detected:
                    disk_detected_ms.append(int_ms)
                    disk_detected_selfcaldir.append(selfcaldir)
                else:
                    disk_non_detected_ms.append(int_ms)
                    disk_non_detected_selfcaldir.append(selfcaldir)
                try:
                    gcal = gaintables[0]
                    cal_metadata = get_caltable_metadata(gcal)
                    freq_start = cal_metadata["Channel 0 frequency (MHz)"]
                    ch_start = freq_to_MWA_coarse(freq_start)
                    coarse_chan = f"{ch_start}"
                    final_gain_caltable = (
                        caldir + f"/selfcal_{obsid}_ch_{coarse_chan}.gcal"
                    )
                    os.system(f"rm -rf {final_gain_caltable}")
                    os.system(f"cp -r {gcal} {final_gain_caltable}")
                    gcal_list.append(final_gain_caltable)
                    if len(gaintables) > 1:
                        bpass = gaintables[1]
                        cal_metadata = get_caltable_metadata(bpass)
                        freq_start = cal_metadata["Channel 0 frequency (MHz)"]
                        ch_start = freq_to_MWA_coarse(freq_start)
                        coarse_chan = f"{ch_start}"
                        final_bpass_caltable = (
                            caldir + f"/selfcal_{obsid}_ch_{coarse_chan}.bcal"
                        )
                        os.system(f"rm -rf {final_bpass_caltable}")
                        os.system(f"cp -r {bpass} {final_bpass_caltable}")
                        bpass_list.append(final_bpass_caltable)
                    os.system(
                        f"rm -rf {workdir}/.intselfcal_*_{os.path.basename(mslist[i])}"
                    )
                    os.system(
                        f"touch {workdir}/.intselfcal_succeed_{os.path.basename(mslist[i])}"
                    )
                    succeed_intselfcal += 1
                except Exception:
                    logger.exception(
                        "Exception occured in filtering intensity self-calibration caltables.",
                        exc_info=True,
                    )
                    os.system(
                        f"rm -rf {workdir}/.intselfcal_*_{os.path.basename(mslist[i])}"
                    )
                    os.system(
                        f"touch {workdir}/.intselfcal_failed_{os.path.basename(mslist[i])}"
                    )
                    failed_intselfcal += 1

        total_disk_detected_ms = len(disk_detected_ms)
        total_non_disk_detected_ms = len(disk_non_detected_ms)

        if do_polcal:
            #######################################
            # Polarisation selfcal
            #######################################
            partial_do_polselfcal = partial(
                do_polselfcal,
                metafits=str(metafits),
                refant=str(refant),
                max_iter=10,
                max_DR=float(max_DR),
                min_iter=max(3, int(polselfcal_min_iter)),
                threshold=float(stop_thresh),
                DR_convergence_frac=float(conv_frac),
                uvrange=str(uvrange),
                minuv_l=float(minuv_l),
                weight=str(weight),
                robust=float(robust),
                solint=str(pol_solint),
            )
            polcal_mslist = []
            if len(disk_detected_ms) == 0:
                logger.warning(
                    "Quiet sun disk is not detected in any of the measurement set. Phase alignment and polarisation calibration may not be reliable.\n"
                )
                tasks = []
                all_int_ms = disk_detected_ms + disk_non_detected_ms
                all_selfcaldir_list = (
                    disk_detected_selfcaldir + disk_non_detected_selfcaldir
                )
                for i in range(len(all_int_ms)):
                    ms = all_int_ms[i]
                    obsid = get_MWA_OBSID(ms)
                    coarse_chan = get_MWA_coarse_chan(ms)
                    coarse_chan = f"{min(coarse_chan)}"
                    logfile_prefix = f"{workdir}/logs/selfcal_{obsid}_ch_{coarse_chan}"
                    logger.info(f"Measurement set name: {ms}.")
                    logger.info(
                        f"Polarisation self-cal log file: {logfile_prefix}_pol.log"
                    )
                    selfcaldir = all_selfcaldir_list[i].split("_int")[0] + "_pol"
                    if os.path.exists(selfcaldir):
                        os.system(f"rm -rf {selfcaldir}")
                    tasks.append(
                        delayed(partial_do_polselfcal)(
                            msname=ms,
                            workdir=workdir,
                            selfcaldir=selfcaldir,
                            disk_present=False,
                            ncpu=n_threads,
                            mem=mem_limit,
                            logfile=f"{logfile_prefix}_pol.log",
                        )
                    )
                    polcal_mslist.append(ms)
                logger.info("Starting all polarisation self-calibration.\n")
                results = list(dask_client.gather(dask_client.compute(tasks)))
            else:
                tasks = []
                for i in range(len(disk_detected_ms)):
                    ms = disk_detected_ms[i]
                    obsid = get_MWA_OBSID(ms)
                    coarse_chan = get_MWA_coarse_chan(ms)
                    coarse_chan = f"{min(coarse_chan)}"
                    logfile_prefix = f"{workdir}/logs/selfcal_{obsid}_ch_{coarse_chan}"
                    logger.info(f"Measurement set name: {ms}.")
                    logger.info(
                        f"Polarisation self-cal log file: {logfile_prefix}_pol.log"
                    )
                    selfcaldir = disk_detected_selfcaldir[i].split("_int")[0] + "_pol"
                    if os.path.exists(selfcaldir):
                        os.system(f"rm -rf {selfcaldir}")
                    tasks.append(
                        delayed(partial_do_polselfcal)(
                            msname=ms,
                            workdir=workdir,
                            selfcaldir=selfcaldir,
                            disk_present=True,
                            ncpu=n_threads,
                            mem=mem_limit,
                            logfile=f"{logfile_prefix}_pol.log",
                        )
                    )
                    polcal_mslist.append(ms)
                logger.info(
                    "Starting all polarisation self-calibration for disk detected measurement sets.\n"
                )
                results = list(dask_client.gather(dask_client.compute(tasks)))

                ##############################################
                # Results of first set of polarisation selfcal
                ##############################################
                for i in range(len(results)):
                    r = results[i]
                    pol_msg = r[0]
                    gaintables = r[2]
                    leakage_file = r[3]
                    pol_DR = r[4]
                    pol_DR_list.append(pol_DR)
                    if pol_msg != 0:
                        logger.error(
                            f"Polarisation self-calibration was not successful for ms: {polcal_mslist[i]}."
                        )
                        os.system(
                            f"rm -rf {workdir}/.polselfcal_*_{os.path.basename(polcal_mslist[i])}"
                        )
                        os.system(
                            f"touch {workdir}/.polselfcal_failed_{os.path.basename(polcal_mslist[i])}"
                        )
                        failed_polselfcal += 1
                    else:
                        try:
                            dcal = gaintables[0]
                            cal_metadata = get_quartical_table_metadata(dcal)
                            freq_start = cal_metadata["Channel 0 frequency (MHz)"]
                            ch_start = freq_to_MWA_coarse(freq_start)
                            coarse_chan = f"{ch_start}"
                            final_leakage_caltable = (
                                caldir + f"/selfcal_{obsid}_ch_{coarse_chan}.dcal"
                            )
                            os.system(f"rm -rf {final_leakage_caltable}")
                            os.system(f"cp -r {dcal} {final_leakage_caltable}")
                            dcal_list.append(final_leakage_caltable)
                            final_leakage_info = (
                                caldir + f"/selfcal_{obsid}_ch_{coarse_chan}.leakage"
                            )
                            os.system(f"rm -rf {final_leakage_info}")
                            os.system(f"cp -r {leakage_file} {final_leakage_info}")
                            leakage_file_list.append(final_leakage_info)
                            os.system(
                                f"rm -rf {workdir}/.polselfcal_*_{os.path.basename(polcal_mslist[i])}"
                            )
                            os.system(
                                f"touch {workdir}/.polselfcal_succeed_{os.path.basename(polcal_mslist[i])}"
                            )
                            succeed_polselfcal += 1
                        except Exception:
                            logger.exception(
                                "Error occured in filtering polarisation self-calibration caltables.",
                                exc_info=True,
                            )
                            os.system(
                                f"rm -rf {workdir}/.polselfcal_*_{os.path.basename(polcal_mslist[i])}"
                            )
                            os.system(
                                f"touch {workdir}/.polselfcal_failed_{os.path.basename(polcal_mslist[i])}"
                            )
                            failed_polselfcal += 1

                ######################################
                # If there are non-disk detected ms
                ######################################
                if len(disk_non_detected_ms) > 0:
                    q_poly, u_poly, v_poly = leakage_fitting(leakage_file_list)
                    if len(q_poly) == 0 or len(u_poly) == 0 or len(v_poly) == 0:
                        leakage_info_polynomial = []
                    else:
                        leakage_info_polynomial = [q_poly, u_poly, v_poly]
                    tasks = []
                    polcal_mslist = []
                    for i in range(len(disk_non_detected_ms)):
                        ms = disk_non_detected_ms[i]
                        obsid = get_MWA_OBSID(ms)
                        coarse_chan = get_MWA_coarse_chan(ms)
                        coarse_chan = f"{min(coarse_chan)}"
                        logfile_prefix = (
                            f"{workdir}/logs/selfcal_{obsid}_ch_{coarse_chan}"
                        )
                        logger.info(f"Measurement set name: {ms}.")
                        logger.info(
                            f"Polarisation self-cal log file: {logfile_prefix}_pol.log"
                        )
                        selfcaldir = (
                            disk_non_detected_selfcaldir[i].split("_int")[0] + "_pol"
                        )
                        if os.path.exists(selfcaldir):
                            os.system(f"rm -rf {selfcaldir}")
                        tasks.append(
                            delayed(partial_do_polselfcal)(
                                msname=ms,
                                workdir=workdir,
                                selfcaldir=selfcaldir,
                                disk_present=False,
                                ncpu=n_threads,
                                mem=mem_limit,
                                leakage_info_polynomial=leakage_info_polynomial,
                                logfile=f"{logfile_prefix}_pol.log",
                            )
                        )
                        polcal_mslist.append(ms)
                    logger.info(
                        "Starting all polarisation self-calibration for non-disk detected measurement sets.\n"
                    )
                    results = list(dask_client.gather(dask_client.compute(tasks)))

                    for i in range(len(results)):
                        r = results[i]
                        pol_msg = r[0]
                        gaintables = r[2]
                        leakage_file = r[3]
                        pol_DR = r[4]
                        pol_DR_list.append(pol_DR)
                        if pol_msg != 0:
                            logger.error(
                                f"Polarisation self-calibration was not successful for ms: {polcal_mslist[i]}."
                            )
                            os.system(
                                f"rm -rf {workdir}/.polselfcal_*_{os.path.basename(polcal_mslist[i])}"
                            )
                            os.system(
                                f"touch {workdir}/.polselfcal_failed_{os.path.basename(polcal_mslist[i])}"
                            )
                            failed_polselfcal += 1
                        else:
                            try:
                                dcal = gaintables[0]
                                cal_metadata = get_quartical_table_metadata(dcal)
                                freq_start = cal_metadata["Channel 0 frequency (MHz)"]
                                ch_start = freq_to_MWA_coarse(freq_start)
                                coarse_chan = f"{ch_start}"
                                final_leakage_caltable = (
                                    caldir + f"/selfcal_{obsid}_ch_{coarse_chan}.dcal"
                                )
                                os.system(f"cp -r {dcal} {final_leakage_caltable}")
                                dcal_list.append(final_leakage_caltable)
                                final_leakage_info = (
                                    caldir
                                    + f"/selfcal_{obsid}_ch_{coarse_chan}.leakage"
                                )
                                os.system(f"rm -rf {final_leakage_info}")
                                os.system(f"cp -r {leakage_file} {final_leakage_info}")
                                leakage_file_list.append(final_leakage_info)
                                os.system(
                                    f"rm -rf {workdir}/.polselfcal_*_{os.path.basename(polcal_mslist[i])}"
                                )
                                os.system(
                                    f"touch {workdir}/.polselfcal_succeed_{os.path.basename(polcal_mslist[i])}"
                                )
                                succeed_polselfcal += 1
                            except Exception:
                                logger.exception(
                                    "Error occured in filtering polarisation self-calibration caltables.",
                                    exc_info=True,
                                )
                                os.system(
                                    f"rm -rf {workdir}/.polselfcal_*_{os.path.basename(polcal_mslist[i])}"
                                )
                                os.system(
                                    f"touch {workdir}/.polselfcal_failed_{os.path.basename(polcal_mslist[i])}"
                                )
                                failed_polselfcal += 1

        ###################################
        # Deleteing if not keeping backup
        ###################################
        if not keep_backup:
            for ms in mslist:
                int_selfcaldir = (
                    workdir
                    + "/"
                    + os.path.basename(ms).split(".ms")[0]
                    + "_selfcal_int"
                )
                os.system(f"rm -rf {int_selfcaldir}")
                if do_polcal:
                    pol_selfcaldir = (
                        workdir
                        + "/"
                        + os.path.basename(ms).split(".ms")[0]
                        + "_selfcal_pol"
                    )
                    os.system(f"rm -rf {pol_selfcaldir}")

        if len(gcal_list) > 0:
            logger.info("Final gaincal selfcal caltables:")
            for gcal in gcal_list:
                logger.info(gcal)
            msg = 0
            if len(bpass_list) > 0:
                logger.info("Final bandpass selfcal caltables:")
                for bpass in bpass_list:
                    logger.info(bpass)
            else:
                logger.warning("No bandpass self-calibration is present.")
            if len(dcal_list) > 0 and do_polcal:
                logger.info("Final polarisation selfcal caltables:")
                for dcal in dcal_list:
                    logger.info(dcal)
        else:
            logger.error("No self-calibration is successful.")
            msg = 1
        logger.info(f"Total self-calibration measurement sets: {len(mslist)}")
        logger.info(
            f"Total successful intensity self-calibration: {succeed_intselfcal}"
        )
        logger.info(f"Total failed intensity self-calibration: {failed_intselfcal}")
        int_succeed, int_failed = succeed_intselfcal, failed_intselfcal
        if do_polcal:
            logger.info(
                f"Total successful polarisation self-calibration: {succeed_polselfcal}"
            )
            logger.info(
                f"Total failed polarisation self-calibration: {failed_polselfcal}"
            )
            pol_succeed, pol_failed = succeed_polselfcal, failed_polselfcal
        if succeed_intselfcal == 0:
            msg = 1
        if len(int_DR_list) > 0:
            avg_int_DR = round(np.nanmedian(int_DR_list), 2)
            max_int_DR = round(np.nanmax(int_DR_list), 2)
        else:
            avg_int_DR = 0
            max_int_DR = 0
        if len(pol_DR_list) > 0:
            avg_pol_DR = round(np.nanmedian(pol_DR_list), 2)
            max_pol_DR = round(np.nanmax(pol_DR_list), 2)
        else:
            avg_pol_DR = 0
            max_pol_DR = 0
        logger.info(f"Average intensity self-calibration dynamic range: {avg_int_DR}")
        if do_polcal:
            logger.info(
                f"Average polarisation self-calibration dynamic range: {avg_pol_DR}"
            )
        print_banner("Self-calibration is done successfully.")
    except Exception:
        logger.exception("Exception occured in self-calibration.", exc_info=True)
        msg = 1
        avg_int_DR = 0
        avg_pol_DR = 0
        max_int_DR = 0
        max_pol_DR = 0
        print_banner("Self-calibration is failed.")
    finally:
        time.sleep(5)
        clean_shutdown(observer)
        for msname in mslist:
            if os.path.exists(msname):
                drop_cache(msname)
        if dask_cluster is not None:
            dask_client.close()
            dask_cluster.close()
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")
    return (
        msg,
        int_succeed,
        int_failed,
        pol_succeed,
        pol_failed,
        avg_int_DR,
        avg_pol_DR,
        max_int_DR,
        max_pol_DR,
        total_disk_detected_ms,
        total_non_disk_detected_ms,
    )


def cli():
    parser = argparse.ArgumentParser(
        description="Self-calibration", formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required positional argument)",
    )
    basic_args.add_argument(
        "metafits",
        type=str,
        help="Metafits file",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Caltable directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration and imaging parameters\n###################"
    )
    adv_args.add_argument(
        "--no_cal_applied",
        action="store_false",
        dest="cal_applied",
        help="Basic calibration is not applied",
    )
    adv_args.add_argument(
        "--start_thresh",
        type=float,
        default=5,
        help="Starting CLEANing threshold",
        metavar="Float",
    )
    adv_args.add_argument(
        "--stop_thresh",
        type=float,
        default=3,
        help="Stop CLEANing threshold",
        metavar="Float",
    )
    adv_args.add_argument(
        "--max_iter",
        type=int,
        default=30,
        help="Maximum number of selfcal iterations (in each mode, phaseonly, amplitude-phase, polselfcal)",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--max_DR",
        type=float,
        default=100000,
        help="Maximum dynamic range",
        metavar="Float",
    )
    adv_args.add_argument(
        "--intselfcal_min_iter",
        type=int,
        default=3,
        help="Minimum number of intensity selfcal iterations",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--polselfcal_min_iter",
        type=int,
        default=3,
        help="Minimum number of polarisation selfcal iterations",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--conv_frac",
        type=float,
        default=0.1,
        help="Fractional change in DR to determine convergence",
        metavar="Float",
    )
    adv_args.add_argument(
        "--int_solint",
        type=str,
        default="60s",
        help="Solution interval for gain calibration",
    )
    adv_args.add_argument(
        "--pol_solint",
        type=str,
        default="240s",
        help="Solution interval for polarisation calibration",
    )
    adv_args.add_argument(
        "--uvrange",
        type=str,
        default="",
        help="Calibration UV-range (CASA format)",
    )
    adv_args.add_argument(
        "--minuv_l",
        type=float,
        default=0,
        help="Minimum UV-lambda used for imaging",
        metavar="Float",
    )
    adv_args.add_argument("--weight", type=str, default="briggs", help="Imaging weight")
    adv_args.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust parameter for briggs weight",
        metavar="Float",
    )
    adv_args.add_argument(
        "--applymode",
        type=str,
        default="calonly",
        help="Solution apply mode",
        metavar="String",
    )
    adv_args.add_argument(
        "--min_tol_factor",
        type=float,
        default=10.0,
        help="Minimum tolerable variation in temporal direction in percentage",
        metavar="Float",
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of self-calibration rounds",
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose logs")
    adv_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    (
        msg,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = main(
        mslist=args.mslist,
        metafits=args.metafits,
        workdir=args.workdir,
        cal_applied=args.cal_applied,
        caldir=args.caldir,
        start_thresh=args.start_thresh,
        stop_thresh=args.stop_thresh,
        max_iter=args.max_iter,
        max_DR=args.max_DR,
        intselfcal_min_iter=args.intselfcal_min_iter,
        polselfcal_min_iter=args.polselfcal_min_iter,
        conv_frac=args.conv_frac,
        int_solint=args.int_solint,
        pol_solint=args.pol_solint,
        uvrange=args.uvrange,
        minuv_l=args.minuv_l,
        weight=args.weight,
        robust=args.robust,
        applymode=args.applymode,
        min_tol_factor=args.min_tol_factor,
        keep_backup=args.keep_backup,
        verbose=args.verbose,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
    )
    return msg
