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
    do_flag_backup,
)
from paircars.utils.imaging import (
    calc_field_of_view,
    calc_cellsize,
    get_fft_size,
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
)
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.resource_utils import drop_cache, limit_threads
from paircars.utils.selfcal_utils import (
    quiet_sun_selfcal,
    selfcal_round,
    flag_non_disk,
    do_uvsub_flag,
)
from paircars.utils.udocker_utils import (
    check_udocker_container,
    initialize_wsclean_container,
    initialize_quartical_container,
)

logging.getLogger("distributed").setLevel(logging.ERROR)
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
    uvrange="",
    minuv=0,
    solint="60s",
    weight="briggs",
    robust=0.0,
    do_apcal=True,
    min_tol_factor=1.0,
    applymode="calonly",
    solar_selfcal=True,
    use_solarflagger=False,
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
    uvrange : str, optional
        UV-range for calibration
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    solint : str, optional
        Solutions interval
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    do_apcal : bool, optional
        Perform ap-selfcal or not
    min_tol_factor : float, optional
         Minimum tolerable variation in temporal direction in percentage
    applymode : str, optional
        Solution apply mode
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    use_solarflagger : bool, optional
        Use solar flagging or not
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
        Whether non-disk data chunks flagging was successful or not
    float
        Final dynamic range
    """
    ncpu = max(1, ncpu)
    mem = abs(mem)

    limit_threads(n_threads=ncpu)
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
                f"Removing pre-existing intensity selfcal directory: {selfcaldir}"
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
                    "Calibration solutions were not applied and target metafits is also not supplied. Provide any one of them."
                )
                return 1, msname, [], False, 0
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
            intlogger.info(f"Spliting corrected data to ms : {selfcalms}")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="corrected",
                )
        else:
            intlogger.info(f"Spliting data to ms : {selfcalms}")
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
        intlogger.info("Checking initial flagging...")
        unflag_chans, flag_chans = get_chans_flag(msname)
        if len(unflag_chans) > 0:
            temp_ms = f"{msname}.tempsplit"
            unflag_chans = [f"{i}" for i in unflag_chans]
            unflag_spw = f"0:{';'.join(unflag_chans)}"
            intlogger.info(f"Spliting only unflagged spectral window: {unflag_spw}")
            split(vis=msname, outputvis=temp_ms, datacolumn="all", spw=unflag_spw)
            os.system(f"rm -rf {msname} {msname}.flagversions")
            os.system(f"mv {temp_ms} {msname}")

        ############################################
        # Imaging and calibration parameters
        ############################################
        intlogger.info("Estimating imaging Parameters.")
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

        ############################################
        # Initiating selfcal Parameters
        ############################################
        intlogger.info("Estimating self-calibration parameters.")
        DR1 = 0.0
        DR2 = 0.0
        DR3 = 0.0
        RMS1 = -1.0
        RMS2 = -1.0
        RMS3 = -1.0
        num_iter = 0
        num_iter_after_ap = 0
        num_iter_fixed_sigma = 0
        num_iter_after_flag = 0
        last_sigma_DR1 = 0
        sigma_reduced_count = 0
        calmode = "p"
        threshold = start_threshold
        last_round_gaintable = []
        last_round_ms = ""
        use_previous_model = False
        nondisk_flag = True
        min_DR = 0
        issue_occured = False
        min_iter = max(3, min_iter)  # Minimum 3 iterations
        do_flag = False
        restore_flag = True
        os.system("rm -rf *_selfcal_present*")

        ###########################################
        # Starting using Gaussian model
        ###########################################
        intlogger.info("Starting self-calibration using Gaussian source model.")
        msg, _ = quiet_sun_selfcal(
            msname, intlogger, selfcaldir, refant=str(refant), solint=solint
        )
        if msg == 0:
            intlogger.info(
                "Starting self-calibration using Gaussian model is successful."
            )
        else:
            if msg == 2:
                nondisk_flag = False
            intlogger.warning(
                "Starting self-calibration using Gaussian model is not successful."
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
            intlogger.info("######################################")
            if num_iter_after_flag > 0 and do_flag:
                do_flag = False
                if (
                    DR3 < 0.85 * DR2 and DR3 < 0.9 * DR2
                ):  # If DR is decreasing, restore flags
                    restore_flag = True
                    min_iter += 1
                    intlogger.warning(
                        "DR is decreaging after flagging. Restoring previous uv-domain flags."
                    )
                else:
                    restore_flag = False
            (
                msg,
                gaintable,
                dyn,
                rms,
                final_image,
                final_model,
                final_residual,
                _,
                max_pixel_offset,
            ) = selfcal_round(
                msname,
                metafits,
                intlogger,
                selfcaldir,
                cellsize,
                imsize,
                round_number=num_iter,
                uvrange=uvrange,
                minuv=minuv,
                calmode=calmode,
                solint=solint,
                refant=str(refant),
                applymode=applymode,
                min_tol_factor=min_tol_factor,
                threshold=threshold,
                use_previous_model=use_previous_model,
                weight=weight,
                robust=robust,
                use_solar_mask=solar_selfcal,
                fluxscale_mwa=fluxscale_mwa,
                do_intensity_cal=True,
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
                        "No model flux is picked up in first round. Trying with lowest threshold.\n"
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
                        max_pixel_offset,
                    ) = selfcal_round(
                        msname,
                        metafits,
                        intlogger,
                        selfcaldir,
                        cellsize,
                        imsize,
                        round_number=num_iter,
                        uvrange=uvrange,
                        minuv=minuv,
                        calmode=calmode,
                        solint=solint,
                        refant=str(refant),
                        applymode=applymode,
                        min_tol_factor=min_tol_factor,
                        threshold=end_threshold,
                        use_previous_model=False,
                        weight=weight,
                        robust=robust,
                        use_solar_mask=solar_selfcal,
                        fluxscale_mwa=fluxscale_mwa,
                        do_intensity_cal=True,
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
                        clean_shutdown(sub_observer)
                        return msg, msname, [], nondisk_flag, 0
                    else:
                        threshold = end_threshold
                else:
                    os.system("rm -rf *_selfcal_present*")
                    return msg, msname, [], nondisk_flag, 0
            elif msg > 1:
                intlogger.error("Self-calibration failed.")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return msg, msname, [], nondisk_flag, 0
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
            intlogger.info("######################################")
            intlogger.info(
                "RMS based dynamic ranges: "
                + str(DR1)
                + ","
                + str(DR2)
                + ","
                + str(DR3)
            )
            intlogger.info(
                "RMS of the images: " + str(RMS1) + "," + str(RMS2) + "," + str(RMS3)
            )
            intlogger.info(f"Maximum pixel offset: {max_pixel_offset}")
            if DR3 > 1.1 * DR2 and (
                calmode == "p" or (calmode == "ap" and num_iter_after_ap > 1)
            ):
                use_previous_model = True
            else:
                use_previous_model = False

            #################################
            # Checking DR decrease conditions
            #################################
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
            cond2 = (
                (DR3 < 0.7 * DR2 or DR3 < 0.9 * min_DR)
                and calmode == "ap"
                and num_iter_after_ap > 1
            )
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
                        "Dynamic range decreasing in phase-only self-cal."
                    )
                if cond2:
                    intlogger.warning(
                        "Dynamic range dropped suddenly or drop below starting dynamic range."
                    )
                if cond3:
                    intlogger.warning(
                        "Dynamic range is decreasing after minimum numbers of 'ap' round.\n"
                    )
                ##################################################
                # Performing steps
                ##################################################
                if do_apcal and calmode == "p":
                    intlogger.info("Changed calmode to 'ap'.")
                    calmode = "ap"
                    use_previous_model = False
                elif calmode == "ap" and threshold > end_threshold:
                    threshold -= 1
                    intlogger.info(f"Reducing threshold to: {threshold}")
                else:
                    if not use_solarflagger and DR3 < 100:
                        use_solarflagger = True
                    if use_solarflagger and not do_flag and num_iter_after_flag == 0:
                        intlogger.info("Trying uvsub flagging.")
                        do_flag = True
                        num_iter_after_flag += 1
                        use_previous_model = False
                    else:
                        intlogger.warning(
                            "Stopping self-calibration. Using last round caltable as final.\n"
                        )
                        if not use_solarflagger and DR3 < 100:
                            intlogger.info(
                                "Performing final flagging because DR is less than 100."
                            )
                            do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=ncpu)
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        clean_shutdown(sub_observer)
                        return 0, msname, last_round_gaintable, nondisk_flag, DR2

            ###########################
            # If maximum DR has reached
            ###########################
            if DR3 > max_DR and num_iter_after_ap > min_iter:
                intlogger.info("Maximum dynamic range is reached.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, msname, gaintable, nondisk_flag, DR3

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
                        "Starting final self-calibration rounds with threshold = "
                        + str(end_threshold)
                        + "sigma...\n"
                    )
                    threshold = end_threshold
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                else:
                    if not use_solarflagger and DR3 < 100:
                        intlogger.info(
                            "Performing final flagging because DR is less than 100."
                        )
                        do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=ncpu)
                    intlogger.info("Selfcal calibration has converged.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, msname, gaintable, nondisk_flag, DR3
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
                        use_previous_model = False
                    ######################################
                    # Reducing threshold if already in apcal
                    ######################################
                    elif (do_apcal and num_iter_after_ap > min_iter) or not do_apcal:
                        threshold -= 1
                        intlogger.info("Reducing threshold to : " + str(threshold))
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
                    if not use_solarflagger and DR3 < 100:
                        intlogger.info(
                            "Performing final flagging because DR is less than 100."
                        )
                        do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=ncpu)
                    intlogger.info("Self-calibration has converged.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, msname, gaintable, nondisk_flag, DR3
                #########################################
                # In apcal and maximum iteration has reached
                #########################################
                elif num_iter > min_iter and (
                    (not do_apcal and num_iter == max_iter)
                    or (do_apcal and calmode == "ap" and num_iter_after_ap == max_iter)
                ):
                    if not use_solarflagger and DR3 < 100:
                        intlogger.info(
                            "Performing final flagging because DR is less than 100."
                        )
                        do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=ncpu)
                    intlogger.info(
                        "Self-calibration is finished. Maximum iteration is reached.\n"
                    )
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, msname, gaintable, nondisk_flag, DR3
            num_iter += 1
            os.system(f"cp -r {msname} {msname}.round{num_iter}")
            if calmode == "ap":
                num_iter_after_ap += 1
            if num_iter_after_ap > 0:
                if not use_solarflagger and DR3 < 100:
                    use_solarflagger = True
                if (
                    use_solarflagger
                    and not do_flag
                    and num_iter_after_flag == 0
                    and num_iter_after_ap == 0
                ):
                    intlogger.info("Trying uvsub flagging.")
                    do_flag = True
                    num_iter_after_flag += 1
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
    min_iter=5,
    threshold=3.0,
    DR_convergence_frac=0.1,
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    solar_selfcal=True,
    use_solarflagger=False,
    try_nondisk_flag=True,
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
    DR_convergence_frac : float, optional
        Dynamic range fractional change to consider as converged
    uvrange : str, optional
        UV-range for calibration
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    use_solarflagger : bool, optional
        Use solar flagger or not
    try_nondisk_flag : bool, optional
        Try to flag non-disk data chunks or not
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

    limit_threads(n_threads=ncpu)
    from casatasks import split, flagdata, flagmanager

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
                f"Removing pre-existing polarisation selfcal directory: {selfcaldir}"
            )
            os.system(f"rm -rf {selfcaldir}")
        os.makedirs(selfcaldir, exist_ok=True)
        os.chdir(selfcaldir)
        selfcalms = selfcaldir + "/polselfcal_" + os.path.basename(msname)
        if os.path.exists(selfcalms):
            os.system(f"rm -rf {selfcalms}")
        if os.path.exists(f"{selfcalms}.flagversions"):
            os.system(f"rm -rf {selfcalms}.flagversions")

        ################################
        # Trying to flag non-disk chunks
        ################################
        split_spw = ""
        if try_nondisk_flag:
            pollogger.info(
                "Non-disk data chunk flagging was successful during intensity self-calibration. Trying before polarisation self-calibration."
            )
            do_flag_backup(msname, flagtype="nondisk")
            result = flag_non_disk(msname)
            if result == 0:
                pollogger.info("Flagged non-disk data chunks.")
                unflag_chans, flag_chans = get_chans_flag(msname)
                split_spw = "0:" + ";".join([str(i) for i in unflag_chans])
            else:
                pollogger.warning("Could not flag non-disk data chunks.")
                split_spw = ""
            with suppress_output():
                if result != 0:
                    flagmanager(vis=msname, mode="restore", versionname="nondisk_1")
                flagmanager(vis=msname, mode="delete", versionname="nondisk_1")

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
            pollogger.info(f"Spliting corrected data to ms : {selfcalms}")
            with suppress_output():
                split(
                    vis=msname,
                    spw=split_spw,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="corrected",
                )
        else:
            pollogger.warning("Corrected data column is not present.")
            pollogger.info(f"Spliting data to ms : {selfcalms}")
            with suppress_output():
                split(
                    vis=msname,
                    spw=split_spw,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="data",
                )
        msname = selfcalms

        ################################################################
        # Initial flagging -- zeros, extreme bad data
        ################################################################
        pollogger.info("Checking initial flagging.")
        with suppress_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="data",
                flagbackup=False,
            )
        unflag_chans, flag_chans = get_chans_flag(msname)
        if len(unflag_chans) > 0:
            temp_ms = f"{msname}.tempsplit"
            unflag_chans = [f"{i}" for i in unflag_chans]
            unflag_spw = f"0:{';'.join(unflag_chans)}"
            pollogger.info(f"Spliting only unflagged spectral window: {unflag_spw}")
            split(vis=msname, outputvis=temp_ms, datacolumn="all", spw=unflag_spw)
            os.system(f"rm -rf {msname} {msname}.flagversions")
            os.system(f"mv {temp_ms} {msname}")

        ############################################
        # Imaging and calibration parameters
        ############################################
        pollogger.info("Estimating imaging Parameters.")
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

        ############################################
        # Initiating selfcal Parameters
        ############################################
        pollogger.info("Estimating self-calibration parameters.")
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
        calc_chunks = True
        do_flag = False
        restore_flag = True
        num_iter_after_flag = 0
        last_round_gaintable = []
        last_leakage_file = ""
        last_round_ms = ""
        solve_array_leakage = True
        issue_occured = False
        num_iter_after_reset = 0
        min_iter = max(5, min_iter)  # Minimum 5 iterations
        os.system("rm -rf *_selfcal_present*")

        ##########################################
        # Starting selfcal loops
        ##########################################
        while True:
            issue_occured = False  # Reseting in every round
            ##################################
            # Selfcal round parameters
            ##################################
            pollogger.info("######################################")
            pollogger.info("Selfcal iteration : " + str(num_iter))
            pollogger.info("######################################")
            if num_iter == 0:
                pbcor = True
                leakagecor = True
                pbuncor = False
            elif num_iter < 3:
                pbcor = False
                leakagecor = True
                pbuncor = False
            elif num_iter == 3:
                pbcor = False
                leakagecor = True
                pbuncor = True
            else:
                if num_iter == min_iter:
                    solve_array_leakage = False  # This is to make sure if it failed, last round ms has same state of polcal
                pbcor = True
                leakagecor = True
                pbuncor = True

            if num_iter_after_flag > 0 and do_flag:
                do_flag = False
                if (
                    DR3 < 0.9 * DR2
                ):  # If DR after flagging is smaller than 90% of last round DR, restore flags
                    pollogger.warning(
                        "Restoring previous uv-bin flags because dynamic range did not improve."
                    )
                    restore_flag = True
                    min_iter += 1
                else:
                    restore_flag = False
            (
                msg,
                gaintable,
                dyn,
                rms,
                final_image,
                final_model,
                final_residual,
                leakage_info,
                max_pixel_offset,
            ) = selfcal_round(
                msname,
                metafits,
                pollogger,
                selfcaldir,
                cellsize,
                imsize,
                round_number=num_iter,
                uvrange=uvrange,
                minuv=minuv,
                calc_chunks=calc_chunks,
                refant=str(refant),
                threshold=threshold,
                weight=weight,
                robust=robust,
                use_solar_mask=solar_selfcal,
                do_polcal=True,
                do_intensity_cal=False,
                pbcor=pbcor,
                leakagecor=leakagecor,
                pbuncor=pbuncor,
                do_flag=do_flag,
                restore_flag=restore_flag,
                solve_array_leakage=solve_array_leakage,
                ncpu=ncpu,
                mem=round(mem, 2),
            )
            if msg == 1:
                pollogger.error("No model flux is picked up.")
                os.system("rm -rf *_selfcal_present*")
                return msg, msname, [], "", 0
            elif msg > 2:
                pollogger.error("Polarisation self-calibration failed.")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return msg, msname, [], "", 0
            elif msg == 2:
                if calc_chunks is False:
                    if num_iter > min_iter:
                        pollogger.warning(
                            "Minor issues in polarisation self-calibration model prediction. Stopped at previous round."
                        )
                        if os.path.exists(last_round_ms):
                            os.system(f"rm -rf {msname}")
                            os.system(f"cp -r {last_round_ms} {msname}")
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        clean_shutdown(sub_observer)
                        return 0, msname, last_round_gaintable, last_leakage_file, DR2
                    else:
                        issue_occured = True
                        pollogger.error(
                            "Minor issues in polarisation self-calibration model prediction. Minimum iteration has not covered."
                        )
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        clean_shutdown(sub_observer)
                        return msg, msname, [], "", 0
                else:
                    issue_occured = True
                    pollogger.warning(
                        "Minor issues in polarisation self-calibration model prediction. Retrying with entire spectro-temporal chunks."
                    )
                    calc_chunks = False
            else:
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
                leakage_file = f"{gaintable[0].split('.dcal')[0]}.leakage.npy"
                np.save(
                    leakage_file, [q_leakage, u_leakage, v_leakage, q_err, u_err, v_err]
                )
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
                pollogger.info("######################################")
                pollogger.info(
                    "RMS based dynamic ranges: "
                    + str(DR1)
                    + ","
                    + str(DR2)
                    + ","
                    + str(DR3)
                )
                pollogger.info(
                    "RMS of the images: "
                    + str(RMS1)
                    + ","
                    + str(RMS2)
                    + ","
                    + str(RMS3)
                )
                pollogger.info(
                    f"Stokes I to Q leakage: {round(QL1*100.0,3)}, {round(QL2*100.0,3)}, {round(QL3*100.0,3)}%."
                )
                pollogger.info(
                    f"Stokes I to U leakage: {round(UL1*100.0,3)}, {round(UL2*100.0,3)}, {round(UL3*100.0,3)}%."
                )
                pollogger.info(
                    f"Stokes I to V leakage: {round(VL1*100.0,3)}, {round(VL2*100.0,3)}, {round(VL3*100.0,3)}%."
                )
                pollogger.info(f"Maximum pixel offset: {max_pixel_offset}")
                leakage_converged = (QL3 == 0.0 and UL3 == 0.0 and VL3 == 0.0) or (
                    (QL2 - QL3) <= 0.01 and (UL2 - UL3) <= 0.01 and (VL2 - VL3) <= 0.01
                )

                ########################################
                # Leakage pr big DR related issues
                #########################################
                ###################################################################
                # Condition 1: If solving per antenna decrease DR, solve per array
                ###################################################################
                if not solve_array_leakage and (DR3 < 0.9 * DR2 or RMS3 > 1.1 * RMS2):
                    pollogger.warning(
                        "Solving over array instead of antenna, as DR decreases."
                    )
                    solve_array_leakage = True
                    issue_occured = True
                    num_iter_after_reset = 0
                    if os.path.exists(last_round_ms):
                        pollogger.info("Replacing with previous measurement set.")
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")
                ##########################################
                # Condition 2: If leakage increased
                ##########################################
                if (num_iter == 2 or num_iter > 4) and (
                    (abs(QL3) - abs(QL2)) > 0.1
                    or (abs(UL3) - abs(UL2)) > 0.1
                    or (abs(VL3) - abs(VL2)) > 0.1
                ):
                    issue_occured = True
                    pollogger.warning("Leakage increased by 10%.")
                    if os.path.exists(last_round_ms):
                        pollogger.info("Replacing with previous measurement set.")
                        num_iter -= 1
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")

                #########################################
                # Condition 3: If leakage becomes nan
                #########################################
                if np.isnan(QL3) or np.isnan(UL3) or np.isnan(VL3):
                    issue_occured = True
                    if num_iter == 0:
                        pollogger.error(
                            "Leakages become nan. Serious calibration issue occured at the first round."
                        )
                        return 1, msname, [], "", 0
                    pollogger.warning(
                        "Leakages become nan. Serious calibration issue occured."
                    )
                    if os.path.exists(last_round_ms):
                        pollogger.info("Replacing with previous measurement set.")
                        num_iter -= 1
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")
                    if not solve_array_leakage:
                        num_iter_after_reset = 0
                        pollogger.info("Solving over array instead of antenna.")
                        solve_array_leakage = True
                    else:
                        if not use_solarflagger and DR3 < 100:
                            use_solarflagger = True
                        if (
                            use_solarflagger
                            and not do_flag
                            and num_iter_after_flag == 0
                        ):
                            pollogger.info("Trying uvsub flagging.")
                            do_flag = True
                            restore_flag = False
                        else:
                            os.system("rm -rf *_selfcal_present*")
                            time.sleep(5)
                            clean_shutdown(sub_observer)
                            if num_iter > min_iter:
                                pollogger.warning(
                                    "Stopping self-calibration. Using last round caltables."
                                )
                                return (
                                    0,
                                    msname,
                                    last_round_gaintable,
                                    last_leakage_file,
                                    DR2,
                                )
                            else:
                                pollogger.error(
                                    "Leakages become nan. Serious calibration issue occured at the before completing minimum rounds."
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
                            f"Dynamic range decreased below start dynamic range: {min_DR}."
                        )
                    if cond2:
                        pollogger.warning(
                            "Dynamic range is decreasing after minimum numbers of rounds."
                        )
                    if cond3:
                        pollogger.warning(
                            "Dynamic range dropped suddenly. Using last round caltable as final."
                        )
                    ###################################
                    # Replacing previous ms
                    ###################################
                    issue_occured = True
                    if os.path.exists(last_round_ms):
                        pollogger.info("Replacing with previous measurement set.")
                        os.system(f"rm -rf {msname}")
                        os.system(f"cp -r {last_round_ms} {msname}")
                    if not solve_array_leakage:
                        num_iter_after_reset = 0
                        pollogger.info("Solving over array instead of antenna.")
                        solve_array_leakage = True
                    else:
                        if not use_solarflagger and DR3 < 100:
                            use_solarflagger = True
                        if (
                            use_solarflagger
                            and not do_flag
                            and num_iter_after_flag == 0
                        ):
                            pollogger.info("Trying uvsub flagging.")
                            do_flag = True
                            num_iter_after_flag += 1
                        else:
                            if num_iter > min_iter:
                                pollogger.warning(
                                    "Stopping self-calibration. Using last round caltables."
                                )
                                os.system("rm -rf *_selfcal_present*")
                                time.sleep(5)
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
                                    "Encountered this error before minimum number of rounds."
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
        clean_shutdown(sub_observer)
        return 1, msname, [], "", 0


def do_full_selfcal(
    msname="",
    workdir="",
    selfcaldir="",
    metafits="",
    cal_applied=True,
    start_threshold=5,
    end_threshold=3,
    max_iter=30,
    max_DR=100000,
    intselfcal_min_iter=3,
    polselfcal_min_iter=5,
    DR_convergence_frac=0.1,
    uvrange="",
    minuv=0,
    solint="60s",
    weight="briggs",
    robust=0.0,
    do_apcal=True,
    do_polcal=True,
    min_tol_factor=1.0,
    applymode="calonly",
    solar_selfcal=True,
    use_solarflagger=False,
    ncpu=1,
    mem=1,
    logfile_prefix="selfcal",
    logger=None,
):
    """
    Perform both intensity and polarisation self-calibration

    Returns
    -------
    int
        Intensity selfcal success message
    int
        Polarisation selfcal success message
    list
        Intensity selfcal gaintables
    list
        Polarisation selfcal gaintables
    str
        Leakage information file
    float
        Final intensity selfcal dynamic range
    float
        Final polarisation selfcal dynamic range
    """
    if logger is None:
        logger = get_logger_safe()

    ncpu = max(1, ncpu)
    mem = abs(mem)

    selfcaldir = selfcaldir.rstrip("/")
    logfile_prefix = logfile_prefix.rstrip("/")
    logger.info(f"Starting intensity self-calibration for ms: {msname}.")
    unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
    msmd = msmetadata()
    msmd.open(msname)
    refant_ids = sorted(
        [msmd.antennaids(antname)[0] for antname in unflagged_antenna_names]
    )[0]
    refant = str(refant_ids)
    msmd.close()

    intensity_selfcal_msg, selfcal_ms, gaintable, try_nondisk_flag, int_DR = do_selfcal(
        msname=msname,
        workdir=workdir,
        selfcaldir=f"{selfcaldir}_int",
        metafits=metafits,
        cal_applied=cal_applied,
        refant=str(refant),
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        max_iter=max_iter,
        max_DR=max_DR,
        min_iter=max(3, intselfcal_min_iter),
        DR_convergence_frac=DR_convergence_frac,
        uvrange=uvrange,
        minuv=minuv,
        solint=solint,
        weight=weight,
        robust=robust,
        do_apcal=do_apcal,
        min_tol_factor=min_tol_factor,
        applymode=applymode,
        solar_selfcal=solar_selfcal,
        use_solarflagger=use_solarflagger,
        ncpu=ncpu,
        mem=mem,
        logfile=f"{logfile_prefix}_int.log",
    )
    if intensity_selfcal_msg != 0:
        logger.error("Error occured in intensity self-calibration.")
        return intensity_selfcal_msg, 1, [], [], "", int_DR, 0
    elif do_polcal is False:
        logger.info("Polarisation calibration is not requested.")
        return 0, 1, gaintable, [], "", int_DR, 0
    else:
        logger.info(f"Starting polarisation self-calibration for ms: {msname}.")
        pol_selfcal_msg, pol_selfcal_ms, quartical_table, leakage_file, pol_DR = (
            do_polselfcal(
                msname=selfcal_ms,
                workdir=workdir,
                selfcaldir=f"{selfcaldir}_pol",
                metafits=metafits,
                max_iter=max(10, int(max_iter / 3)),
                max_DR=max_DR,
                min_iter=max(5, polselfcal_min_iter),
                threshold=end_threshold,
                DR_convergence_frac=DR_convergence_frac,
                uvrange=uvrange,
                minuv=minuv,
                weight=weight,
                robust=robust,
                solar_selfcal=solar_selfcal,
                use_solarflagger=use_solarflagger,
                try_nondisk_flag=try_nondisk_flag,
                ncpu=ncpu,
                mem=mem,
                logfile=f"{logfile_prefix}_pol.log",
            )
        )
        return (
            intensity_selfcal_msg,
            pol_selfcal_msg,
            gaintable,
            quartical_table,
            leakage_file,
            int_DR,
            pol_DR,
        )


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
    polselfcal_min_iter=5,
    conv_frac=0.1,
    solint="60s",
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    applymode="calonly",
    min_tol_factor=1.0,
    do_apcal=True,
    do_polcal=True,
    solar_selfcal=True,
    use_solarflagger=False,
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
        Minimum number of iterations before checking for convergence for polarisation selfcal. Default is 5.
    conv_frac : float, optional
        Convergence criterion: fractional change in dynamic range below which iteration stops. Default is 0.1.
    solint : str, optional
        Solution interval for gain calibration (e.g., "inf", "10s", "int"). Default is "60s".
    uvrange : str, optional
        UV range to be used for imaging and calibration, in CASA format. Default is "" (all baselines).
    minuv : float, optional
        Minimum baseline length (in wavelengths) to include. Default is 10.
    weight : str, optional
        Weighting scheme for imaging (e.g., "natural", "uniform", "briggs"). Default is "briggs".
    robust : float, optional
        Robustness parameter for Briggs weighting (ignored if not using "briggs"). Default is 0.0.
    applymode : str, optional
        Apply mode for calibration tables ("calonly", "calflag", etc.). Default is "calonly".
    min_tol_factor : float, optional
        Minimum factor for tolerance comparison during convergence checks. Default is 1.0.
    do_apcal : bool, optional
        Whether to apply polarization and bandpass calibration before starting selfcal. Default is True.
    do_polcal : bool, optional
        Whether perform polarisation self-calibration or not
    solar_selfcal : bool, optional
        If True, uses solar-specific masking and flux normalization. Default is True.
    use_solarflagger : bool, optional
        Use solar flagger or not. Default is False.
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
    logger.debug(f"Current working directory: {os.getcwd()}")

    if caldir == "" or not os.path.exists(caldir):
        caldir = f"{workdir}/caltables"
    os.makedirs(caldir, exist_ok=True)
    logger.debug(f"Output caltables directory: {caldir}.")

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
    if observer is None:
        logger.info(
            "Remote link or jobname is blank. Not transmiting to remote logger."
        )

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        int_succeed = 0
        int_failed = len(mslist)
        pol_succeed = 0
        pol_failed = len(mslist)
        avg_int_DR = 0
        avg_pol_DR = 0
        max_int_DR = 0
        max_pol_DR = 0

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            max_worker=len(mslist) + 1,
        )
        if dask_client is None:
            logger.critical("Error occured in creating local cluster.")
            return 1, 0, 0, 0, 0, 0, 0, 0, 0
        scale_worker_and_wait(dask_cluster, dask_client, nworker)

    ###########################
    # WSClean container
    ###########################
    container_name = "paircarswsclean"
    container_present = check_udocker_container(container_name)
    if not container_present:
        logger.debug(f"Initializing {container_name}.")
        container_name = initialize_wsclean_container(name=container_name, verbose=True)
        if container_name is None:
            logger.critical(
                f"Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1, int_succeed, int_failed, pol_succeed, pol_failed, 0, 0, 0, 0

    if do_polcal:
        container_name = "paircarsquartical"
        container_present = check_udocker_container(container_name)
        if not container_present:
            logger.debug(f"Initializing {container_name}.")
            container_name = initialize_quartical_container(
                name=container_name, verbose=True
            )
            if container_name is None:
                logger.critical(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1, int_succeed, int_failed, pol_succeed, pol_failed, 0, 0, 0, 0

    try:
        for banner in print_banner(
            "Starting self-calibrations.", no_print=True
        ).splitlines():
            logger.info(banner)

        if do_polcal and do_apcal is False:
            logger.info(
                "Polarisation self-calibration is requested without amplitude-phase intensity self-calibration. Switching off polarisation self-calibration."
            )
            do_polcal = False
        header = fits.getheader(metafits)
        obsid = header["GPSTIME"]
        partial_do_selfcal = partial(
            do_full_selfcal,
            metafits=str(metafits),
            cal_applied=bool(cal_applied),
            start_threshold=float(start_thresh),
            end_threshold=float(stop_thresh),
            max_iter=int(max_iter),
            max_DR=float(max_DR),
            intselfcal_min_iter=int(intselfcal_min_iter),
            polselfcal_min_iter=int(polselfcal_min_iter),
            DR_convergence_frac=float(conv_frac),
            uvrange=str(uvrange),
            minuv=float(minuv),
            solint=str(solint),
            weight=str(weight),
            robust=float(robust),
            do_apcal=do_apcal,
            do_polcal=do_polcal,
            applymode=applymode,
            min_tol_factor=float(min_tol_factor),
            solar_selfcal=solar_selfcal,
            use_solarflagger=use_solarflagger,
            logger=logger,
        )

        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                logger.warning(f"Issue in : {ms}")
                os.system(f"rm -rf {ms}")
        mslist = filtered_mslist
        if len(mslist) == 0:
            logger.critical("No filtered ms to continue.")
            return 1, int_succeed, int_failed, pol_succeed, pol_failed, 0, 0, 0, 0

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
        logger.info("#################################")

        os.makedirs(f"{workdir}/logs", exist_ok=True)
        tasks = []
        for ms in mslist:
            obsid = get_MWA_OBSID(ms)
            coarse_chan = get_MWA_coarse_chan(ms)
            if len(coarse_chan) > 1:
                coarse_chan = f"{min(coarse_chan)}-{max(coarse_chan)}"
            else:
                coarse_chan = f"{min(coarse_chan)}"
            logfile_prefix = f"{workdir}/logs/selfcal_{obsid}_ch_{coarse_chan}"
            logger.info(f"Measurement set name: {ms}.")
            logger.info(f"Self-cal log file: {logfile_prefix}_int.log")
            if do_polcal:
                logger.info(f"Polarisation self-cal log file: {logfile_prefix}_pol.log")
            tasks.append(
                delayed(partial_do_selfcal)(
                    ms,
                    workdir,
                    workdir + "/" + os.path.basename(ms).split(".ms")[0] + "_selfcal",
                    ncpu=n_threads,
                    mem=mem_limit,
                    logfile_prefix=logfile_prefix,
                )
            )
        logger.info("Starting all self-calibration.")
        results = list(dask_client.gather(dask_client.compute(tasks)))

        gcal_list = []
        bpass_list = []
        dcal_list = []
        leakage_file_list = []
        succeed_intselfcal = 0
        failed_intselfcal = 0
        succeed_polselfcal = 0
        failed_polselfcal = 0
        int_DR_list = []
        pol_DR_list = []
        for i in range(len(results)):
            r = results[i]
            int_msg = r[0]
            int_DR = r[5]
            pol_DR = r[6]
            int_DR_list.append(int_DR)
            pol_DR_list.append(pol_DR)
            if int_msg != 0:
                logger.error(
                    f"Intensity self-calibration was not successful for ms: {mslist[i]}."
                )
                os.system(
                    f"touch {workdir}/.intselfcal_failed_{os.path.basename(mslist[i])}"
                )
                failed_intselfcal += 1
            else:
                try:
                    gaintables = r[2]
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
                        f"touch {workdir}/.intselfcal_succeed_{os.path.basename(mslist[i])}"
                    )
                    succeed_intselfcal += 1
                except Exception:
                    logger.exception(
                        "Exception occured in filtering intensity self-calibration caltables.",
                        exc_info=True,
                    )
                    os.system(
                        f"touch {workdir}/.intselfcal_failed_{os.path.basename(mslist[i])}"
                    )
                    failed_intselfcal += 1

            if do_polcal:
                pol_msg = r[1]
                if pol_msg != 0:
                    logger.error(
                        f"Polarisation self-calibration was not successful for ms: {mslist[i]}."
                    )
                    os.system(
                        f"touch {workdir}/.polselfcal_failed_{os.path.basename(mslist[i])}"
                    )
                    failed_polselfcal += 1
                else:
                    try:
                        leakage_file = r[4]
                        quartical_tables = r[3]
                        dcal = quartical_tables[0]
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
                            f"touch {workdir}/.polselfcal_succeed_{os.path.basename(mslist[i])}"
                        )
                        succeed_polselfcal += 1
                    except Exception:
                        logger.exception(
                            "Error occured in filtering polarisation self-calibration caltables.",
                            exc_info=True,
                        )
                        os.system(
                            f"touch {workdir}/.polselfcal_failed_{os.path.basename(mslist[i])}"
                        )
                        failed_polselfcal += 1
        if not keep_backup:
            for ms in mslist:
                int_selfcaldir = (
                    workdir
                    + "/"
                    + os.path.basename(ms).split(".ms")[0]
                    + "_selfcal_int"
                )
                os.system(f"rm -rf {int_selfcaldir}")
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
            if len(dcal_list) > 0:
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
        logger.info(
            f"Average polarisation self-calibration dynamic range: {avg_pol_DR}"
        )
    except Exception:
        logger.exception("Exception occured in self-calibration.", exc_info=True)
        msg = 1
        avg_int_DR = 0
        avg_pol_DR = 0
        max_int_DR = 0
        max_pol_DR = 0
    finally:
        time.sleep(5)
        clean_shutdown(observer)
        for ms in mslist:
            if os.path.exists(ms):
                drop_cache(ms)
        if dask_cluster is not None:
            dask_client.shutdown()
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
        default=5,
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
    adv_args.add_argument("--solint", type=str, default="60s", help="Solution interval")
    adv_args.add_argument(
        "--uvrange",
        type=str,
        default="",
        help="Calibration UV-range (CASA format)",
    )
    adv_args.add_argument(
        "--minuv",
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
        default=1.0,
        help="Minimum tolerable variation in temporal direction in percentage",
        metavar="Float",
    )
    adv_args.add_argument(
        "--no_apcal",
        action="store_false",
        dest="do_apcal",
        help="Do not perform ap-selfcal",
    )
    adv_args.add_argument(
        "--no_polcal",
        action="store_false",
        dest="do_polcal",
        help="Do not perform polarisation selfcal",
    )
    adv_args.add_argument(
        "--no_solar_selfcal",
        action="store_false",
        dest="solar_selfcal",
        help="Do not perform solar self-calibration",
    )
    adv_args.add_argument(
        "--use_solarflagger",
        action="store_true",
        help="Use solar flagger or not",
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

    msg, _, _, _, _, _, _, _, _ = main(
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
        solint=args.solint,
        uvrange=args.uvrange,
        minuv=args.minuv,
        weight=args.weight,
        robust=args.robust,
        applymode=args.applymode,
        min_tol_factor=args.min_tol_factor,
        do_apcal=args.do_apcal,
        do_polcal=args.do_polcal,
        solar_selfcal=args.solar_selfcal,
        use_solarflagger=args.use_solarflagger,
        keep_backup=args.keep_backup,
        verbose=args.verbose,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
    )
    return msg
