import sys
import traceback
import time
import glob
import os
import socket
import requests
import getpass
import contextlib
import numpy as np
import argparse
import logging
from casatools import msmetadata
from astropy.io import fits
from datetime import datetime as dt
from multiprocessing import Event
from dask.distributed import get_client
from prefect import flow
from prefect.context import get_run_context
from prefect_dask.task_runners import DaskTaskRunner
from prefect.settings import get_current_settings
from prefect.futures import wait
from paircars.utils.basic_utils import (
    internet_available,
    print_banner,
    get_gpstime_to_date,
)
from paircars.utils.calibration import (
    calc_bw_smearing_freqwidth,
    calc_time_smearing_timewidth,
    max_time_solar_smearing,
)
from paircars.utils.casatasks import reset_weights_and_flags
from paircars.utils.flagging import do_flag_backup
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    get_remote_logger_link,
    get_remote_logger_password,
    get_emails,
    init_logger,
    generate_password,
    get_logger_safe,
)
from paircars.utils.mwa_utils import (
    get_ncoarse,
    get_MWA_OBSID,
    download_MWA_metafits,
    get_MWA_coarse_chan,
)
from paircars.utils.proc_manage_utils import (
    get_jobid,
    save_main_process_info,
    scale_worker_and_wait,
    get_local_dask_cluster,
    get_scheduler_name,
)
from paircars.utils.resource_utils import drop_cache
from paircars.data.sendmail import (
    send_paircars_notification as send_notification,
)
from paircars.clusterutils.slurm_cluster import (
    get_slurm_dask_cluster,
    is_slurm_job,
)
from paircars.utils.prefect_logger_utils import (
    start_flow_log_saver,
)
from paircars.pipeline.init_data import init_paircars_data
from paircars.pipeline.tasks import run_ds_jobs
from paircars.pipeline.flows import (
    basic_cal_subflow,
    selfcal_subflow,
    applysol_subflow,
    imaging_subflow,
)
from paircars.pipeline.tasks import (
    send_task_notification,
    run_make_msplot,
)

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


@flow(
    name="P-AIRCARS Master control",
    version="3.0",
    description="Calibration and Imaging Pipeline for MWA Solar Observation",
    log_prints=True,
)
def master_control(
    target_datadir,
    workdir,
    outdir,
    # Metafits and calibrators
    target_metafits="",
    calibrator_datadir="",
    calibrator_metafits="",
    # Pre-calibration
    do_forcereset_weightflag=False,
    # Calibration
    do_basic_cal=True,
    do_polcal=False,
    do_applycal=True,
    only_amplitude=False,
    redo_basic_cal=False,
    use_uvbinflagger=True,
    # Target data preparation
    freqrange="",
    timerange="",
    uvrange="",
    # Self-calibration
    do_selfcal=True,
    do_apply_selfcal=True,
    do_ap_selfcal=True,
    int_solint="60s",
    pol_solint="240s",
    redo_selfcal=False,
    # Sidereal correction
    do_sidereal_cor=False,
    # Dynamic spectra
    make_ds=True,
    # Imaging
    do_imaging=True,
    weight="briggs",
    robust=0.0,
    minuv_l=10,
    image_freqres=1.28,
    image_timeres=10.0,
    pol="IQUV",
    clean_threshold=1.0,
    use_multiscale=True,
    cutout_rsun=10.0,
    make_overlay=False,
    make_TB=False,
    save_hpc=True,
    make_msplot=False,
    # Compress image
    compress_image=False,
    keep_original=False,
    # Resource settings
    cpu_frac=0.8,
    mem_frac=0.8,
    max_worker=2,
    keep_backup=False,
    keep_calibrated_ms=False,
    # Remote logging
    remote_logger=False,
    jobid=None,
    job_password=None,
    adaptive=False,
    verbose=False,
):
    """
    Master controller of the entire pipeline

    Parameters
    ----------
    target_datadir : str
        Target measurement set directory
    workdir : str
        Work directory path
    outdir : str
        Output directory

    target_metafits : str
        Target metafits file
    calibrator_datadir : str, optional
        Calibrator data directory
    calibrator_metafits : str, optional
        Calibrator metafits file

    do_forcereset_weightflag : bool, optional
        Reset weights and flags of the input ms

    do_basic_cal : bool, optional
        Perform basic calibration
    do_applycal : bool, optional
        Apply basic calibration on target scans
    only_amplitude : bool, optional
        Apply only amplitude part of gain solution from calibrator
    redo_basic_cal : bool, optional
        Redo basic calibration
    use_uvbinflagger : bool, optional
        Use uvbin flagger on corrected data or not

    freqrange : str, optional
        Frequency range to image in MHz (xx1~xx2,xx3~xx4,)
    timerange : str, optional
        Time range to image in YYYY/MM/DD/hh:mm:ss format (tt1~tt2,tt3~tt4,...)
    uvrange : str, optional
        UV-range for calibration

    do_polcal : bool, optional
        Perform full-polarization calibration and imaging

    do_selfcal : bool, optional
        Perform self-calibration
    do_apply_selfcal : bool, optional
        Apply self-calibration solutions
    do_ap_selfcal : bool, optional
        Perform amplitude-phase self-cal or not
    int_solint : str, optional
        Solution intervals in gain self-cal
    pol_solint : str, optional
        Solution intervals in polarisation self-cal
    redo_selfcal : bool, optional
        Redo self-calibration or not

    do_sidereal_cor : bool, optional
        Perform solar sidereal motion correction or not
    make_ds : bool, optional
        Make dynamic spectra

    do_imaging : bool, optional
        Perform final imaging
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting (-1 to 1)
    minuv_l : float, optional
        Minimum UV-lambda for final imaging
    image_freqres : float, optional
        Image frequency resolution in MHz (-1 means full bandwidth)
    image_timeres : float, optional
        Image temporal resolution in seconds (-1 means full scan duration)
    pol : str, optional
        Stokes parameters of final imaging
    clean_threshold : float, optional
        CLEAN threshold of final imaging in sigma
    use_multiscale : bool, optional
        Use multiscale scales or not
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 10.0 solar radii)
    make_overlay : bool, optional
        Make EUV MWA overlay for all images or not (default : per coarse channel images will be overlaid at 60s intervals)
    make_TB : bool, optional
        Make brightness temperature maps or not
    save_hpc : bool, optional
        Save helioprojective fits or not
    compress_image : bool, optional
        Save compressed image or not
    keep_original : bool, optional
        In case of compress image, whether keep original fits or not
    make_msplot : bool, optional
        Make diagnostic plots of measurement sets

    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    max_worker: int, optional
        Maximum number of workers or parallel processes (default: number of coarse channels)
    keep_backup : bool, optional
        Keep backup of of all intermediate data poducts, calibrated ms, self-cal rounds and including images, models and residual images
    keep_calibrated_ms : bool, optional
        Keep calibrated measurement sets or not

    remote_logger : bool, optional
        Enable remote logging of the pipeline status
    jobid : str, optional
        Job ID
    job_password : str, optional
        User specified job password for remote logger
    adaptive : bool, optional
        Whether do adaptive scaling or not
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message
    """
    start_time = time.time()
    masterlogger = get_logger_safe()
    if verbose:
        masterlogger.setLevel(logging.DEBUG)

    masterlogger.info("P-AIRCARS workflow started.")
    emails = get_emails()

    #######################################################
    # Checking validity of target directories and metafits
    #######################################################
    if target_datadir.startswith("~"):
        masterlogger.critical("Please provide full path of target directory.")
        return 1
    else:
        target_datadir = os.path.abspath(target_datadir)
        if os.path.exists(target_datadir) is False:
            masterlogger.critical(
                f"Target data directory: {target_datadir} does not exist. Provide correct full path."
            )
            return 1
        else:
            masterlogger.debug(f"Target data directory: {target_datadir}")

    #################################################
    # Checking workdir and outdir paths
    #################################################
    if workdir.startswith("~"):
        masterlogger.critical("Please provide full path of work directory.")
        return 1
    else:
        workdir = os.path.abspath(workdir)
    if outdir.startswith("~"):
        masterlogger.critical("Please provide full path of output directory.")
        return 1
    else:
        outdir = os.path.abspath(outdir)

    if jobid is None:
        jobid = get_jobid()
        for banner in print_banner(
            f"P-AIRCARS Job ID: {jobid}", no_print=True
        ).splitlines():
            masterlogger.info(banner)

    #########################################
    # Some validity checks for resources
    #########################################
    max_worker = max(2, max_worker)  # Minimum 2 workers are needed
    cpu_frac = min(0.8, abs(cpu_frac))
    mem_frac = min(0.8, abs(mem_frac))

    masterlogger.info("Sorting out target data.")
    #############################################
    # Listing target ms
    #############################################
    target_mslist = sorted(glob.glob(f"{target_datadir}/*.ms"))
    if len(target_mslist) == 0:
        masterlogger.critical(
            f"No measurement set is present in target data directory: {target_datadir}"
        )
        if emails != "":
            email_msg = "No measurement set is present in the target data directory."
            send_task_notification(emails, email_msg, jobid, "N/A", "N/A")
        return 1
    else:
        masterlogger.debug("All measurement sets in target directory:")
        for ms in target_mslist:
            masterlogger.debug(ms)

    #################################################
    # Verifying whether all target ms from same obsid
    #################################################
    target_ms_obsids = [get_MWA_OBSID(ms) for ms in target_mslist]
    all_same_obsids = all(x == target_ms_obsids[0] for x in target_ms_obsids)
    if not all_same_obsids:
        masterlogger.critical(
            "All target measurement sets are not belong to same OBSID. Keep only measurement sets with same OBSID inside the target directory. P-AIRCARS has stopped."
        )
        return 1
    else:
        target_obsid = target_ms_obsids[0]

    ##############################################
    # Determining target ms coarse channels
    ##############################################
    target_ms_coarse_chans = []
    for target_ms in target_mslist:
        coarse_chans = get_MWA_coarse_chan(target_ms)
        for ch in coarse_chans:
            if ch not in target_ms_coarse_chans:
                target_ms_coarse_chans.append(ch)
    masterlogger.debug(
        f"Target measurement set coarse channels: {target_ms_coarse_chans}"
    )

    ##############################################
    # Downloading target metafits if not exist
    ##############################################
    if target_metafits == "" or target_metafits is None:
        if os.path.exists(f"{target_datadir}/{target_obsid}.metafits"):
            target_metafits = f"{target_datadir}/{target_obsid}.metafits"
            download_metafits = False
        else:
            download_metafits = True
    elif not os.path.exists(target_metafits):
        download_metafits = True
    else:
        download_metafits = False
    if download_metafits:
        masterlogger.debug(f"Downloading metafits for OBSID: {target_obsid}")
        try:
            target_metafits = download_MWA_metafits(target_obsid, outdir=target_datadir)
        except Exception:
            if emails != "":
                email_msg = f"Target metafits for OBSID: {target_obsid} is not provided and also could not be downloaded. P-AIRCARS has stopped."
                send_task_notification(emails, email_msg, jobid, "N/A", "N/A")
            masterlogger.exception(
                f"Target metafits for OBSID: {target_obsid} is not provided and also could not be downloaded. P-AIRCARS has stopped.",
                exc_info=True,
            )
            return 1

    ##################################################
    # Downloading target metafits if not match with ms
    ##################################################
    metafits_obsid = fits.getheader(target_metafits)["GPSTIME"]
    if metafits_obsid != target_obsid:
        masterlogger.info(
            f"Mismatch between target ms OBSID: {target_obsid} and metafits OBSID: {metafits_obsid}. Downloading metafits for OBSID: {target_obsid}."
        )
        masterlogger.debug(f"Downloading metafits for OBSID: {target_obsid}")
        try:
            target_metafits = download_MWA_metafits(target_obsid, outdir=target_datadir)
        except Exception:
            if emails != "":
                email_msg = f"Target metafits for OBSID: {target_obsid} could not be downloaded. P-AIRCARS has stopped."
                send_task_notification(emails, email_msg, jobid, "N/A", "N/A")
            masterlogger.exception(
                f"Target metafits for OBSID: {target_obsid} could not be downloaded. P-AIRCARS has stopped.",
                exc_info=True,
            )
            return 1

    ################################################
    # Final target OBSID and frequency configuration
    ################################################
    target_header = fits.getheader(target_metafits)
    target_obsid = int(target_header["GPSTIME"])
    target_freq_config = target_header["CHANNELS"]
    target_coarse_chans = [int(c) for c in target_freq_config.split(",")]
    target_coarse_chans = list(set(target_coarse_chans) & set(target_ms_coarse_chans))
    for banner in print_banner(
        f"Target observation ID: {target_obsid}", no_print=True
    ).splitlines():
        masterlogger.info(banner)
    masterlogger.info(f"Target coarse channels: {target_coarse_chans}")

    ################################################
    # Filtering calibrators
    ################################################
    masterlogger.info("Sorting out calibrator data.")
    cal_datadir_list = calibrator_datadir.split(",")
    cal_metafits_list = calibrator_metafits.split(",")
    final_cal_datadir_list = []
    final_cal_obsid_list = []
    final_cal_metafits_list = []
    final_cal_coarsechan_list = []

    for i in range(len(cal_datadir_list)):
        cal_datadir = cal_datadir_list[i]
        #############################################
        # Listing calibrator ms
        #############################################
        if cal_datadir != "" and os.path.exists(cal_datadir):
            cal_mslist = sorted(glob.glob(f"{cal_datadir}/*.ms"))
            if len(cal_mslist) == 0:
                masterlogger.warning(
                    f"No measurement set is present in calibrator data directory: {cal_datadir}"
                )
                has_cal = False
            else:
                has_cal = True
        else:
            has_cal = False
        ######################################################
        # Verifying whether all calibraror ms from same obsid
        ######################################################
        if has_cal:
            cal_ms_obsids = [get_MWA_OBSID(ms) for ms in cal_mslist]
            all_same_obsids = all(x == cal_ms_obsids[0] for x in cal_ms_obsids)
            if not all_same_obsids:
                masterlogger.warning(
                    "All calibrator measurement sets are not belong to same OBSID. Not using this calibrator."
                )
                has_cal = False
            else:
                cal_obsid = cal_ms_obsids[0]
        ##############################################
        # Searching for calibrator metafits
        ##############################################
        cal_metafits = None
        if has_cal and len(cal_metafits_list) > 0:
            for metafits in cal_metafits_list:
                if (metafits == "" or not os.path.exists(metafits)) and os.path.exists(
                    f"{cal_datadir}/{cal_obsid}.metafits"
                ):
                    metafits = f"{cal_datadir}/{cal_obsid}.metafits"
                metafits_obsid = fits.getheader(metafits)["GPSTIME"]
                if metafits_obsid == cal_obsid:
                    cal_metafits = metafits
                    break
        ######################################################
        # Downloading calibrator metafits if not match with ms
        ######################################################
        if has_cal and cal_metafits is None:
            masterlogger.debug(
                f"Downloading metafits for calibrator OBSID: {cal_obsid}"
            )
            try:
                cal_metafits = download_MWA_metafits(cal_obsid, outdir=cal_datadir)
            except Exception:
                masterlogger.warning(
                    f"Calibrator metafits for OBSID: {cal_obsid} could not be downloaded. Not using this calibrator."
                )
                has_cal = False
        ####################################################################
        # Including in final list if have overlapping frequency with target
        ####################################################################
        if has_cal and cal_metafits is not None:
            cal_header = fits.getheader(cal_metafits)
            cal_obsid = int(cal_header["GPSTIME"])
            if (
                abs(cal_obsid - target_obsid) < 24 * 3600
            ):  # Only if calibrator is 24 hours apart
                cal_mslist = sorted(glob.glob(f"{cal_datadir}/*.ms"))
                cal_ms_coarse_chans = []
                for cal_ms in cal_mslist:
                    coarse_chans = get_MWA_coarse_chan(cal_ms)
                    for ch in coarse_chans:
                        if ch not in cal_ms_coarse_chans:
                            cal_ms_coarse_chans.append(ch)
                cal_freq_config = cal_header["CHANNELS"]
                cal_coarse_chans = [int(c) for c in cal_freq_config.split(",")]
                cal_coarse_chans = list(
                    set(cal_coarse_chans) & set(cal_ms_coarse_chans)
                )
                has_overlap = bool(set(cal_coarse_chans) & set(target_coarse_chans))
                if not has_overlap:
                    masterlogger.warning(
                        f"Calibrator with OBSID: {cal_obsid} do not have frequency overlap with target."
                    )
                    masterlogger.info(f"Target coarse channels: {target_coarse_chans}")
                    masterlogger.info(f"Calibrator coarse channels: {cal_coarse_chans}")
                else:
                    final_cal_datadir_list.append(cal_datadir)
                    final_cal_obsid_list.append(cal_obsid)
                    final_cal_metafits_list.append(cal_metafits)
                    final_cal_coarsechan_list.append(cal_coarse_chans)

    ####################################################
    # Check whether there is calibrator available or not
    ####################################################
    calibrator_dic = {}
    if len(final_cal_datadir_list) > 0:
        ####################################################
        # Arranging in time
        ####################################################
        final_cal_datadir_list = np.array(final_cal_datadir_list)
        final_cal_obsid_list = np.array(final_cal_obsid_list)
        final_cal_metafits_list = np.array(final_cal_metafits_list)
        final_cal_coarsechan_list = np.array(final_cal_coarsechan_list)
        pos = np.argsort(abs(final_cal_obsid_list - target_obsid))
        final_cal_datadir_list = final_cal_datadir_list[pos].tolist()
        final_cal_obsid_list = final_cal_obsid_list[pos].tolist()
        final_cal_metafits_list = final_cal_metafits_list[pos].tolist()
        final_cal_coarsechan_list = final_cal_coarsechan_list[pos].tolist()
        all_coarse_chans = []
        for i in range(len(final_cal_datadir_list)):
            cal_obsid = final_cal_obsid_list[i]
            cal_datadir = final_cal_datadir_list[i]
            cal_metafits = final_cal_metafits_list[i]
            coarse_chans = final_cal_coarsechan_list[i]
            overlapping_coarse_chans = list(
                set(coarse_chans) & set(target_coarse_chans)
            )
            filtered_overlapping_coarse_chans = []
            for c in overlapping_coarse_chans:
                if c not in all_coarse_chans:
                    filtered_overlapping_coarse_chans.append(c)
                    all_coarse_chans.append(c)
            if len(filtered_overlapping_coarse_chans) > 0:
                calibrator_dic[cal_obsid] = [
                    cal_datadir,
                    cal_metafits,
                    filtered_overlapping_coarse_chans,
                ]

    if len(calibrator_dic) == 0:
        has_cal = False
        masterlogger.warning("No calibrator data is found after filtering.")
    else:
        has_cal = True
        masterlogger.info(
            f"Total {len(calibrator_dic)} calibrator observations are sorted. Observation ID(s) are: {list(calibrator_dic.keys())}"
        )

    #######################################
    # Preparing target working directories
    #######################################
    masterlogger.info("Preparing working directories.")
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(target_mslist[0])) + "/workdir"

    workdir = workdir.rstrip("/")
    if outdir == "":
        outdir = workdir

    target_obsdate, _ = get_gpstime_to_date(target_obsid)
    workdir = f"{workdir}/{target_obsdate}"
    os.makedirs(workdir, exist_ok=True)

    temp_workdir = f"{workdir}/{target_obsid}_{jobid}_target"
    try:
        os.makedirs(temp_workdir, exist_ok=True)
        if os.path.exists(f"{workdir}/inputs.txt"):
            os.system(f"mv {workdir}/inputs.txt {temp_workdir}/inputs.txt")
        workdir = temp_workdir
    except Exception:
        masterlogger.exception(
            f"Work directory: {temp_workdir} can not be created. Please check the path carefully.",
            exc_info=True,
        )
        return 1

    ####################################
    # Preparing target output directories
    ####################################
    outdir = outdir.rstrip("/")
    outdir = f"{outdir}/{target_obsdate}"
    os.makedirs(outdir, exist_ok=True)

    target_outdir = f"{outdir}/{target_obsid}_target"
    try:
        os.makedirs(target_outdir, exist_ok=True)
    except Exception:
        masterlogger.exception(
            f"Output directory: {target_outdir} can not created. Please check the path carefully.",
            exc_info=True,
        )
        return 1
    selfcaldir = f"{target_outdir}/caltables"
    os.makedirs(selfcaldir, exist_ok=True)

    ######################################################
    # Making calibrator output directories
    ######################################################
    if has_cal:
        cal_outdir = f"{outdir}/calibrators"
        try:
            os.makedirs(cal_outdir, exist_ok=True)
        except Exception:
            masterlogger.exception(
                f"Calibrator output directory: {cal_outdir} can not created. Please check the path carefully.",
                exc_info=True,
            )
            has_cal = False
        basic_caldir = f"{cal_outdir}/caltables"
        os.makedirs(basic_caldir, exist_ok=True)
    else:
        basic_caldir = ""

    ##########################
    # Change to workdir
    ##########################
    os.chdir(workdir)

    #####################################
    # Setup dask client
    #####################################
    dask_dir = None
    try:
        dask_client = get_client()
        dask_cluster = dask_client.cluster
    except Exception:
        masterlogger.debug(
            f"Creating dask cluster. CPU fraction: {cpu_frac}, memory fraction: {mem_frac}, maxmim worker: {max_worker}"
        )
        dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            max_worker=max_worker,
        )
        if dask_client is None:
            masterlogger.critical("Error occured in creating local cluster.")
            return 1
    dask_addr = dask_client.scheduler.address

    #####################################
    # Initiating paircars data
    #####################################
    init_paircars_data()

    ############################################
    # Determine number of threads of main worker
    ############################################
    n_threads = os.environ.get("OMP_NUM_THREADS")
    if n_threads is None:
        masterlogger.warning(
            "Number of threads is not available in environment. Using one thread."
        )
        n_threads = 1
    else:
        n_threads = max(1, int(n_threads))
        masterlogger.debug(f"Number of threads per worker to use: {n_threads}")

    #########################################
    # Setup remote loggger and email notifier
    #########################################
    masterlogger.info("Setting up remote logger and email notifier.")
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    master_logfile = f"{logdir}/master_{jobid}.log"
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id, flow_name, master_logfile, poll_interval=3, stop_event=stop_event
    )
    observer = None
    try:
        #####################################
        # Reading remotelink and emails
        #####################################
        remote_link = ""
        internet_on = internet_available()
        if not internet_on:
            masterlogger.warning(
                "Internet connection is not available for remote logging."
            )
        else:
            if remote_logger:
                try:
                    remote_link = get_remote_logger_link()
                except Exception:
                    pass
                if remote_link == "":
                    masterlogger.warning("Please provide a valid remote link.")
                    remote_logger = False

        if not remote_logger:
            timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            internet_on = internet_available()
            hostname = socket.gethostname()
            if internet_on and emails != "":
                email_subject = f"P-AIRCARS Log: {timestamp}, OBSID: {target_obsid}, Hostname: {hostname}"
                email_msg = f"P-AIRCARS Job ID: {jobid}"
                success_msg, error_msg = send_notification(
                    emails, email_subject, email_msg
                )
        else:
            ####################################
            # Job name and logging password
            ####################################
            hostname = socket.gethostname()
            timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            username = getpass.getuser()
            jobname = f"{username}-{hostname}:{timestamp}:{target_obsid}"
            if job_password is None:
                password = get_remote_logger_password()
            else:
                password = job_password
            if password == "":
                password = generate_password()
            np.save(
                f"{workdir}/.jobname_password.npy",
                np.array([jobname, password], dtype="object"),
            )
            ############
            # Logger
            ############
            if os.path.exists(f"{workdir}/.jobname_password.npy"):
                time.sleep(5)
                jobname, password = np.load(
                    f"{workdir}/.jobname_password.npy", allow_pickle=True
                )
                if master_logfile is not None and os.path.exists(master_logfile):
                    observer = init_logger(
                        "master_log",
                        master_logfile,
                        log_type="master",
                        jobname=jobname,
                        password=password,
                    )
            if observer is None:
                masterlogger.warning(
                    "Remote link or jobname is blank. Not transmiting to remote masterlogger."
                )
            #####################
            # Notify over email
            #####################
            internet_on = internet_available()
            hostname = socket.gethostname()
            if emails != "" and internet_on:
                email_subject = f"P-AIRCARS Log: {timestamp}, OBSID: {target_obsid}, Hostname: {hostname}"
                email_msg = (
                    f"P-AIRCARS Job ID: {jobid}\n"
                    f"Remote logger Job ID: {jobname}\n"
                    f"Remote access password: {password}"
                )
                success_msg, error_msg = send_notification(
                    emails, email_subject, email_msg
                )

        #####################################
        # Printing basic info of the pipeline
        #####################################
        for banner in print_banner(
            f"Work directory: {workdir}", no_print=True
        ).splitlines():
            masterlogger.info(banner)
        for banner in print_banner(
            f"Final product directory: {outdir}", no_print=True
        ).splitlines():
            masterlogger.info(banner)
        if remote_logger:
            masterlogger.info("####################################")
            masterlogger.info(f"{remote_link}")
            masterlogger.info(f"Remote Job ID: {jobname}")
            masterlogger.info(f"Remote access password: {password}")
            masterlogger.info("####################################")

        if not has_cal:
            for banner in print_banner(
                f"No suitable calibrators are available for target OBSID: {target_obsid}.",
                no_print=True,
            ).splitlines():
                masterlogger.info(banner)
            if emails != "":
                email_msg = f"No suitable calibrators are available for target OBSID: {target_obsid}."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"master flow {flow_name}",
                )

        ###########################################
        # Setting up mutual conditions
        ###########################################
        # Switch on applycal if selfcal is requested
        if do_selfcal:
            if not do_applycal:
                masterlogger.debug(
                    "Switching on apply basic calibrations, because self-calibration is requested."
                )
                do_applycal = True

        # Switch on applycal and apply selfcal if imaging is requested
        if do_imaging:
            if not do_applycal:
                do_applycal = True
                masterlogger.debug(
                    "Switching on apply basic calibrations, because imaging is requested."
                )
            if not do_apply_selfcal:
                masterlogger.debug(
                    "Switching on apply self-calibrations, because imaging is requested."
                )
                do_apply_selfcal = True

        #####################################################################
        # Checking if ms is full pol for polarization calibration and imaging
        #####################################################################
        if do_polcal:
            masterlogger.info(
                "Checking measurement set suitability for polarization calibration...."
            )
            for msname in target_mslist:
                msmd = msmetadata()
                msmd.open(msname)
                npol = msmd.ncorrforpol()[0]
                msmd.close()
                if npol < 4:
                    masterlogger.warning(
                        f"Measurement set: {msname} is not full-polar. Do not performing polarization analysis."
                    )
                    do_polcal = False
                    break

        #################################################
        # Determining maximum allowed frequency averaging
        #################################################
        highest_freq_ms = target_mslist[0]
        init_coarse_chan = max(get_MWA_coarse_chan(highest_freq_ms))
        for msname in target_mslist:
            coarse_chan = max(get_MWA_coarse_chan(msname))
            if coarse_chan > init_coarse_chan:
                init_coarse_chan = coarse_chan
                highest_freq_ms = msname

        masterlogger.info(
            f"Estimating optimal frequency averaging using highest frequency measurement set: {highest_freq_ms}."
        )
        max_freqres = calc_bw_smearing_freqwidth(highest_freq_ms, full_FoV=False)
        msmd = msmetadata()
        msmd.open(msname)
        freqres = msmd.chanres(0, unit="MHz")[0]
        msmd.close()
        if freqres > 0.16:
            masterlogger.info(
                f"Frequency resolution: {round(freqres*1000,1)}kHz is more than 160kHz. Assuming channel flagging is already done before averaing."
            )
        if image_freqres > 0:
            image_freqres = max(image_freqres, freqres)
            freqavg = round(min(image_freqres, max_freqres), 2)
        else:
            freqavg = freqres
        freqavg = min(0.16, freqavg)
        image_freqres = round(image_freqres, 2)

        total_ncoarse = 0
        for msname in target_mslist:
            ncoarse = get_ncoarse(msname)
            total_ncoarse += ncoarse
        total_ncoarse = max(1, total_ncoarse)
        masterlogger.debug(
            f"Total number of coarse channels in target: {total_ncoarse}."
        )

        ################################################
        # Determining maximum allowed temporal averaging
        ################################################
        masterlogger.debug(
            f"Estimating optimal temporal averaging using highest frequency measurement set: {highest_freq_ms}."
        )
        solar_smearing = max_time_solar_smearing(highest_freq_ms)
        max_timeres = min(
            calc_time_smearing_timewidth(highest_freq_ms),
            solar_smearing,
        )
        msmd.open(highest_freq_ms)
        times = msmd.timesforspws(0)
        timeres = np.nanmean(np.diff(times))
        msmd.close()
        quack_timestamps = int(4.0 / timeres)
        if image_timeres > solar_smearing:
            masterlogger.info(
                f"Image time integration is more than {solar_smearing}ss, which may cause smearing due to solar differential rotation."
            )
        if image_timeres > 0:
            image_timeres = max(image_timeres, timeres)
            timeavg = round(min(image_timeres, max_timeres), 2)
        else:
            timeavg = timeres
        timeavg = min(2.0, timeavg)
        image_timeres = round(image_timeres, 2)
        masterlogger.info(
            f"Measurement set frequency resolution: {freqres}MHz, time resolution: {timeres}s."
        )
        masterlogger.info(
            f"Frequency averaging: {freqavg}MHz, time averaging: {timeavg}s."
        )
        masterlogger.info(
            f"Imaging frequency resolution: {image_freqres}MHz, time resolution: {image_timeres}s."
        )

        #############################
        # Reset any previous weights
        #############################
        masterlogger.info("Resetting previous flags and weights.")
        if len(target_mslist) > 0:
            for msname in target_mslist:
                masterlogger.debug(f"Resetting for target ms: {msname}")
                reset_weights_and_flags(
                    msname, n_threads=n_threads, force_reset=do_forcereset_weightflag
                )
        if has_cal:
            cal_obsids = calibrator_dic.keys()
            for cal_obsid in cal_obsids:
                cal_datadir = calibrator_dic[cal_obsid][0]
                calibrator_mslist = glob.glob(f"{cal_datadir}/*.ms")
                if len(calibrator_mslist) > 0:
                    for msname in calibrator_mslist:
                        masterlogger.debug(f"Resetting for calibrator ms: {msname}")
                        reset_weights_and_flags(
                            msname,
                            n_threads=n_threads,
                            force_reset=do_forcereset_weightflag,
                        )
        masterlogger.info("Reset is done.")

        ##########################################
        # Basic calibration flows
        ##########################################
        if has_cal and do_basic_cal:
            cal_obsids = list(calibrator_dic.keys())
            succeed = 0
            all_bandpass_tables = []
            all_crossphase_tables = []
            for cal_obsid in cal_obsids:
                cal_datadir, cal_metafits, coarse_chans = calibrator_dic[cal_obsid]
                if adaptive:
                    cal_mslist = glob.glob(f"{cal_datadir}/*.ms")
                total_ncoarse = 0
                for calms in cal_mslist:
                    ms_coarse_chans = get_MWA_coarse_chan(calms)
                    ms_coarse_chans = list(set(ms_coarse_chans) & set(coarse_chans))
                    ncoarse = len(ms_coarse_chans)
                    total_ncoarse += ncoarse
                    scale_worker_and_wait(
                        dask_cluster,
                        dask_client,
                        max(1, min(total_ncoarse, max_worker))
                        + 1,  # 1 worker for prefect task
                    )
                for banner in print_banner(
                    f"Starting basic calibration subflow for calibrator OBSID: {cal_obsid}, coarse channels: {coarse_chans}",
                    no_print=True,
                ).splitlines():
                    masterlogger.info(banner)
                (
                    basical_msg,
                    bandpass_tables,
                    crossphase_tables,
                ) = basic_cal_subflow.with_options(
                    flow_run_name=f"basiccal_subflow_{cal_obsid}",
                    task_runner=DaskTaskRunner(address=dask_addr),
                )(
                    cal_obsid=cal_obsid,
                    cal_datadir=cal_datadir,
                    cal_metafits=cal_metafits,
                    coarse_chans=coarse_chans,
                    target_obsid=target_obsid,
                    target_metafits=target_metafits,
                    workdir=workdir,
                    cal_outdir=cal_outdir,
                    basic_caldir=basic_caldir,
                    redo_basic_cal=redo_basic_cal,
                    do_polcal=do_polcal,
                    keep_backup=keep_backup,
                    quack_timestamps=quack_timestamps,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    jobid=jobid,
                    timestamp=timestamp,
                    emails=emails,
                    remote_logger=remote_logger,
                    verbose=verbose,
                )
                if basical_msg == 0:
                    succeed += 1
                    all_bandpass_tables += bandpass_tables
                    all_crossphase_tables += crossphase_tables
                    masterlogger.info("Basic calibration subflow is successful.")
                else:
                    masterlogger.warning("Basic calibration subflow is failed.")
            failed = len(cal_obsids) - succeed
            masterlogger.info(f"Total calibrators observations : {len(cal_obsids)}.")
            masterlogger.info(f"Total succeeded: {succeed}.")
            masterlogger.info(f"Total failed: {failed}.")
            masterlogger.info(
                "Basic calibration subflows for all calibrators are done."
            )
            if emails != "":
                email_msg = f"Basic calibration of all calibrators are done.\nSucceeded: {succeed}, failed: {failed}."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"master flow {flow_name}",
                )
            if len(all_bandpass_tables) == 0:
                masterlogger.warning(
                    "No bandpass solutions obtained from any calibrators. Calibrating solely using self-calibration."
                )
                has_cal = False
                if emails != "":
                    email_msg = "No bandpass solutions obtained from any calibrators. Calibrating solely using self-calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"master flow {flow_name}",
                    )
            elif len(all_crossphase_tables) == 0:
                masterlogger.warning(
                    "No crosshand phase solutions obtained from any calibrators. Image-based crosshand phase calibration will be attempted."
                )
                if emails != "":
                    email_msg = "No crosshand phase solutions obtained from any calibrators. Image-based crosshand phase calibration will be attempted."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"master flow {flow_name}",
                    )

        ##################################################
        # Self-calibration flows
        ##################################################
        if do_selfcal:
            if adaptive:
                total_ncoarse = 0
                for targetms in target_mslist:
                    ms_coarse_chans = get_MWA_coarse_chan(targetms)
                    ncoarse = len(ms_coarse_chans)
                    total_ncoarse += ncoarse
                masterlogger.debug(
                    f"Total coarse channels for splited target measurement sets: {total_ncoarse}."
                )
                scale_worker_and_wait(
                    dask_cluster,
                    dask_client,
                    max(1, min(total_ncoarse, max_worker))
                    + 1,  # One worker for prefect task
                )
            for banner in print_banner(
                "Starting self-calibration subflow.", no_print=True
            ).splitlines():
                masterlogger.info(banner)
            (
                selfcal_msg,
                selfcal_gaintable,
                selfcal_bandpass,
                selfcal_leakage,
            ) = selfcal_subflow.with_options(
                flow_run_name=f"selfcal_subflow_{target_obsid}",
                task_runner=DaskTaskRunner(address=dask_addr),
            )(
                target_mslist=target_mslist,
                target_metafits=target_metafits,
                target_obsid=target_obsid,
                workdir=workdir,
                basic_caldir=basic_caldir,
                selfcaldir=selfcaldir,
                target_outdir=target_outdir,
                redo_selfcal=redo_selfcal,
                has_cal=has_cal,
                do_sidereal_cor=do_sidereal_cor,
                keep_backup=keep_backup,
                int_solint=int_solint,
                pol_solint=pol_solint,
                timeavg=timeavg,
                freqavg=freqavg,
                image_timeres=image_timeres,
                image_freqres=image_freqres,
                quack_timestamps=quack_timestamps,
                only_amplitude=only_amplitude,
                do_ap_selfcal=do_ap_selfcal,
                do_polcal=do_polcal,
                uvrange=uvrange,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                jobid=jobid,
                timestamp=timestamp,
                emails=emails,
                remote_logger=remote_logger,
                verbose=verbose,
            )
            if selfcal_msg == 0 and len(selfcal_gaintable) > 0:
                masterlogger.info("Self-calibration subflow is successful.")
            else:
                masterlogger.warning(
                    "Self-calibration subflow is not successful. No solutions are available to apply."
                )
                do_apply_selfcal = False

        ##############################################
        # Apply solutions subflow
        ##############################################
        if do_applycal or do_apply_selfcal:
            if adaptive:
                total_ncoarse = 0
                for targetms in target_mslist:
                    ms_coarse_chans = get_MWA_coarse_chan(targetms)
                    ncoarse = len(ms_coarse_chans)
                    total_ncoarse += ncoarse
                scale_worker_and_wait(
                    dask_cluster,
                    dask_client,
                    max(1, min(total_ncoarse, max_worker))
                    + 1,  # One worker for prefect task
                )
            for banner in print_banner(
                "Starting apply solutions subflow.", no_print=True
            ).splitlines():
                masterlogger.info(banner)
            applycal_msg, split_target_mslist = applysol_subflow.with_options(
                flow_run_name=f"applysol_subflow_{target_obsid}",
                task_runner=DaskTaskRunner(address=dask_addr),
            )(
                target_mslist=target_mslist,
                target_metafits=target_metafits,
                target_obsid=target_obsid,
                workdir=workdir,
                basic_caldir=basic_caldir,
                selfcaldir=selfcaldir,
                target_outdir=target_outdir,
                do_applycal=do_applycal,
                do_apply_selfcal=do_apply_selfcal,
                has_cal=has_cal,
                do_polcal=do_polcal,
                do_sidereal_cor=do_sidereal_cor,
                use_uvbinflagger=use_uvbinflagger,
                freqavg=freqavg,
                timeavg=timeavg,
                quack_timestamps=quack_timestamps,
                only_amplitude=only_amplitude,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                jobid=jobid,
                timestamp=timestamp,
                emails=emails,
                remote_logger=remote_logger,
                verbose=verbose,
            )
            if applycal_msg == 0 and len(split_target_mslist) > 0:
                masterlogger.info("Apply solution subflow is successful.")
                split_target_mslist = sorted(glob.glob(f"{workdir}/target*_ch_*.ms"))
            else:
                masterlogger.critical(
                    "Apply solution subflow is failed. No calibrated target measurement set is available for imaging."
                )
                if emails != "":
                    email_msg = (
                        "Error occured in applying solutions. P-AIRCARS has stopped."
                    )
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"master flow {flow_name}",
                    )
                return 1

        #######################################
        # Run dynamic spectra making
        #######################################
        if make_ds:
            if adaptive:
                scale_worker_and_wait(
                    dask_cluster,
                    dask_client,
                    max(1, min(len(split_target_mslist), max_worker))
                    + 1,  # One worker for prefect task
                )
            if emails != "":
                email_msg = f"[{target_obsid}] Started making solar dynamic spectra."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"master flow {flow_name}",
                )
            print_banner("Starting task: Making dynamic spectra of solar target.")
            try:
                future_maskms = run_ds_jobs.with_options(
                    task_run_name=f"make_ds_{target_obsid}",
                ).submit(
                    ",".join(split_target_mslist),
                    target_metafits,
                    workdir,
                    target_outdir,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                wait([future_maskms])
                msg, succeed, failed = future_maskms.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Making solar dynamic spectra are done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"master flow {flow_name}",
                    )
                print_banner("Finished task: Making solar dynamic spectra are done.")
            except Exception:
                masterlogger.exception(
                    "Error in making dynamic spectra.", exc_info=True
                )
                if emails != "":
                    email_msg = (
                        f"[{target_obsid}] Error occured in making dynamic spectra."
                    )
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"master flow {flow_name}",
                    )

        ###################################
        # Imaging subflow
        ###################################
        if do_imaging:
            if adaptive:
                scale_worker_and_wait(
                    dask_cluster,
                    dask_client,
                    max(1, min(len(split_target_mslist), max_worker))
                    + 1,  # One worker for prefect task
                )
            masterlogger.info("Starting imaging subflow.")
            imaging_msg = imaging_subflow.with_options(
                flow_run_name=f"imaging_subflow_{target_obsid}",
                task_runner=DaskTaskRunner(address=dask_addr),
            )(
                split_target_mslist=split_target_mslist,
                target_metafits=target_metafits,
                target_obsid=target_obsid,
                workdir=workdir,
                selfcaldir=selfcaldir,
                target_outdir=target_outdir,
                do_imaging=do_imaging,
                do_polcal=do_polcal,
                keep_backup=keep_backup,
                make_overlay=make_overlay,
                make_TB=make_TB,
                save_hpc=save_hpc,
                image_freqres=image_freqres,
                image_timeres=image_timeres,
                pol=pol,
                freqrange=freqrange,
                timerange=timerange,
                minuv_l=minuv_l,
                weight=weight,
                robust=robust,
                clean_threshold=clean_threshold,
                use_multiscale=use_multiscale,
                cutout_rsun=cutout_rsun,
                compress_image=compress_image,
                keep_original=keep_original,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                jobid=jobid,
                timestamp=timestamp,
                emails=emails,
                remote_logger=remote_logger,
                verbose=verbose,
            )
            if imaging_msg != 0:
                masterlogger.critical("Error occured in imaging subflow.")
                if emails != "":
                    email_msg = "Error occured in imaging. P-AIRCARS has stopped."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"master flow {flow_name}",
                    )
                return 1
            else:
                masterlogger.info("Imaging subflow is successful.")

        ##############################################
        # Making diagnostic plots of measurement sets
        ##############################################
        if make_msplot:
            ###########################################
            # Ploting calibrator ms
            ###########################################
            split_cal_mslist = sorted(glob.glob(f"{workdir}/calibrator*_ch_*.ms"))
            if len(split_cal_mslist) == 0:
                masterlogger.warning(
                    "No calibrator measurement set is present for ploting."
                )
            else:
                if adaptive:
                    scale_worker_and_wait(
                        dask_cluster,
                        dask_client,
                        max(1, min(len(split_cal_mslist), max_worker))
                        + 1,  # One worker for prefect task
                    )
                msplot_outdir = f"{cal_outdir}/ms_diagnostics_plots"
                os.makedirs(msplot_outdir, exist_ok=True)
                if len(split_cal_mslist) > 0:
                    if emails != "":
                        email_msg = "Started making diagnostic plots for calibrator measurement sets."
                        send_task_notification(
                            emails, email_msg, jobid, target_obsid, timestamp
                        )
                    for banner in print_banner(
                        "Starting task: Making diagnostic plots of calibrator measurement sets.",
                        no_print=True,
                    ).splitlines():
                        masterlogger.info(banner)
                    try:
                        future_cal_plot = run_make_msplot.with_options(
                            task_run_name=f"msplot_cal_{jobid}"
                        ).submit(
                            ",".join(split_cal_mslist),
                            workdir,
                            msplot_outdir,
                            jobid=jobid,
                            cpu_frac=round(cpu_frac, 2),
                            mem_frac=round(mem_frac, 2),
                            remote_log=remote_logger,
                        )
                        msg = future_cal_plot.result()
                        if emails != "":
                            email_msg = "Making diagnostic plots for calibrator measurement sets are done."
                            send_task_notification(
                                emails, email_msg, jobid, target_obsid, timestamp
                            )
                        for banner in print_banner(
                            "Finished task: Making diagnostic plots for calibrator measurment sets are done.",
                            no_print=True,
                        ).splitlines():
                            masterlogger.info(banner)
                    except Exception:
                        masterlogger.exception(
                            "!!!! WARNING: Diagnostic plot of calibrator measurment sets are not successful. !!!!",
                            exc_info=True,
                        )
                        if emails != "":
                            email_msg = "Error occured in making diagnostic plots of calibrator measurement sets."
                            send_task_notification(
                                emails, email_msg, jobid, target_obsid, timestamp
                            )

            ###########################################
            # Ploting target ms
            ###########################################
            split_target_mslist = sorted(glob.glob(f"{workdir}/target*_ch_*.ms"))
            if len(split_target_mslist) == 0:
                masterlogger.warning("No target measurment set is present for ploting.")
            else:
                if adaptive:
                    scale_worker_and_wait(
                        dask_cluster,
                        dask_client,
                        max(1, min(len(split_target_mslist), max_worker))
                        + 1,  # One worker for prefect task
                    )
                msplot_outdir = f"{target_outdir}/ms_diagnostics_plots"
                os.makedirs(msplot_outdir, exist_ok=True)
                for banner in print_banner(
                    "Starting task: Making diagnostic plots of target measurement sets.",
                    no_print=True,
                ).splitlines():
                    masterlogger.info(banner)
                try:
                    future_target_plot = run_make_msplot.with_options(
                        task_run_name=f"msplot_target_{jobid}"
                    ).submit(
                        ",".join(split_target_mslist),
                        workdir,
                        msplot_outdir,
                        jobid=jobid,
                        cpu_frac=round(cpu_frac, 2),
                        mem_frac=round(mem_frac, 2),
                        remote_log=remote_logger,
                    )
                    msg = future_target_plot.result()
                    if msg == 0:
                        if emails != "":
                            email_msg = "Making diagnostic plots for target measurement sets are done."
                            send_task_notification(
                                emails, email_msg, jobid, target_obsid, timestamp
                            )
                        for banner in print_banner(
                            "Finished task: Making diagnostic plots for target measurment sets are done.",
                            no_print=True,
                        ).splitlines():
                            masterlogger.info(banner)
                    else:
                        masterlogger.error(
                            "Finished task: Error occured in making diagnostic plots for target measurment sets."
                        )
                        if emails != "":
                            email_msg = "Error occured in making diagnostic plots of target measurement sets."
                            send_task_notification(
                                emails, email_msg, jobid, target_obsid, timestamp
                            )
                except Exception:
                    masterlogger.exception(
                        "!!!! WARNING: Diagnostic plot of target measurment sets are not successful. !!!!",
                        exc_info=True,
                    )
                    if emails != "":
                        email_msg = "Error occured in making diagnostic plots of target measurement sets."
                        send_task_notification(
                            emails, email_msg, jobid, target_obsid, timestamp
                        )

        ###########################################
        # Successful exit
        ###########################################
        for banner in print_banner(
            "P-AIRCARS calibration and imaging pipeline is successfully executed.",
            no_print=True,
        ).splitlines():
            masterlogger.info(banner)
        if emails != "":
            email_msg = "P-AIRCARS processing is done successfully."
            send_task_notification(emails, email_msg, jobid, target_obsid, timestamp)
        return 0
    except Exception as e:
        if emails != "":
            email_msg = f"Error in running P-AIRCARS.\n{e}"
            send_task_notification(emails, email_msg, jobid, target_obsid, timestamp)
        masterlogger.exception("Error occured in running P-AIRCARS.", exc_info=True)
        return 1
    finally:
        time.sleep(5)
        try:
            datalist = sorted(glob.glob(f"{target_datadir}/*"))
            for data in datalist:
                drop_cache(data)
            cal_datadir_list = calibrator_datadir.split(",")
            for cal_datadir in cal_datadir_list:
                callist = sorted(glob.glob(f"{cal_datadir}/*"))
                for cal in callist:
                    drop_cache(cal)
            ######################################
            # Keeping flag backups
            ######################################
            # Flag backups of calibrator measurement sets
            ######################################
            final_cal_mslist = sorted(glob.glob(f"{workdir}/calibrator*_ch_*.ms"))
            if len(final_cal_mslist) > 0:
                os.makedirs(f"{cal_outdir}/ms_flags", exist_ok=True)
                masterlogger.info(
                    f"Doing flag backup for calibrator measurement sets in: {cal_outdir}/ms_flags"
                )
                for cal_ms in final_cal_mslist:
                    do_flag_backup(cal_ms, flagtype="finalflag")
                    if os.path.exists(
                        f"{cal_outdir}/ms_flags/{os.path.basename(cal_ms)}.flagversions"
                    ):
                        os.system(
                            f"rm -rf {cal_outdir}/ms_flags/{os.path.basename(cal_ms)}.flagversions"
                        )
                    os.system(f"mv {cal_ms}.flagversions {cal_outdir}/ms_flags/")
                    if not keep_backup:
                        os.system(f"rm -rf {cal_ms}*")
            ######################################
            # Removing selfcal measurement sets
            ######################################
            final_selfcal_mslist = sorted(glob.glob(f"{workdir}/selfcal*_ch_*.ms"))
            if len(final_selfcal_mslist) > 0:
                for selfcal_ms in final_selfcal_mslist:
                    if not keep_backup:
                        os.system(f"rm -rf {selfcal_ms}*")
            ######################################
            # Flag backups of target measurement sets
            ######################################
            final_split_target_mslist = sorted(glob.glob(f"{workdir}/target*_ch_*.ms"))
            if len(final_split_target_mslist) > 0:
                os.makedirs(f"{target_outdir}/ms_flags", exist_ok=True)
                masterlogger.info(
                    f"Doing flag backup target measurement sets in: {target_outdir}/ms_flags"
                )
                for target_ms in final_split_target_mslist:
                    do_flag_backup(target_ms, flagtype="finalflag")
                    if os.path.exists(
                        f"{target_outdir}/ms_flags/{os.path.basename(target_ms)}.flagversions"
                    ):
                        os.system(
                            f"rm -rf {target_outdir}/ms_flags/{os.path.basename(target_ms)}.flagversions"
                        )
                    os.system(f"mv {target_ms}.flagversions {target_outdir}/ms_flags/")
                    if keep_calibrated_ms:
                        calibrated_msdir = f"{target_outdir}/calibrated_ms"
                        os.makedirs(calibrated_msdir, exist_ok=True)
                        os.system(f"mv {target_ms} {calibrated_msdir}")
                    elif not keep_backup:
                        os.system(f"rm -rf {target_ms}*")
        except Exception:
            masterlogger.warning("Minor issues in final flag backup.")
            traceback.print_exc()
        time.sleep(5)
        drop_cache(workdir)
        drop_cache(outdir)
        end_time = time.time()
        run_time = end_time - start_time
        masterlogger.info(f"Total run time: {run_time}")
        stop_event.set()
        time.sleep(60)
        log_thread_flow.join()
        if dask_dir is not None:
            os.system(f"rm -rf {dask_dir}")
        if observer is not None:
            clean_shutdown(observer)


def cli():
    parser = argparse.ArgumentParser(
        description="Run P-AIRCARS for calibration and imaging of solar observations.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    # === Essential parameters ===
    essential = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    essential.add_argument(
        "target_datadir", type=str, help="Target measurement set directory"
    )
    essential.add_argument(
        "--workdir",
        type=str,
        dest="workdir",
        required=True,
        help="Working directory",
    )
    essential.add_argument(
        "--outdir",
        type=str,
        dest="outdir",
        required=True,
        help="Output products directory",
    )
    essential.add_argument(
        "--target_metafits",
        type=str,
        default="",
        dest="target_metafits",
        help="Target metafits file",
    )
    essential.add_argument(
        "--cal_datadir",
        type=str,
        dest="cal_datadir",
        default="",
        help="Calibrator measurement set directory",
    )
    essential.add_argument(
        "--cal_metafits",
        type=str,
        dest="cal_metafits",
        default="",
        help="Calibrator metafits file",
    )

    # === Advanced calibration parameters ===
    advanced_cal = parser.add_argument_group(
        "###################\nAdvanced calibration parameters\n###################"
    )
    advanced_cal.add_argument(
        "--int_solint",
        type=str,
        default="60s",
        help="Solution interval for gain calibration (e.g. 'int', '10s', '5min', 'inf')",
    )
    advanced_cal.add_argument(
        "--pol_solint",
        type=str,
        default="240s",
        help="Solution interval for polarisation calibration (e.g. 'int', '10s', '5min', 'inf')",
    )
    advanced_cal.add_argument(
        "--cal_uvrange",
        type=str,
        default="",
        help="UV range to filter data for calibration (e.g. '>100klambda', '100~10000lambda')",
    )
    advanced_cal.add_argument(
        "--no_polcal",
        action="store_false",
        dest="do_polcal",
        help="Disable polarization calibration",
    )
    advanced_cal.add_argument(
        "--redo_basic_cal",
        action="store_true",
        help="Redo basic calibration or not",
    )
    advanced_cal.add_argument(
        "--redo_selfcal",
        action="store_true",
        help="Redo self-calibration",
    )
    advanced_cal.add_argument(
        "--no_uvbinflagger",
        action="store_false",
        dest="use_uvbinflagger",
        help="Use uvbin flagger on corrected data or not",
    )
    advanced_cal.add_argument(
        "--only_amplitude",
        action="store_true",
        help="Apply only amplitude part of gain solution from calibrator or not",
    )

    # === Advanced imaging parameters ===
    advanced_image = parser.add_argument_group(
        "###################\nAdvanced imaging parameters\n###################"
    )
    advanced_image.add_argument(
        "--freqrange",
        type=str,
        default="",
        help="Frequency range in MHz to select during imaging (comma-separated, e.g. '100~110,130~140')",
    )
    advanced_image.add_argument(
        "--timerange",
        type=str,
        default="",
        help="Time range to select during imaging (comma-separated, e.g. '2014/09/06/09:30:00~2014/09/06/09:45:00,2014/09/06/10:30:00~2014/09/06/10:45:00')",
    )
    advanced_image.add_argument(
        "--image_freqres",
        type=float,
        default=1.28,
        help="Output image frequency resolution in MHz (-1 = full)",
    )
    advanced_image.add_argument(
        "--image_timeres",
        type=float,
        default=10.0,
        help="Output image time resolution in seconds (-1 = full)",
    )
    advanced_image.add_argument(
        "--pol",
        type=str,
        default="IQUV",
        help="Stokes parameter(s) to image ('I' or 'IQUV')",
    )
    advanced_image.add_argument(
        "--minuv_l",
        type=float,
        default=10,
        help="Minimum baseline length (in wavelengths) to include in imaging",
    )
    advanced_image.add_argument(
        "--weight",
        type=str,
        default="briggs",
        help="Imaging weighting scheme (e.g. 'briggs', 'natural', 'uniform')",
    )
    advanced_image.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust parameter for Briggs weighting (-2 to +2)",
    )
    advanced_image.add_argument(
        "--no_multiscale",
        action="store_false",
        dest="use_multiscale",
        help="Disable multiscale CLEAN for extended structures",
    )
    advanced_image.add_argument(
        "--clean_threshold",
        type=float,
        default=1.0,
        help="Clean threshold in sigma for final deconvolution",
    )
    advanced_image.add_argument(
        "--cutout_rsun",
        type=float,
        default=10.0,
        help="Field of view cutout radius in solar radii",
    )
    advanced_image.add_argument(
        "--do_overlay",
        action="store_true",
        dest="make_overlay",
        help="Make overlay plot on EUV images for all images (default is to make overlays only one image per coarse channels at 10s intervals)",
    )
    advanced_image.add_argument(
        "--make_TB",
        action="store_true",
        help="Make brightness temperature maps or not",
    )
    advanced_image.add_argument(
        "--no_save_hpc",
        action="store_false",
        dest="save_hpc",
        help="Do not save helioprojective fits",
    )
    advanced_image.add_argument(
        "--compress_image",
        action="store_true",
        help="Compress final images",
    )
    advanced_image.add_argument(
        "--keep_original",
        action="store_true",
        help="Keep original images in case of compress image is switched on",
    )

    # === Advanced options ===
    advanced = parser.add_argument_group(
        "###################\nAdvanced pipeline parameters\n###################"
    )
    advanced.add_argument(
        "--make_msplot",
        action="store_true",
        help="Make diagnostic plots of measurement sets",
    )
    advanced.add_argument(
        "--no_ds",
        action="store_false",
        dest="make_ds",
        help="Disable making solar dynamic spectra",
    )
    advanced.add_argument(
        "--do_forcereset_weightflag",
        action="store_true",
        help="Force reset of weights and flags (disabled by default)",
    )
    advanced.add_argument(
        "--no_basic_cal",
        action="store_false",
        dest="do_basic_cal",
        help="Disable basic gain calibration",
    )
    advanced.add_argument(
        "--do_sidereal_cor",
        action="store_true",
        dest="do_sidereal_cor",
        help="Sidereal motion correction for Sun (disabled by default)",
    )
    advanced.add_argument(
        "--no_selfcal",
        action="store_false",
        dest="do_selfcal",
        help="Disable self-calibration",
    )
    advanced.add_argument(
        "--no_ap_selfcal",
        action="store_false",
        dest="do_ap_selfcal",
        help="Disable amplitude-phase self-calibration",
    )
    advanced.add_argument(
        "--no_applycal",
        action="store_false",
        dest="do_applycal",
        help="Disable application of basic calibration solutions",
    )
    advanced.add_argument(
        "--no_apply_selfcal",
        action="store_false",
        dest="do_apply_selfcal",
        help="Disable application of self-calibration solutions",
    )
    advanced.add_argument(
        "--no_imaging",
        action="store_false",
        dest="do_imaging",
        help="Disable final imaging",
    )
    advanced.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of intermediate steps",
    )
    advanced.add_argument(
        "--keep_calibrated_ms",
        action="store_true",
        help="Keep calibrated measurement sets or not",
    )
    advanced.add_argument(
        "--no_remote_logger",
        action="store_false",
        dest="remote_logger",
        help="Disable remote logger",
    )
    advanced.add_argument(
        "--log2term",
        action="store_true",
        help="Show logs in terminal",
    )
    advanced.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs",
    )

    # === Advanced local system/ per node hardware resource parameters ===
    advanced_resource = parser.add_argument_group(
        "###################\nAdvanced hardware resource parameters for local system or per node on HPC cluster\n###################"
    )
    advanced_resource.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="Fraction of CPU usuage per node",
    )
    advanced_resource.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Fraction of memory usuage per node",
    )
    advanced_resource.add_argument(
        "--max_worker",
        type=int,
        default=1,
        help="Maximum number of parallel processes",
    )
    advanced_resource.add_argument(
        "--jobid",
        type=int,
        default=None,
        help="User provided P-AIRCARS job ID",
    )
    advanced_resource.add_argument(
        "--job_password",
        type=str,
        default=None,
        help="User specified job password",
    )
    advanced_resource.add_argument(
        "--cluster",
        action="store_true",
        dest="cluster",
        help="Running in cluster environment",
    )
    advanced_resource.add_argument(
        "--adaptive",
        action="store_true",
        dest="adaptive",
        help="Whether do adaptive scaling or not in cluster environment (In local environment, it is always true)",
    )

    # === Advanced job scheduler parameters ===
    advanced_slurm = parser.add_argument_group(
        "###################\nAdvanced slurm cluster settings\n###################"
    )
    advanced_slurm.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Partition name (Required)",
    )
    advanced_slurm.add_argument(
        "--account",
        type=str,
        default=None,
        help="Account name (If your cluster requires this, you should provide. Otherwise job can not be started)",
    )
    advanced_slurm.add_argument(
        "--walltime",
        type=str,
        default=None,
        help="Wall time, each slurm job can execute in maximum this time",
    )
    advanced_slurm.add_argument(
        "--num_node",
        type=str,
        default=-1,
        help="Maximum number of nodes to use (default: -1, use all nodes available in the account)",
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    scheduler_name = get_scheduler_name()
    ##################################################
    # Prefect settings check
    ##################################################
    prefect_settings = get_current_settings()
    prefect_env = prefect_settings.to_environment_variables()
    api_url = prefect_env.get("PREFECT_API_URL")

    ######################################
    # Check connection to prefect server
    ######################################
    if api_url is not None:
        check_url = f"{api_url}/health"
        try:
            requests.get(check_url, timeout=60)
        except Exception:
            traceback.print_exc()
            print(f"Could not reach prefect server at: {api_url} from compute node.")
            return 1
    else:
        print("Prefecr server is not running.")
        if scheduler_name == "local":
            print("P-AIRCARS will use ephmeral mode.")
        else:
            print(
                f"First start prefect server to run P-AIRCARS in {scheduler_name} cluster."
            )
            return 1

    if args.jobid is None:
        jobid = get_jobid()
    else:
        jobid = args.jobid

    ###########################################################
    # Estimating jobs memory size (5 times each measurment set)
    ###########################################################
    if os.path.exists(args.target_datadir) is False:
        print(f"Target data directory: {args.target_datadir} does not exist.")
        return
    target_mslist = glob.glob(f"{args.target_datadir}/*.ms")
    if len(target_mslist) == 0:
        print(
            f"No measurement set is present in the target directory: {args.target_datadir}"
        )
        return

    total_ncoarse = 0
    for msname in target_mslist:
        ncoarse = get_ncoarse(msname)
        total_ncoarse += ncoarse
    total_ncoarse = max(1, total_ncoarse)

    if args.cal_datadir:
        if os.path.exists(args.cal_datadir):
            cal_mslist = glob.glob(f"{args.cal_datadir}/*.ms")
            if len(cal_mslist) == 0:
                print(
                    f"No calibrator measurement set is present in: {args.cal_datadir}"
                )
        else:
            print("Calibrator data direcotry does not exist.")

    ###############################################
    # Setup cluster environment
    ###############################################
    if scheduler_name == "local":
        pid = os.getpid()
    elif scheduler_name == "slurm":
        pid = os.environ.get("SLURM_JOB_ID")
    else:
        print("P-AIRCARS is only ready for local or slurm cluster.")
        return 1

    if args.cluster and scheduler_name == "local":
        print(
            "User wants to use cluster architechture, but no job scheduler is available. Stopping P-AIRCARS."
        )
        return

    if args.mem_frac <= 0:
        mem_frac = 0.8
    else:
        mem_frac = args.mem_frac
    if args.cpu_frac <= 0:
        cpu_frac = 0.8
    else:
        cpu_frac = args.cpu_frac

    if args.max_worker is None:
        max_worker = total_ncoarse
    else:
        max_worker = max(int(args.max_worker), total_ncoarse)
    max_worker = max(1, max_worker)  # Minimum 1 worker is needed

    slurm_job = is_slurm_job()
    jobfile_name = ""

    if not args.cluster or scheduler_name == "local" or slurm_job is False:
        #######################################
        # Set up local cluster
        #######################################
        print("Setting up local cluster....")
        dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
            args.workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            max_worker=max_worker,
        )
        if dask_client is None:
            print("Error occured in creating local cluster.")
            return 1

        scheduler_address = dask_client.scheduler.address
        jobfile_name = save_main_process_info(
            pid,
            jobid,
            scheduler_address,
            args.target_datadir,
            os.path.abspath(args.workdir),
            os.path.abspath(args.outdir),
            args.cpu_frac,
            args.mem_frac,
        )
        adaptive = True  # For local cluster, always do adaptive scaling to avoid occupying resources
    else:
        if scheduler_name == "slurm":
            if args.partition is None:
                print("Please provide partition name to submit SLURM jobs.")
                return
            ########################################
            # Setting up slurm cluster
            ########################################
            print("Setting up slurm cluster....")
            cluster_result = get_slurm_dask_cluster(
                args.workdir,
                jobid=jobid,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                max_worker=max_worker,
                min_mem=1,
                partition=args.partition,
                account=args.account,
                walltime=args.walltime,
                num_node=args.num_node,
            )
            if cluster_result is None:
                print("Error occured in creating slurm cluster.")
                return 1
            else:
                dask_client, dask_cluster, dask_dir, nworker = cluster_result
            scheduler_address = dask_client.scheduler.address
            jobfile_name = save_main_process_info(
                pid,
                jobid,
                scheduler_address,
                args.target_datadir,
                os.path.abspath(args.workdir),
                os.path.abspath(args.outdir),
                args.cpu_frac,
                args.mem_frac,
            )
            adaptive = args.adaptive
            if not adaptive:
                nworker = min(total_ncoarse, nworker)
                scale_worker_and_wait(
                    dask_cluster, dask_client, max(1, nworker) + 1
                )  # One worker for prefect task
        else:
            print(
                f"P-AIRCARS is under development for job scheduler: {scheduler_name}. Stopping P-AIRCARS."
            )
            return

    ##########################################
    # Starting pipeline
    ##########################################
    try:
        dask_addr = dask_client.scheduler.address
        print_banner("Starting P-AIRCARS Pipeline....")
        print(f"Total dask workers: {nworker}")
        msg = master_control.with_options(
            flow_run_name=f"paircars_{jobid}",
            task_runner=DaskTaskRunner(address=dask_addr),
        )(
            args.target_datadir,
            args.workdir,
            args.outdir,
            target_metafits=args.target_metafits,
            calibrator_datadir=args.cal_datadir,
            calibrator_metafits=args.cal_metafits,
            # Pre-calibration
            do_forcereset_weightflag=args.do_forcereset_weightflag,
            # Basic calibration
            do_basic_cal=args.do_basic_cal,
            do_applycal=args.do_applycal,
            only_amplitude=args.only_amplitude,
            redo_basic_cal=args.redo_basic_cal,
            use_uvbinflagger=args.use_uvbinflagger,
            # Target data preparation
            freqrange=args.freqrange,
            timerange=args.timerange,
            uvrange=args.cal_uvrange,
            # Polarization calibration
            do_polcal=args.do_polcal,
            # Self-calibration
            do_selfcal=args.do_selfcal,
            do_apply_selfcal=args.do_apply_selfcal,
            do_ap_selfcal=args.do_ap_selfcal,
            int_solint=args.int_solint,
            pol_solint=args.pol_solint,
            redo_selfcal=args.redo_selfcal,
            # Sidereal correction
            do_sidereal_cor=args.do_sidereal_cor,
            # Dynamic spectra
            make_ds=args.make_ds,
            # Imaging
            do_imaging=args.do_imaging,
            weight=args.weight,
            robust=args.robust,
            minuv_l=args.minuv_l,
            image_freqres=args.image_freqres,
            image_timeres=args.image_timeres,
            pol=args.pol,
            clean_threshold=args.clean_threshold,
            use_multiscale=args.use_multiscale,
            cutout_rsun=args.cutout_rsun,
            make_overlay=args.make_overlay,
            make_TB=args.make_TB,
            save_hpc=args.save_hpc,
            make_msplot=args.make_msplot,
            # Compress image
            compress_image=args.compress_image,
            keep_original=args.keep_original,
            # Resource settings
            cpu_frac=args.cpu_frac,
            mem_frac=args.mem_frac,
            max_worker=nworker,
            keep_backup=args.keep_backup,
            keep_calibrated_ms=args.keep_calibrated_ms,
            # Remote logging
            remote_logger=args.remote_logger,
            jobid=jobid,
            job_password=args.job_password,
            adaptive=adaptive,
            verbose=args.verbose,
        )
        if msg == 0:
            print_banner("P-AIRCARS execution is finished: Successful.")
        else:
            print_banner("P-AIRCARS execution is finished: Unsuccessful.")
    except Exception:
        traceback.print_exc()
    finally:
        time.sleep(5)
        print("Closing clusters...")
        with contextlib.suppress(Exception):
            dask_client.cancel(dask_client.futures)
        with contextlib.suppress(Exception):
            dask_client.close()
        with contextlib.suppress(Exception):
            dask_cluster.close()
        os.system(f"rm -rf {dask_dir}")
        print("Cluster closed.")
        if jobfile_name != "" and os.path.exists(jobfile_name):
            os.system(f"rm -rf {jobfile_name}")
