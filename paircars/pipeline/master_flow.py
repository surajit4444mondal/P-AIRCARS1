import os
import logging
import psutil
import numpy as np
import argparse
import traceback
import copy
import time
import glob
import sys
import os
import socket
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass
from casatools import msmetadata
from datetime import datetime as dt
from multiprocessing import Process, Event
from paircars.utils import *
from dask.distributed import get_client
from dotenv import load_dotenv
from prefect import flow, task
from prefect.context import get_run_context
from prefect_dask.task_runners import DaskTaskRunner
from prefect_dask import get_dask_client
from paircars.pipeline import (
    mwa_make_ds,
    do_target_split,
    flagging,
    import_model,
    basic_cal,
    do_apply_basiccal,
    do_sidereal_cor,
    do_selfcal,
    do_apply_selfcal,
    do_imaging,
    mwa_pbcor,
)

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


@task(name="making_dynamic_spectra", retries=2, retry_delay_seconds=10, log_prints=True)
def run_ds_jobs(
    mslist,
    metafits,
    workdir,
    outdir,
    plot_quantity="TB",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Make dynamic spectra of the solar target

    Parameters
    ----------
    mslist : str
        Measurement sets (comma separated)
    metafits : str
        Metafits file
    workdir : str
        Name of the work directory
    outdir : str
        Name of the output directory
    plot_quantity : str, optional
        Plot quantity (TB or flux)
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message
    """
    ds_basename = "ds_targets"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{ds_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_ds = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ##################
        print("###########################")
        print("Making dynamic spectra of solar target .....")
        print("###########################")
        ##########################
        # Making dynamic spectrum
        ##########################
        with get_dask_client() as dask_client:
            msg = mwa_make_ds.main(
                mslist,
                metafits,
                workdir,
                outdir,
                plot_quantity=plot_quantity,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=start_remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_ds.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Dynamic spectrum making is failed.")
    else:
        return msg


@task(name="spliting_ms", retries=2, retry_delay_seconds=10, log_prints=True)
def run_target_split_jobs(
    mslist,
    workdir,
    datacolumn="data",
    timeres=-1,
    freqres=-1,
    prefix="target",
    time_window=-1,
    time_interval=-1,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Split measurement set

    Parameters
    ----------
    mslist: str
        Name of the measurement sets (comma separated)
    workdir : str
        Working directory
    datacolumn : str, optional
        Data column
    timeres : float, optional
        Time bin to average in seconds
    freqres : float, optional
        Frequency averaging in MHz
    prefix : str, optional
        Prefix of splited targets
    time_window : float, optional
        Time window in seconds
    time_interval : float, optional
        Time interval in seconds
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for spliting measurement set
    """
    split_basename = f"split_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{split_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_split = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ############
        print("###########################")
        print(f"Spliting {prefix} .....")
        print("###########################")
        ##################
        # Spliting ms
        ##################
        with get_dask_client() as dask_client:
            msg = do_target_split.main(
                mslist,
                workdir=workdir,
                datacolumn=datacolumn,
                time_window=time_window,
                time_interval=time_interval,
                freqres=freqres,
                timeres=timeres,
                prefix=prefix,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_split.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Spliting measurement set into coarse channels is failed.")
    else:
        return msg


@task(name="flagging", retries=2, retry_delay_seconds=10, log_prints=True)
def run_flag(
    mslist,
    metafits,
    workdir,
    outdir,
    flag_calibrators=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Run flagging jobs

    Parameters
    ----------
    mslist: str
        Name of the measurement sets (comma separted)
    metafits : str
        Metafits file
    workdir : str
        Working directory
    outdir : str
        Output directory
    flag_calibrators : bool, optional
        Flag calibrator fields
    jobid : int, optional
        Job ID
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message
    """
    if flag_calibrators:
        flagdimension = "freqtime"
        flagfield_type = "cal"
        use_tfcrop = True
        flag_basename = f"flagging_{flagfield_type}_calibrator"
    else:
        flagdimension = "freq"
        flagfield_type = "target"
        use_tfcrop = False
        flag_basename = f"flagging_{flagfield_type}_target"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{flag_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_flag = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ##############
        print("###########################")
        print("Flagging ....")
        print("###########################")
        ########################
        # Calibrator ms flagging
        ########################
        with get_dask_client() as dask_client:
            msg = flagging.main(
                mslist,
                metafits,
                workdir=workdir,
                outdir=outdir,
                datacolumn="DATA",
                flag_bad_ants=True,
                flag_bad_spw=True,
                use_tfcrop=use_tfcrop,
                flag_autocorr=True,
                flagdimension=flagdimension,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_grac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_flag.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Flagging is failed.")
    else:
        return msg


@task(
    name="importing_model_visibilities",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True,
)
def run_import_model(
    mslist,
    metafits,
    workdir,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Importing calibrator models

    Parameters
    ----------
    mslist : str
        Name of the measurement sets (comma separated)
    metafits : str
        Metafits file
    workdir : str
        Working directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message
    """
    model_basename = "modeling"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{model_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_model = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ##############
        print("###########################")
        print("Importing model visibilities ....")
        print("###########################")
        ########################
        # Calibrator ms flagging
        ########################
        with get_dask_client() as dask_client:
            msg = import_model.main(
                mslist,
                metafits,
                workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_model.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Importing calibrator model is failed.")
    else:
        return msg


@task(name="basic_calibration", retries=2, retry_delay_seconds=10, log_prints=True)
def run_basic_cal_jobs(
    mslist,
    workdir,
    outdir,
    perform_polcal=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    keep_backup=False,
    remote_log=False,
):
    """
    Perform basic calibration

    Parameters
    ----------
    mslist: str
        Name of the measurement sets (comma seperated)
    workdir : str
        Working directory
    outdir : str
        Output directory
    perform_polcal : bool, optional
        Perform full polarization calibration
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    keep_backup : bool, optional
        Keep backups
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for basic calibration
    """
    cal_basename = "basic_cal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{cal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_cal = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ##############
        print("###########################")
        print("Performing basic calibration .....")
        print("###########################")
        ########################
        # Basic calibration
        ########################
        with get_dask_client() as dask_client:
            msg = basic_cal.main(
                mslist,
                workdir,
                outdir,
                perform_polcal=perform_polcal,
                keep_backup=keep_backup,
                start_remote_log=remote_log,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_cal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Basic calibration is failed.")
    else:
        return msg


@task(
    name="applying_basic_calibration",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True,
)
def run_apply_basiccal_sol(
    mslist,
    calibrator_metafits,
    target_metafits,
    workdir,
    caldir,
    overwrite_datacolumn=True,
    applymode="calflag",
    prefix="target",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Apply basic calibration solutions on splited target scans

    Parameters
    ----------
    mslist: str
        Target measurement set list (comma separated)
    calibrator_metafits : str
        Calibrator metafits
    target_metafits : str
        Target metafits
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    overwrite_datacolumn : bool
        Overwrite data column or not
    applymode : str, optional
        Applycal mode
    prefix : str, optional
        Applying on target of selfcal ms
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    """
    applycal_basename = f"apply_basiccal_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applycal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_apply = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ######################
        print("###########################")
        print("Applying basic calibration solutions on solar target .....")
        print("###########################")
        ######################
        # Applying basic calibration
        ######################
        with get_dask_client() as dask_client:
            msg = do_apply_basiccal.main(
                mslist,
                calibrator_metafits,
                target_metafits,
                workdir,
                caldir,
                applymode=applymode,
                overwrite_datacolumn=overwrite_datacolumn,
                start_remote_log=remote_log,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_apply.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Applying basic calibration solutions is failed.")
    else:
        return msg


@task(
    name="solar_sidereal_correction", retries=2, retry_delay_seconds=10, log_prints=True
)
def run_solar_siderealcor_jobs(
    mslist,
    workdir,
    prefix="target",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Apply sidereal motion correction of the Sun

    Parameters
    ----------
    mslist: str
        List of the measurement sets (comma separated)
    workdir : str
        Work directory
    prefix : str, optional
        Measurement set prefix
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message
    """
    sidereal_basename = f"cor_sidereal_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{sidereal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_sidereal = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        #######################
        print("###########################")
        print("Correcting sidereal motion .....")
        print("###########################")
        #######################
        # Sidereal motion correction
        #######################
        with get_dask_client() as dask_client:
            msg = do_sidereal_cor.main(
                mslist,
                workdir=workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_sidereal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Solar sidereal motion correction is failed.")
    else:
        return msg


@task(name="selfcal", retries=2, retry_delay_seconds=10, log_prints=True)
def run_selfcal_jobs(
    mslist,
    workdir,
    caldir,
    cal_applied,
    start_thresh=5.0,
    stop_thresh=3.0,
    max_iter=100,
    max_DR=1000,
    min_iter=2,
    conv_frac=0.3,
    solint="60s",
    do_apcal=True,
    solar_selfcal=True,
    keep_backup=False,
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    applymode="calonly",
    min_tol_factor=10.0,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Self-calibration on target scans

    Parameters
    ----------
    mslist: str
        Target measurement set list (comma separated)
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    cal_applied : bool
        Whether calibration solutions are applied or not
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    start_threshold : int, optional
        Start CLEAN threhold
    end_threshold : int, optional
        End CLEAN threshold
    max_iter : int, optional
        Maximum numbers of selfcal iterations
    max_DR : float, optional
        Maximum dynamic range
    min_iter : int, optional
        Minimum numbers of seflcal iterations at different stages
    conv_frac : float, optional
        Dynamic range fractional change to consider as converged
    uvrange : str, optional
        UV-range for calibration
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    weight : str, optional
        Image weighitng scheme
    robust : float, optional
        Robustness parameter for briggs weighting
    solint : str, optional
        Solutions interval
    do_apcal : bool, optional
        Perform ap-selfcal or not
    min_tol_factor : float, optional
        Minimum tolerance in temporal variation in imaging
    applymode : str, optional
        Solution apply mode
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for self-calibration
    """
    selfcal_basename = "selfcal_targets"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{selfcal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_selfcal = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ########################
        print("###########################")
        print("Performing self-calibration of solar targets .....")
        print("###########################")
        ########################
        # Selfcal jobs
        ########################
        with get_dask_client() as dask_client:
            msg = do_selfcal.main(
                mslist,
                workdir,
                caldir,
                cal_applied,
                start_thresh=float(start_thresh),
                stop_thresh=float(stop_thresh),
                max_iter=float(max_iter),
                max_DR=float(max_DR),
                min_iter=float(min_iter),
                conv_frac=float(conv_frac),
                solint=solint,
                uvrange=uvrange,
                minuv=float(minuv),
                weight=weight,
                robust=float(robust),
                applymode=applymode,
                min_tol_factor=float(min_tol_factor),
                do_apcal=do_apcal,
                solar_selfcal=solar_selfcal,
                keep_backup=keep_backup,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_selfcal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Self-calibration is failed.")
    else:
        return msg


@task(
    name="applying_self-calibration", retries=2, retry_delay_seconds=10, log_prints=True
)
def run_apply_selfcal_sol(
    target_mslist,
    workdir,
    caldir,
    overwrite_datacolumn=True,
    applymode="calflag",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Apply self-calibration solutions on splited target scans

    Parameters
    ----------
    target_mslist: str
        Target measurement set list (comma separated)
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    applymode : str, optional
        Applycal mode
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    overwrite_datacolumn : bool
        Overwrite data column or not
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    """
    applycal_basename = "apply_selfcal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applycal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_applyselfcal = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ##################
        print("###########################")
        print("Applying self-calibration solutions on targets .....")
        print("###########################")
        ########################
        # Applying self-calibration
        ########################
        with get_dask_client() as dask_client:
            msg = do_apply_selfcal.main(
                mslist,
                workdir,
                caldir,
                applymode=applymode,
                overwrite_datacolumn=overwrite_datacolumn,
                start_remote_log=remote_log,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_applyselfcal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Applying self-calibration solutions is failed.")
    else:
        return msg


@task(name="imaging", retries=2, retry_delay_seconds=10, log_prints=True)
def run_imaging_jobs(
    mslist,
    workdir,
    outdir,
    freqrange="",
    timerange="",
    minuv=-1,
    weight="briggs",
    robust=0.0,
    pol="IQUV",
    freqres=-1,
    timeres=-1,
    band="",
    threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    cutout_rsun=2.5,
    make_overlay=True,
    savemodel=False,
    saveres=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Imaging on target scans

    Parameters
    ----------
    mslist: str
        Target measurement set list (comma separated)
    workdir : str
        Working directory
    outdir : str
        Output image directory
    freqrange : str, optional
        Frequency range to image in MHz
    timerange : str, optional
        Time range to image (YYYY/MM/DD/hh:mm:ss~YYYY/MM/DD/hh:mm:ss)
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    pol : str, optional
        Stokes parameters to image
    freqres : float, optional
        Frequency resolution of spectral chunk in MHz (default : -1, no spectral chunking)
    timeres : float, optional
        Time resolution of temporal chunks in MHz (default : -1, no temporal chunking)
    band : str, optional
        Band name
    threshold : float, optional
        CLEAN threshold
    use_multiscale : bool, optional
        Use multiscale or not
    use_solar_mask : bool, optional
        Use solar mask or not
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 2.5 solar radii)
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for imaging
    """
    imaging_basename = "imaging_targets"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{imaging_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_imaging = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ######################
        print("###########################")
        print("Performing imaging of target scans .....")
        print("###########################")
        #######################
        # Performing imaging
        #######################
        with get_dask_client() as dask_client:
            msg = do_imaging.main(
                mslist,
                workdir,
                outdir,
                freqrange=freqrange,
                timerange=timerange,
                pol=pol,
                freqres=float(freqres),
                timeres=float(timeres),
                weight=weight,
                robust=float(robust),
                minuv=float(minuv),
                threshold=float(threshold),
                band=band,
                cutout_rsun=float(cutout_rsun),
                use_multiscale=use_multiscale,
                use_solar_mask=use_solar_mask,
                savemodel=savemodel,
                saveres=saveres,
                make_overlay=make_overlay,
                start_remote_log=remote_log,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                jobid=jobid,
                logfile=logfile,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_imaging.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Imaging is failed.")
    else:
        return msg


@task(name="applying_primary_beam", retries=2, retry_delay_seconds=10, log_prints=True)
def run_apply_pbcor(
    imagedir,
    metafits,
    workdir,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
):
    """
    Apply primary beam corrections on all images

    Parameters
    ----------
    imagedir: str
        Image directory name
    metafits : str
        Metafits file
    workdir : str
        Work directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger

    Returns
    -------
    int
        Success message for applying primary beam correction on all images
    """
    applypbcor_basename = "apply_pbcor"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applypbcor_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_pbcor = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    try:
        ###################
        print("###########################")
        print("Applying primary beam corrections on all images .....")
        print("###########################")
        #####################
        # Applying primary beam correction
        #####################
        with get_dask_client() as dask_client:
            msg = mwa_pbcor.main(
                imagedir,
                metafits,
                workdir=workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
            )
    finally:
        stop_event.set()
        log_thread_pbcor.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Primary beam correction is failed.")
    else:
        return msg


@flow(
    name="P-AIRCARS Master control",
    version="3.0",
    description="Calibration and Imaging Pipeline for MWA Solar Observation",
    log_prints=True,
)
def master_control(
    target_datadir,
    target_metafits,
    workdir,
    outdir,
    calibrator_datadir="",
    calibrator_metafits="",
    solar_data=True,
    # Pre-calibration
    do_forcereset_weightflag=False,
    do_cal_partition=True,
    do_cal_flag=True,
    do_import_model=True,
    # Basic calibration
    do_basic_cal=True,
    do_applycal=True,
    # Target data preparation
    do_target_split=True,
    target_scans=[],
    freqrange="",
    timerange="",
    uvrange="",
    # Polarization calibration
    do_polcal=False,
    # Self-calibration
    do_selfcal=True,
    do_selfcal_split=True,
    do_apply_selfcal=True,
    do_ap_selfcal=True,
    solint="5min",
    # Sidereal correction
    do_sidereal_cor=False,
    # Dynamic spectra
    make_ds=True,
    # Imaging
    do_imaging=True,
    do_pbcor=True,
    weight="briggs",
    robust=0.0,
    minuv=0,
    image_freqres=-1,
    image_timeres=-1,
    pol="IQUV",
    clean_threshold=1.0,
    use_multiscale=True,
    cutout_rsun=2.5,
    make_overlay=True,
    # Resource settings
    cpu_frac=0.8,
    mem_frac=0.8,
    max_worker=-1,
    keep_backup=False,
    # Remote logging
    remote_logger=False,
    jobid=None,
):
    """
    Master controller of the entire pipeline

    Parameters
    ----------
    msname : str
        Measurement set name
    workdir : str
        Work directory path
    outdir : str
        Output directory
    solar_data : bool, optional
        Whether it is solar data or not

    do_forcereset_weightflag : bool, optional
        Reset weights and flags of the input ms
    do_cal_partition : bool, optional
        Make calibrator multi-MS
    do_cal_flag : bool, optional
        Perform flagging on calibrator
    do_import_model : bool, optional
        Import model visibilities of flux and polarization calibrators

    do_basic_cal : bool, optional
        Perform basic calibration
    do_applycal : bool, optional
        Apply basic calibration on target scans

    do_target_split : bool, optional
        Split target scans into chunks
    target_scans : list, optional
        Target scans to self-cal and image
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
    do_selfcal_split : bool, optional
        Split data after each round of self-calibration
    do_apply_selfcal : bool, optional
        Apply self-calibration solutions
    do_ap_selfcal : bool, optional
        Perform amplitude-phase self-cal or not
    solint : str, optional
        Solution intervals in self-cal

    do_sidereal_cor : bool, optional
        Perform solar sidereal motion correction or not

    make_ds : bool, optional
        Make dynamic spectra

    do_imaging : bool, optional
        Perform final imaging
    do_pbcor : bool, optional
        Perform primary beam correction
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting (-1 to 1)
    minuv : float, optional
        Minimum UV-lambda for final imaging
    image_freqres : float, optional
        Image frequency resolution in MHz (-1 means full bandwidth)
    image_timeres : float, optional
        Image temporal resolution in seconds (-1 means full scan duration)
    pol : str, optional
        Stokes parameters of final imaging
    clean_threshold : float, optional
        CLEAN threshold of final imaging
    use_multiscale : bool, optional
        Use multiscale scales or not
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 2.5 solar radii)
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay

    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    max_worker: int, optional
        Maximum workers
    keep_backup : bool, optional
        Keep backup of self-cal rounds and final models and residual images

    remote_logger : bool, optional
        Enable remote logging of the pipeline status
    jobid : str, optional
        Job ID

    Returns
    -------
    int
        Success message
    """
    #############################################
    # Listing target and calibrator ms
    # Determining where to use calibrator or not
    #############################################
    target_mslist = glob.glob(f"{target_datadir}/*.ms")
    target_header = fits.getheader(target_metafits)
    target_obsid = target_header["GPSTIME"]
    target_freq_config = target_header["CHANNELS"]

    calibrator_mslist = glob.glob(f"{calibrator_datadir}/*.ms")
    if len(calibrator_mslist) == 0:
        want_cont = input(
            f"No calibrator observation is provided. Do you want to continue without calibrator? Y/Yes."
        )
        if want_cont.upper() == "YES" or want_cont.upper() == "Y":
            has_cal = False
        else:
            return 1
    elif os.path.exists(calibrator_metafits):
        calibrator_header = fits.getheader(calibrator_metafits)
        calibrator_obsid = calibrator_header["GPSTIME"]
        calibrator_freq_config = calibrator_header["CHANNELS"]
        if np.abs(calibrator_obsid - target_obsid) > 12 * 3600:
            want_cont = input(
                "Calibrator observations were taken 12 hours apart. Do you want to continue without calibrator? Y/Yes."
            )
            if want_cont.upper() == "YES" or want_cont.upper() == "Y":
                has_cal = False
            else:
                return 1
        elif target_freq_config != calibrator_freq_config:
            print(f"Target coarse channels: {target_freq_config}.")
            print(f"Calibrator coarse channels: {calibrator_freq_config}.")
            want_cont = input(
                "Calibrator and target frequency configuration is different. Do you want to continue without calibrator? Y/Yes."
            )
            if want_cont.upper() == "YES" or want_cont.upper() == "Y":
                has_cal = False
            else:
                return 1
        else:
            has_cal = True
    else:
        want_cont = input(
            f"Calibrator ms is available. No calibrator metafits is provided. Do you want to continue without calibrator? Y/Yes."
        )
        if want_cont.upper() == "YES" or want_cont.upper() == "Y":
            has_cal = False
        else:
            return 1

    ###################################
    # Preparing working directories
    ###################################
    print("Preparing working directories....")
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(target_mslist[0])) + "/workdir"
    workdir = workdir.rstrip("/")
    print(f"Work directory: {workdir}.")

    #################################
    # Setup logger
    #################################
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/main.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id, flow_name, logfile, poll_interval=3, stop_event=stop_event
    )
    dask_dir = None
    try:
        dask_client = get_client()
        dask_cluster = dask_client.cluster
    except:
        dask_client, dask_cluster, dask_dir = get_local_dask_cluster(
            2, workdir, cpu_frac=cpu_frac, mem_frac=mem_frac
        )
    current_worker = get_total_worker(dask_cluster)

    #####################################
    # Initiating paircars data
    #####################################
    from paircars.pipeline.init_data import init_paircars_data

    init_paircars_data()

    ###################################################
    # Measurement set check and other working directory
    ###################################################
    if outdir == "":
        outdir = workdir
    outdir = outdir.rstrip("/")
    caldir = f"{outdir}/caltables"
    caldir = caldir.rstrip("/")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(caldir, exist_ok=True)

    if max_worker < 1:
        if cpu_frac > 0.8:
            cpu_frac = 0.8
        max_worker = int(psutil.cpu_count() * cpu_frac)

    try:
        ####################################
        # Job and process IDs
        ####################################
        pid = os.getpid()
        if jobid is None:
            jobid = get_jobid()
        main_job_file = save_main_process_info(
            pid,
            jobid,
            target_datadir,
            os.path.abspath(workdir),
            os.path.abspath(outdir),
            cpu_frac,
            mem_frac,
        )
        print("###########################")
        print(f"P-AIRCARS Job ID: {jobid}")
        print(f"Work directory: {workdir}")
        print(f"Final product directory: {outdir}")
        print("###########################")
        #####################################
        # Moving into work directory
        #####################################
        os.chdir(workdir)
        if remote_logger:
            trial = 0
            while trial <= 5:
                remote_link = get_remote_logger_link()
                if remote_link != "":
                    break
                else:
                    time.sleep(5)
                    trial += 1
            if remote_link == "":
                print("Please provide a valid remote link.")
                remote_logger = False

        if not remote_logger:
            emails = get_emails()
            timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            if emails != "":
                email_subject = f"P-AIRCARS Logger Details: {timestamp}"

                email_msg = (
                    f"P-AIRCARS user,\n\n"
                    f"P-AIRCARS Job ID: {jobid}\n\n"
                    f"Best,\n"
                    f"P-AIRCARS"
                )
                from paircars.data.sendmail import (
                    send_paircars_notification as send_notification,
                )

                success_msg, error_msg = send_notification(
                    emails, email_subject, email_msg
                )
        else:
            ####################################
            # Job name and logging password
            ####################################
            hostname = socket.gethostname()
            timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            job_name = f"{hostname} :: {timestamp} :: {target_obsid}"
            timestamp1 = dt.utcnow().strftime("%Y%m%dT%H%M%S")
            remote_job_id = f"{hostname}_{timestamp1}_{target_obsid}"
            password = generate_password()
            np.save(
                f"{workdir}/jobname_password.npy",
                np.array([job_name, password], dtype="object"),
            )
            print(
                "############################################################################"
            )
            print(remote_link)
            print(f"Job ID: {job_name}")
            print(f"Remote access password: {password}")
            print(
                "#############################################################################"
            )
            emails = get_emails()
            if emails != "":
                email_subject = f"P-AIRCARS Logger Details: {timestamp}"

                email_msg = (
                    f"P-AIRCARS user,\n\n"
                    f"P-AIRCARS Job ID: {jobid}\n\n"
                    f"Remote logger Job ID: {job_name}\n"
                    f"Remote access password: {password}\n\n"
                    f"Best,\n"
                    f"P-AIRCARS"
                )
                from paircars.data.sendmail import (
                    send_paircars_notification as send_notification,
                )

                success_msg, error_msg = send_notification(
                    emails, email_subject, email_msg
                )

        #####################################
        # Settings for solar data
        #####################################
        if solar_data:
            if not use_solar_mask:
                print("Use solar mask during CLEANing.")
                use_solar_mask = True
            if not solar_selfcal:
                solar_selfcal = True
            full_FoV = False
        else:
            if do_noise_cal:
                print(
                    "Turning off noise diode based calibration for non-solar observation."
                )
                do_noise_cal = False
            if use_solar_mask:
                print("Stop using solar mask during CLEANing.")
                use_solar_mask = False
            if solar_selfcal:
                solar_selfcal = False
            full_FoV = True

        #####################################################################
        # Checking if ms is full pol for polarization calibration and imaging
        #####################################################################
        if do_polcal:
            print(
                "Checking measurement set suitability for polarization calibration...."
            )
            msmd = msmetadata()
            msmd.open(msname)
            npol = msmd.ncorrforpol()[0]
            msmd.close()
            if npol < 4:
                print(
                    "Measurement set is not full-polar. Do not performing polarization analysis."
                )
                do_polcal = False

        #################################################
        # Determining maximum allowed frequency averaging
        #################################################
        print("Estimating optimal frequency averaging....")
        max_freqres = calc_bw_smearing_freqwidth(msname, full_FoV=full_FoV)
        if image_freqres > 0:
            freqavg = round(min(image_freqres, max_freqres), 1)
        else:
            freqavg = round(max_freqres, 1)
        total_ncoarse = 0
        for ms in target_mslist:
            ncoarse = get_ncoarse(ms)
            total_ncoarse += ncoarse

        ################################################
        # Determining maximum allowed temporal averaging
        ################################################
        print("Estimating optimal temporal averaging....")
        if solar_data:  # For solar data, it is assumed Sun is tracked.
            max_timeres = calc_time_smearing_timewidth(msname)
        else:
            max_timeres = min(
                calc_time_smearing_timewidth(msname), max_time_solar_smearing(msname)
            )
        if image_timeres > (2 * 3660):  # If more than 2 hours
            print(
                f"Image time integration is more than 2 hours, which may cause smearing due to solar differential rotation."
            )
        if image_timeres > 0:
            timeavg = round(min(image_timeres, max_timeres), 1)
        else:
            timeavg = round(max_timeres, 1)

        #############################
        # Reset any previous weights
        ############################
        print("Resetting previous flags and weights....")
        cpu_usage = psutil.cpu_percent(interval=1)  # Average over 1 second
        total_cpus = psutil.cpu_count(logical=True)
        available_cpus = int(total_cpus * (1 - cpu_usage / 100.0))
        available_cpus = max(1, available_cpus)  # Avoid zero workers
        for msname in target_mslist:
            reset_weights_and_flags(
                msname, n_threads=available_cpus, force_reset=do_forcereset_weightflag
            )
        for msname in calibrator_mslist:
            reset_weights_and_flags(
                msname, n_threads=available_cpus, force_reset=do_forcereset_weightflag
            )

        #######################################
        # Run dynamic spectra making
        #######################################
        if make_ds:
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_maskms = run_ds_jobs.with_options(
                task_run_name=f"making_dynamic_spectra_{jobid}",
            ).submit(
                target_mslist,
                target_metafits,
                workdir,
                outdir,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_maskms.result()
            except Exception as e:
                print("!!! WARNING : Error in making dynamic spectra. !!!")
                traceback.print_exc()
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        ##############################
        # Run spliting jobs
        ##############################
        # If basic calibration is requested and calibrator ms and metafits are present
        future_cal_split = None
        if do_basic_cal and has_cal:
            prefix = "calibrator"
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_cal_split = run_target_split_jobs.with_options(
                task_run_name=f"spliting_{prefix}_{jobid}"
            ).submit(
                calibrator_mslist,
                workdir,
                datacolumn="data",
                timeres=timeavg,
                freqres=freqavg,
                prefix=prefix,
                time_window=-1,
                time_interval=-1,
                jobid=jobid,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                remote_log=remote_log,
            )
            try:
                msg = future_cal_split.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in spliting calibrator measurement sets. !!!!"
                )
                traceback.print_exc()
                return 1
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)
            split_cal_mslist = glob.glob(workdir + "/calibrators_spw_*.ms")
            if len(split_cal_mslist) == 0:
                print("No splited measurement set is present for basic calibration.")
                has_cal = False

        ###################################################
        # Start spliting selfcal ms, if worker available
        ###################################################
        future_selfcal_split = None
        if (
            do_selfcal and do_selfcal_split and max_worker > 4
        ):  # At-least four worker possible
            prefix = "selfcal"
            try:
                time_interval = float(solint)
            except BaseException:
                if "s" in solint:
                    time_interval = float(solint.split("s")[0])
                elif "min" in solint:
                    time_interval = float(solint.split("min")[0]) * 60
                elif solint == "int":
                    time_interval = image_timeres
                else:
                    time_interval = -1
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_selfcal_split = run_target_split_jobs.with_options(
                task_run_name=f"spliting_{prefix}_{jobid}"
            ).submit(
                target_mslist,
                workdir,
                datacolumn="data",
                timeres=timeavg,
                freqres=freqavg,
                prefix=prefix,
                time_window=min(10, time_interval),
                time_interval=time_interval,
                jobid=jobid,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                remote_log=remote_log,
            )

        ##################################
        # Run flagging jobs on calibrators
        ##################################
        # Only if basic calibration is requested
        if do_cal_flag and do_basic_cal and has_cal:
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_flag = run_flag.with_options(
                task_run_name=f"flagging_{jobid}"
            ).submit(
                split_cal_mslist,
                workdir,
                outdir,
                flag_calibrators=True,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_flag.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Flagging error. Examine calibration solutions with caution. !!!!"
                )
                traceback.print_exc()
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        #################################
        # Import model
        #################################
        # Only if basic calibration is requested
        if do_import_model and do_basic_cal and has_cal:
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_import_model = run_import_model.with_options(
                task_run_name=f"importing_model_visibilities_{jobid}"
            ).submit(
                split_cal_mslist,
                workdir,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_import_model.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in importing calibrator models. Not continuing calibration. !!!!"
                )
                traceback.print_exc()
                return 1
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        ###############################
        # Run basic calibration
        ###############################
        if do_basic_cal and has_cal:
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_basical = run_basic_cal_jobs.with_options(
                task_run_name=f"basic_calibration_{jobid}"
            ).submit(
                split_cal_mslist,
                workdir,
                outdir,
                perform_polcal=do_polcal,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                keep_backup=keep_backup,
                remote_log=remote_logger,
            )
            try:
                msg = future_basical.result()
                for calms in split_cal_mslist:
                    msg, ms_diag_plot = plot_ms_diagnostics(
                        calms,
                        outdir=f"{outdir}/diagnostic_plots",
                        dask_client=dask_client,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                    )
                    if msg == 0:
                        print(
                            f"Calibrator diagnostic plots are saved in : {ms_diag_plot}"
                        )
                    else:
                        print(
                            "Error in creating diagnostic plots for calibrator measurement set: {calms}."
                        )
                caltables = glob.glob(f"{caldir}/*cal")
                for caltable in caltables:
                    msg, caltable_diag_plot = plot_caltable_diagnostics(
                        caltable, outdir=f"{outdir}/diagnostic_plots"
                    )
                    if msg == 0:
                        print(
                            f"Diagnostic plots for caltable {caltable} are saved in : {caltable_diag_plot}."
                        )
                    else:
                        print(
                            f"Error in creating diagnostic plots for caltable {caltable}."
                        )
            except Exception as e:
                print(
                    "!!!! WARNING: Error in basic calibration. Starting without basic calibration. !!!!"
                )
                traceback.print_exc()
                has_cal = False
            finally:
                scale_worker_and_wait(dask_cluster, nworker)

        ##########################################
        # Checking presence of necessary caltables
        ##########################################
        if len(glob.glob(f"{caldir}/*.bcal")) == 0:
            print(f"No bandpass table is present in calibration directory : {caldir}.")
            has_cal = False

        ############################################
        # Spliting for self-cals
        ############################################
        # Spliting only if self-cal is requested
        if not do_selfcal_split and do_selfcal:
            selfcal_target_mslist = glob.glob(workdir + "/selfcals_spw*.ms")
            if len(selfcal_target_mslist) == 0:
                print(
                    "No measurement set is present for self-calibration. Spliting them.."
                )
                do_selfcal_split = True

        ###################################################
        # Start spliting selfcal ms, if not started already
        ###################################################
        if do_selfcal and do_selfcal_split and future_selfcal_split is None:
            prefix = "selfcal"
            try:
                time_interval = float(solint)
            except BaseException:
                if "s" in solint:
                    time_interval = float(solint.split("s")[0])
                elif "min" in solint:
                    time_interval = float(solint.split("min")[0]) * 60
                elif solint == "int":
                    time_interval = image_timeres
                else:
                    time_interval = -1
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_selfcal_split = run_target_split_jobs.with_options(
                task_run_name=f"spliting_{prefix}_{jobid}"
            ).submit(
                target_mslist,
                workdir,
                datacolumn="data",
                timeres=timeavg,
                freqres=freqavg,
                prefix=prefix,
                time_window=min(10, time_interval),
                time_interval=time_interval,
                jobid=jobid,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                remote_log=remote_log,
            )

        ######################################
        # Checking status of self-cal split
        ######################################
        if future_selfcal_split is not None:
            print("Checking status of spliting of target for selfcal ...")
            try:
                msg = future_selfcal_split.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
                )
                do_selfcal = False
                traceback.print_exc()
            finally:
                scale_worker_and_wait(dask_cluster, 1)

        ###################
        # Spliting targets
        ###################
        # If corrected data is requested or imaging is requested
        future_split = None
        if (
            do_target_split and (do_applycal or do_imaging) and max_worker > 4
        ):  # Only if at-least 4 workers
            prefix = "target"
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_split = run_target_split_jobs.with_options(
                task_run_name=f"spliting_{prefix}_{jobid}"
            ).submit(
                target_mslist,
                workdir,
                datacolumn="data",
                freqres=freqavg,
                timeres=timeavg,
                prefix=prefix,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )

        ####################################
        # Filtering any corrupted ms
        #####################################
        selfcal_target_mslist = glob.glob(workdir + "/selfcals_spw*.ms")
        if (selfcal_target_mslist) == 0:
            print(
                "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
            )
            do_selfcal = False
        if do_selfcal:
            print("Checking measurement sets before spawning self-calibrations....")
            filtered_mslist = []  # Filtering in case any ms is corrupted
            for ms in selfcal_target_mslist:
                checkcol = check_datacolumn_valid(ms)
                if checkcol:
                    filtered_mslist.append(ms)
                else:
                    print(f"Issue in : {ms}")
                    os.system(f"rm -rf {ms}")
            selfcal_mslist = filtered_mslist
            if len(selfcal_mslist) == 0:
                print(
                    "No splited target scan ms are available in work directory for selfcal. Not continuing further for selfcal."
                )
                do_selfcal = False
            print(f"Selfcal mslist : {[os.path.basename(i) for i in selfcal_mslist]}")

        #########################################################
        # Applying solutions on targets for self-calibration
        #########################################################
        if do_selfcal and has_cal:  # If calibrator solutions are available
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, len(selfcal_mslist) + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_apply_basical_selfcal = run_apply_basiccal_sol.with_options(
                task_run_name=f"applying_basiccal_selfcal_{jobid}"
            ).submit(
                selfcal_mslist,
                calibrator_metafits,
                target_metafits,
                workdir,
                caldir,
                overwrite_datacolumn=False,
                applymode="calflag",
                prefix="selfcal",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_apply_basical_selfcal.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in applying basic calibration solutions on target. Not continuing further for selfcal.!!!!"
                )
                do_selfcal = False
                traceback.print_exc()
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        ########################################
        # Performing self-calibration
        ########################################
        if do_selfcal:
            os.system(
                "rm -rf " + workdir + "/*selfcal " + workdir + "/caltables/*selfcal*"
            )
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, len(selfcal_mslist) + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            if do_sidereal_cor:
                future_sidereal_cor_selfcal = run_solar_siderealcor_jobs.with_options(
                    task_run_name=f"solar_sidereal_correction_{jobid}"
                ).submit(
                    selfcal_mslist,
                    workdir,
                    prefix="selfcal",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                )
                try:
                    msg = future_sidereal_cor_selfcal.result()
                except Exception as e:
                    print("Sidereal correction is not successful.")
                finally:
                    scale_worker_and_wait(dask_cluster, current_worker)

            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, len(selfcal_mslist) + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_selfcal = run_selfcal_jobs.with_options(
                task_run_name=f"selfcal_{jobid}"
            ).submit(
                selfcal_mslist,
                workdir,
                caldir,
                has_cal,
                solint=solint,
                do_apcal=do_ap_selfcal,
                solar_selfcal=solar_selfcal,
                keep_backup=keep_backup,
                uvrange=uvrange,
                weight="briggs",
                robust=0.0,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_selfcal.result()
                for selfcalms in selfcal_mslist:
                    msg, ms_diag_plot = plot_ms_diagnostics(
                        selfcalms,
                        outdir=f"{outdir}/diagnostic_plots",
                        dask_client=dask_client,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                    )
                    if msg == 0:
                        print(
                            "Diagnostic plots for self-cal measurement set {selfcalms} are saved in : {ms_diag_plot}"
                        )
                    else:
                        print(
                            "Error in creating diagnostic plots for self-cal measurement set {selfcalms}."
                        )
            except Exception as e:
                print(
                    "!!!! WARNING: Error in self-calibration on targets. Not applying self-calibration. !!!!"
                )
                do_apply_selfcal = False
                traceback.print_exc()
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        ########################################
        # Checking self-cal caltables
        ########################################
        if do_imaging or do_apply_selfcal:
            selfcal_tables = glob.glob(caldir + "/*selfcal*")
            if len(selfcal_tables) == 0:
                print(
                    "Self-calibration is not performed and no self-calibration caltable is available."
                )
                do_apply_selfcal = False

        #############################################
        # Spliting targets if not started already
        #############################################
        # If corrected data is requested or imaging is requested
        if do_target_split and (do_applycal or do_imaging) and future_split is None:
            prefix = "target"
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, total_ncoarse + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_split = run_target_split_jobs.with_options(
                task_run_name=f"spliting_{prefix}_{jobid}"
            ).submit(
                target_mslist,
                workdir,
                datacolumn="data",
                freqres=freqavg,
                timeres=timeavg,
                prefix=prefix,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )

        ##########################################
        # Checking target spliting is done or not
        ##########################################
        if future_split is not None:
            print("Checking spliting of targets status...")
            try:
                msg = future_split.result()
            except Exception as e:
                print("!!!! WARNING: Error in spliting targets. !!!!")
                traceback.print_exc()
                return 1
            finally:
                scale_worker_and_wait(dask_cluster, 1)

        if do_imaging or do_applycal or do_apply_selfcal:
            split_target_mslist = glob.glob(workdir + "/target_spw*.ms")
            if len(split_target_mslist) == 0:
                print("!!!! WARNING: No target ms are present. !!!!")
                return 1

            ####################################
            # Filtering any corrupted ms
            #####################################
            print(
                "Checking final valid measurement sets before applying solutions and spawning imaging...."
            )
            filtered_mslist = []  # Filtering in case any ms is corrupted
            for ms in split_target_mslist:
                checkcol = check_datacolumn_valid(ms)
                if checkcol:
                    filtered_mslist.append(ms)
                else:
                    print(f"Issue in : {ms}")
                    os.system("rm -rf {ms}")
            split_target_mslist = filtered_mslist
            if len(split_target_mslist) == 0:
                print("No filtered target ms are available in work directory.")
                return 1

            if do_applycal or do_imaging:
                print(
                    f"Target mslist : {[os.path.basename(i) for i in split_target_mslist]}"
                )

        #########################################################
        # Applying basic solutions on target scans
        #########################################################
        if do_applycal:
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, len(split_target_mslist) + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_apply_basical = run_apply_basiccal_sol.with_options(
                task_run_name=f"applying_basiccal_target_{jobid}"
            ).submit(
                split_target_mslist,
                workdir,
                caldir,
                overwrite_datacolumn=True,
                applymode="calflag",
                prefix="target",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_apply_basical.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further.!!!!"
                )
                traceback.print_exc()
                return 1
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

            if do_sidereal_cor:
                current_worker = get_total_worker(dask_cluster)
                nworker = min(max_worker, len(split_target_mslist) + current_worker)
                scale_worker_and_wait(dask_cluster, nworker)
                future_sidereal_cor = run_solar_siderealcor_jobs.with_options(
                    task_run_name=f"solar_sidereal_correction_{jobid}"
                ).submit(
                    split_target_mslist,
                    workdir,
                    prefix="target",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                )
                try:
                    msg = future_sidereal_cor.result()
                except Exception as e:
                    print("!!!! WARNING: Error in applying sidereal correction.!!!!")
                    traceback.print_exc()
                finally:
                    scale_worker_and_wait(dask_cluster, current_worker)

        ########################################
        # Apply self-calibration
        ########################################
        if do_apply_selfcal:
            split_target_mslist = sorted(split_target_mslist)
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, len(split_target_mslist) + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_apply_selfcal = run_apply_selfcal_sol.with_options(
                task_run_name=f"applying_selfcal_{jobid}"
            ).submit(
                split_target_mslist,
                workdir,
                caldir,
                overwrite_datacolumn=False,
                applymode="calonly",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_apply_selfcal.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in applying self-calibration solutions on targets. !!!!"
                )
                traceback.print_exc()
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        #####################################
        # Target ms diagnostic plots
        #####################################
        if do_apply_selfcal or do_imaging:
            if len(split_target_mslist) > 0:
                for targetms in split_target_mslist:
                    msg, ms_diag_plot = plot_ms_diagnostics(
                        targetms,
                        outdir=f"{outdir}/diagnostic_plots",
                        dask_client=dask_client,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                    )
                    if msg == 0:
                        print(
                            "Diagnostic plots for target measurement set {targetms} are saved in : {ms_diag_plot}"
                        )
                    else:
                        print(
                            "Error in creating diagnostic plots for target measurement set {targetms}."
                        )

        ######################################
        # Imaging
        ######################################
        if do_imaging:
            if (
                do_polcal == False
            ):  # Only if do_polcal is False, overwrite to make only Stokes I
                pol = "I"
            current_worker = get_total_worker(dask_cluster)
            nworker = min(max_worker, len(target_mslist) + current_worker)
            scale_worker_and_wait(dask_cluster, nworker)
            future_imaging = run_imaging_jobs.with_options(
                task_run_name=f"imaging_{jobid}"
            ).submit(
                split_target_mslist,
                workdir,
                outdir,
                freqrange=freqrange,
                timerange=timerange,
                minuv=minuv,
                weight=weight,
                robust=float(robust),
                pol=pol,
                freqres=image_freqres,
                timeres=image_timeres,
                threshold=float(clean_threshold),
                use_multiscale=use_multiscale,
                use_solar_mask=use_solar_mask,
                cutout_rsun=cutout_rsun,
                make_overlay=make_overlay,
                savemodel=keep_backup,
                saveres=keep_backup,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
            )
            try:
                msg = future_imaging.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Final imaging on all measurement sets is not successful. Check the image directory. !!!!"
                )
                traceback.print_exc()
                return 1
            finally:
                scale_worker_and_wait(dask_cluster, current_worker)

        ###########################
        # Primary beam correction
        ###########################
        if do_pbcor:
            if weight == "briggs":
                weight_str = f"{weight}_{robust}"
            else:
                weight_str = weight
            if image_freqres == -1 and image_timeres == -1:
                imagedir = outdir + f"/imagedir_f_all_t_all_w_{weight_str}"
            elif image_freqres != -1 and image_timeres == -1:
                imagedir = (
                    outdir
                    + f"/imagedir_f_{round(float(image_freqres),1)}_t_all_w_{weight_str}"
                )
            elif image_freqres == -1 and image_timeres != -1:
                imagedir = (
                    outdir
                    + f"/imagedir_f_all_t_{round(float(image_timeres),1)}_w_{weight_str}"
                )
            else:
                imagedir = (
                    outdir
                    + f"/imagedir_f_{round(float(image_freqres),1)}_t_{round(float(image_timeres),1)}_w_{weight_str}"
                )
            imagedir = imagedir + "/images"
            images = glob.glob(imagedir + "/*.fits")
            if len(images) == 0:
                print(f"No image is present in image directory: {imagedir}")
            else:
                current_worker = get_total_worker(dask_cluster)
                scale_worker_and_wait(dask_cluster, max_worker)
                future_pbcor = run_apply_pbcor.with_options(
                    task_run_name=f"applying_primary_beam_{jobid}"
                ).submit(
                    imagedir,
                    target_metafits,
                    workdir,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                )
                try:
                    msg = future_pbcor.result()
                except Exception as e:
                    print(
                        "!!!! WARNING: Primary beam corrections of the final images are not successful. !!!!"
                    )
                    traceback.print_exc()
                    return 1
                finally:
                    scale_worker_and_wait(dask_cluster, 1)
                print(f"Final image directory: {os.path.dirname(imagedir)}")

        ###########################################
        # Successful exit
        ###########################################
        print(
            f"Calibration and imaging pipeline is successfully run on measurement set : {msname}"
        )
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1
    finally:
        drop_cache(msname)
        drop_cache(workdir)
        drop_cache(outdir)
        stop_event.set()
        log_thread_flow.join(timeout=5)
        scale_worker_and_wait(dask_cluster, 1)
        if dask_dir is not None:
            os.system(f"rm -rf {dask_dir}")


def cli():
    parser = argparse.ArgumentParser(
        description="Run P-AIRCARS for calibration and imaging of solar observations.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    # === Essential parameters ===
    essential = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    essential.add_argument("msname", type=str, help="Measurement set name")
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

    # === Advanced calibration parameters ===
    advanced_cal = parser.add_argument_group(
        "###################\nAdvanced calibration parameters\n###################"
    )
    advanced_cal.add_argument(
        "--solint",
        type=str,
        default="5min",
        help="Solution interval for calibration (e.g. 'int', '10s', '5min', 'inf')",
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

    # === Advanced imaging and calibration parameters ===
    advanced_image = parser.add_argument_group(
        "###################\nAdvanced imaging parameters\n###################"
    )
    advanced_image.add_argument(
        "--target_scans",
        nargs="*",
        type=str,
        default=[],
        help="List of target scans to process (space-separated, e.g. 3 5 7)",
    )
    advanced_image.add_argument(
        "--freqrange",
        type=str,
        default="",
        help="Frequency range in MHz to select during imaging (comma-seperate, e.g. '100~110,130~140')",
    )
    advanced_image.add_argument(
        "--timerange",
        type=str,
        default="",
        help="Time range to select during imaging (comma-seperated, e.g. '2014/09/06/09:30:00~2014/09/06/09:45:00,2014/09/06/10:30:00~2014/09/06/10:45:00')",
    )
    advanced_image.add_argument(
        "--image_freqres",
        type=int,
        default=-1,
        help="Output image frequency resolution in MHz (-1 = full)",
    )
    advanced_image.add_argument(
        "--image_timeres",
        type=int,
        default=-1,
        help="Output image time resolution in seconds (-1 = full)",
    )
    advanced_image.add_argument(
        "--pol",
        type=str,
        default="IQUV",
        help="Stokes parameter(s) to image (e.g. 'I', 'XX', 'RR', 'IQUV')",
    )
    advanced_image.add_argument(
        "--minuv",
        type=float,
        default=0,
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
        help="Clean threshold in sigma for final deconvolution (Note this is not auto-mask)",
    )
    advanced_image.add_argument(
        "--do_pbcor",
        action="store_true",
        help="Apply primary beam correction after imaging",
    )
    advanced_image.add_argument(
        "--cutout_rsun",
        type=float,
        default=2.5,
        help="Field of view cutout radius in solar radii",
    )
    advanced_image.add_argument(
        "--no_solar_mask",
        action="store_false",
        dest="use_solar_mask",
        help="Disable use solar disk mask during deconvolution",
    )
    advanced_image.add_argument(
        "--no_overlay",
        action="store_false",
        dest="make_overlay",
        help="Disable overlay plot on GOES SUVI after imaging",
    )

    # === Advanced options ===
    advanced = parser.add_argument_group(
        "###################\nAdvanced pipeline parameters\n###################"
    )
    advanced.add_argument(
        "--non_solar_data",
        action="store_false",
        dest="solar_data",
        help="Disable solar data mode",
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
        "--no_noise_cal",
        action="store_false",
        dest="do_noise_cal",
        help="Disable noise calibration",
    )
    advanced.add_argument(
        "--no_cal_partition",
        action="store_false",
        dest="do_cal_partition",
        help="Disable calibrator MS partitioning",
    )
    advanced.add_argument(
        "--no_cal_flag",
        action="store_false",
        dest="do_cal_flag",
        help="Disable initial flagging of calibrators",
    )
    advanced.add_argument(
        "--no_import_model",
        action="store_false",
        dest="do_import_model",
        help="Disable model import",
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
        "--no_selfcal_split",
        action="store_false",
        dest="do_selfcal_split",
        help="Disable split for self-calibration",
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
        "--no_solar_selfcal",
        action="store_false",
        dest="solar_selfcal",
        help="Disable solar-specific self-calibration parameters",
    )
    advanced.add_argument(
        "--no_target_split",
        action="store_false",
        dest="do_target_split",
        help="Disable target data split",
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
        "--keep_backup",
        action="store_true",
        help="Keep backup of intermediate steps",
    )
    advanced_resource.add_argument(
        "--no_remote_logger",
        action="store_false",
        dest="remote_logger",
        help="Disable remote logger",
    )

    # === Advanced local system/ per node hardware resource parameters ===
    advanced_hpc = parser.add_argument_group(
        "###################\nAdvanced cluster environment settings\n###################"
    )
    advanced_hpc.add_argument(
        "--cluster",
        action="store_true",
        dest="cluster",
        help="Running in cluster environment",
    )
    advanced_hpc.add_argument(
        "--nworker",
        type=int,
        default=-1,
        help="Number of compute nodes to use",
    )
    advanced_hpc.add_argument(
        "--scheduler",
        type=str,
        default="slurm",
        help="Cluster job scheduler name (slurm, pbs)",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    result = prefect_server_status()
    if result is not True:
        print("Prefect server is not running. Running pipeline in ephemeral mode.")
    else:
        homedir = os.environ.get("HOME")
        if homedir is None:
            homedir = os.path.expanduser("~")
        username = os.getlogin()
        cachedir = f"{homedir}/.solarpipe"
        ENV_FILE = f"{cachedir}/meersolar_prefect.env"
        load_dotenv(dotenv_path=ENV_FILE, override=False)

    os.system(f"rm -rf {args.workdir}/dask_*")

    ###############################################
    # Setup cluster environment
    ###############################################
    if args.cluster is not True:
        print("Setting up local cluster....")
        dask_client, dask_cluster, dask_dir = get_local_dask_cluster(
            2,
            dask_dir=args.workdir,
            cpu_frac=args.cpu_frac,
            mem_frac=args.mem_frac,
        )
        nworker = max(2, int(psutil.cpu_count() * args.cpu_frac))
        print(f"Total maximum dask workers: {nworker}")
        dask_addr = dask_client.scheduler.address
    else:
        nworker = max(2, args.nworker)
        print(f"Total maximum dask workers: {nworker}")

    ##########################################
    # Starting pipeline
    ##########################################
    try:
        print("########################################")
        print("Starting MeerSOLAR Pipeline....")
        print("#########################################")
        jobid = get_jobid()

        msg = master_control.with_options(
            flow_run_name=f"meersolar_{jobid}",
            task_runner=DaskTaskRunner(address=dask_addr),
        )(
            msname=args.msname,
            workdir=args.workdir,
            outdir=args.outdir,
            solar_data=args.solar_data,
            # Pre-calibration
            do_forcereset_weightflag=args.do_forcereset_weightflag,
            do_cal_partition=args.do_cal_partition,
            do_cal_flag=args.do_cal_flag,
            do_import_model=args.do_import_model,
            # Basic calibration
            do_basic_cal=args.do_basic_cal,
            do_noise_cal=args.do_noise_cal,
            do_applycal=args.do_applycal,
            # Target data preparation
            do_target_split=args.do_target_split,
            target_scans=args.target_scans,
            freqrange=args.freqrange,
            timerange=args.timerange,
            uvrange=args.cal_uvrange,
            # Polarization calibration
            do_polcal=args.do_polcal,
            # Self-calibration
            do_selfcal=args.do_selfcal,
            do_selfcal_split=args.do_selfcal_split,
            do_apply_selfcal=args.do_apply_selfcal,
            do_ap_selfcal=args.do_ap_selfcal,
            solar_selfcal=args.solar_selfcal,
            solint=args.solint,
            # Sidereal correction
            do_sidereal_cor=args.do_sidereal_cor,
            # Dynamic spectra
            make_ds=args.make_ds,
            # Imaging
            do_imaging=args.do_imaging,
            do_pbcor=args.do_pbcor,
            weight=args.weight,
            robust=args.robust,
            minuv=args.minuv,
            image_freqres=args.image_freqres,
            image_timeres=args.image_timeres,
            pol=args.pol,
            clean_threshold=args.clean_threshold,
            use_multiscale=args.use_multiscale,
            use_solar_mask=args.use_solar_mask,
            cutout_rsun=args.cutout_rsun,
            make_overlay=args.make_overlay,
            # Resource settings
            cpu_frac=args.cpu_frac,
            mem_frac=args.mem_frac,
            keep_backup=args.keep_backup,
            # Remote logging
            remote_logger=args.remote_logger,
            jobid=jobid,
            max_worker=nworker,
        )
    except Exception as e:
        traceback.print_exc()
    finally:
        drop_cache(args.msname)
        drop_cache(args.workdir)
        drop_cache(args.outdir)
        dask_client.close()
        dask_cluster.close()
        os.system(f"rm -rf {dask_dir}")


if __name__ == "__main__":
    cli()
