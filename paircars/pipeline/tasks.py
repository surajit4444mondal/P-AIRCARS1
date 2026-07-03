import os
import socket
from multiprocessing import Event
from prefect import task
from prefect.context import get_run_context
from prefect_dask import get_dask_client
from prefect.tasks import exponential_backoff
from paircars.utils.basic_utils import internet_available
from paircars.data.sendmail import (
    send_paircars_notification as send_notification,
)
from paircars.utils.prefect_logger_utils import start_log_task_saver
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
    make_mwa_overlay,
    move_solarcenter,
    make_ms_plot,
)


@task(
    name="move_solarcenter",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_solar_phasecenter_jobs(
    mslist,
    workdir,
    prefix="target",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Move phase center to the Sun

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
    remote_log : bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    phasecor_basename = f"cor_phasecenter_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{phasecor_basename}_{obsid}.log"
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
        # Moving phasecenter motion correction
        #######################
        with get_dask_client() as dask_client:
            msg, succeed, failed = move_solarcenter.main(
                mslist,
                workdir=workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_sidereal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Moving phasecenter to solar center is failed.")
    else:
        return msg, succeed, failed


@task(
    name="make_ds",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
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
    obsid=0,
    verbose=False,
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
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    ds_basename = "ds_target"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{ds_basename}_{obsid}.log"
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
        ##########################
        # Making dynamic spectrum
        ##########################
        with get_dask_client() as dask_client:
            msg, succeed, failed = mwa_make_ds.main(
                mslist,
                metafits,
                workdir,
                outdir,
                plot_quantity=plot_quantity,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_ds.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Dynamic spectrum making is failed.")
    else:
        return msg, succeed, failed


@task(
    name="split",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=60),
    log_prints=True,
)
def run_target_split_jobs(
    mslist,
    metafits,
    workdir,
    datacolumn="data",
    split_coarse_chans=[],
    timeres=-1,
    freqres=-1,
    prefix="target",
    time_window=-1,
    time_interval=-1,
    quack_timestamps=-1,
    force_split=False,
    only_disk=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Split measurement set

    Parameters
    ----------
    mslist: str
        Name of the measurement sets (comma separated)
    metafits : str
        Metafits file
    workdir : str
        Working directory
    datacolumn : str, optional
        Data column
    split_coarse_chans : list, optional
        Split coarse channels
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
    quack_timestamps: int, optional
        Number of timestamps to flag at the beginning and end of each scan ("quack").
    force_split : bool, optional
        Force to split
    only_disk : bool, optional
        Split only disk times
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for spliting measurement set
    int
        Expected splited ms number
    int
        Succeeded splited ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    split_basename = f"split_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{split_basename}_{obsid}.log"
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
        ##################
        # Spliting ms
        ##################
        with get_dask_client() as dask_client:
            msg, expected, succeed = do_target_split.main(
                mslist,
                metafits,
                workdir=workdir,
                datacolumn=datacolumn,
                split_coarse_chans=split_coarse_chans,
                time_window=time_window,
                time_interval=time_interval,
                freqres=freqres,
                timeres=timeres,
                quack_timestamps=quack_timestamps,
                force_split=force_split,
                only_disk=only_disk,
                prefix=prefix,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_split.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Spliting measurement set into coarse channels is failed.")
    else:
        return msg, expected, succeed


@task(
    name="flag",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_flag(
    mslist,
    metafits,
    workdir,
    outdir,
    datacolumn="DATA",
    flag_calibrators=True,
    flag_bad_spw=False,
    flag_quack=True,
    use_rflag=False,
    use_tfcrop=False,
    flagdimension="freqtime",
    flagdata_type="target",
    run_solarflagger=False,
    normalize=False,
    restore_flag=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
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
    datacolumn : str, optional
        Data column
    flag_calibrators : bool, optional
        Flag calibrator fields
    flag_bad_spw : bool, optional
        Flag bad spectral windows
    flag_quack : bool, optional
        Flag quack timestamps
    use_rflag : bool, optional
        Use rflag or not
    use_tfcrop : bool, optional
        Use tfcrop or not
    flagdimension : str, optional
        Flag dimension (freq, time freqtime)
    flagdata_type : str, optional
        Flag data type (cal, selfcal, target)
    run_solarflagger : bool, optional
        Run solar flagger or not
    normalize : bool, optional
        Use normalization in solar flagger
    restore_flag : bool, optional
        Restore flags or not
    jobid : int, optional
        Job ID
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    if flag_calibrators:
        flagdata_type = "cal"
    flag_basename = f"flagging_{flagdata_type}_{datacolumn}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{flag_basename}_{obsid}.log"
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
        ########################
        # Calibrator ms flagging
        ########################
        with get_dask_client() as dask_client:
            msg, succeed, failed = flagging.main(
                mslist,
                metafits,
                workdir=workdir,
                outdir=outdir,
                datacolumn=datacolumn,
                flag_bad_ants=True,
                flag_bad_spw=flag_bad_spw,
                use_tfcrop=use_tfcrop,
                flag_autocorr=True,
                flag_quack=flag_quack,
                flagdimension=flagdimension,
                restore_flag=restore_flag,
                run_solarflagger=run_solarflagger,
                normalize=normalize,
                flagbackup=False,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_flag.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Flagging is failed.")
    else:
        return msg, succeed, failed


@task(
    name="model",
    retries=2,
    retry_delay_seconds=60,
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
    obsid=0,
    verbose=False,
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
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    model_basename = "modeling"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{model_basename}_{obsid}.log"
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
        ###################################
        # Calibrator ms visibility import
        ###################################
        with get_dask_client() as dask_client:
            msg, succeed, failed = import_model.main(
                mslist,
                metafits,
                workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_model.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Importing calibrator model is failed.")
    else:
        return msg, succeed, failed


@task(
    name="calibration",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_basic_cal_jobs(
    mslist,
    metafits,
    workdir,
    outdir,
    perform_polcal=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    keep_backup=False,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Perform basic calibration

    Parameters
    ----------
    mslist: str
        Name of the measurement sets (comma separated)
    metafits: str
        Metafits file
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
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for basic calibration
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    cal_basename = "basic_cal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{cal_basename}_{obsid}.log"
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
        ########################
        # Basic calibration
        ########################
        with get_dask_client() as dask_client:
            msg, succeed, failed = basic_cal.main(
                mslist,
                metafits,
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
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_cal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Basic calibration is failed.")
    else:
        return msg, succeed, failed


@task(
    name="apply_basic_cal",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_apply_basiccal_sol(
    mslist,
    target_metafits,
    workdir,
    caldir,
    overwrite_datacolumn=True,
    only_amplitude=False,
    applymode="calflag",
    prefix="target",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Apply basic calibration solutions on splited target scans

    Parameters
    ----------
    mslist: str
        Target measurement set list (comma separated)
    target_metafits : str
        Target metafits file
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    overwrite_datacolumn : bool
        Overwrite data column or not
    only_amplitude : bool, optional
        Apply only amplitude part of the solution
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
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    applycal_basename = f"apply_basiccal_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applycal_basename}_{obsid}.log"
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
        # Applying basic calibration
        ######################
        with get_dask_client() as dask_client:
            msg, succeed, failed = do_apply_basiccal.main(
                mslist,
                target_metafits,
                workdir,
                caldir,
                applymode=applymode,
                overwrite_datacolumn=overwrite_datacolumn,
                only_amplitude=only_amplitude,
                start_remote_log=remote_log,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_apply.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Applying basic calibration solutions is failed.")
    else:
        return msg, succeed, failed


@task(
    name="sidereal_cor",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_solar_siderealcor_jobs(
    mslist,
    workdir,
    prefix="target",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
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
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    sidereal_basename = f"cor_sidereal_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{sidereal_basename}_{obsid}.log"
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
        # Sidereal motion correction
        #######################
        with get_dask_client() as dask_client:
            msg, succeed, failed = do_sidereal_cor.main(
                mslist,
                workdir=workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_sidereal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Solar sidereal motion correction is failed.")
    else:
        return msg, succeed, failed


@task(
    name="selfcal",
    log_prints=True,
)
def run_selfcal_jobs(
    mslist,
    workdir,
    caldir,
    metafits,
    cal_applied,
    start_thresh=5.0,
    stop_thresh=3.0,
    max_iter=30,
    max_DR=100000,
    intselfcal_min_iter=3,
    polselfcal_min_iter=5,
    conv_frac=0.3,
    solint="60s",
    do_apcal=True,
    do_polcal=True,
    solar_selfcal=True,
    keep_backup=False,
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    applymode="calonly",
    min_tol_factor=1.0,
    use_solarflagger=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
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
    metafits : str
        Metafits file
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
    intselfcal_min_iter : int, optional
        Minimum numbers of intensity seflcal iterations at different stages
    polselfcal_min_iter : int, optional
        Minimum numbers of polarisation selfcal iterations
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
    do_polcal : bool, optional
        Perform polarisation selfcal or not
    min_tol_factor : float, optional
        Minimum tolerance in temporal variation in imaging
    applymode : str, optional
        Solution apply mode
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    use_solarflagger : bool, optional
        Use solar flagger or not
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation IDs
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for self-calibration
    int
        Intensity self-calibration succeeded ms number
    int
        Intensity self-calibration failed ms number
    int
        Polarisation self-calibration succeed ms number
    int
        Polarisation self-calibration failed ms number
    float
        Mean intensity self-calibration dynamic range
    float
        Mean polarisation self-calibration dynamic range
    float
        Maximum intensity self-calibration dynamic range
    float
        Maximum polarisation self-calibration dynamic range
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    selfcal_basename = "selfcal_target"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{selfcal_basename}_{obsid}.log"
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
        # Selfcal jobs
        ########################
        with get_dask_client() as dask_client:
            (
                msg,
                int_succeed,
                int_failed,
                pol_succeed,
                pol_failed,
                int_DR,
                pol_DR,
                max_int_DR,
                max_pol_DR,
            ) = do_selfcal.main(
                mslist,
                metafits,
                workdir,
                caldir,
                cal_applied=cal_applied,
                start_thresh=float(start_thresh),
                stop_thresh=float(stop_thresh),
                max_iter=float(max_iter),
                max_DR=float(max_DR),
                intselfcal_min_iter=int(intselfcal_min_iter),
                polselfcal_min_iter=int(polselfcal_min_iter),
                conv_frac=float(conv_frac),
                solint=solint,
                uvrange=uvrange,
                minuv=float(minuv),
                weight=weight,
                robust=float(robust),
                applymode=applymode,
                min_tol_factor=float(min_tol_factor),
                do_apcal=do_apcal,
                do_polcal=do_polcal,
                solar_selfcal=solar_selfcal,
                use_solarflagger=use_solarflagger,
                keep_backup=keep_backup,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_selfcal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Self-calibration is failed.")
    else:
        return (
            msg,
            int_succeed,
            int_failed,
            pol_succeed,
            pol_failed,
            int_DR,
            pol_DR,
            max_int_DR,
            max_pol_DR,
        )


@task(
    name="apply_selfcal",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_apply_selfcal_sol(
    mslist,
    metafits,
    workdir,
    caldir,
    overwrite_datacolumn=True,
    applymode="calflag",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Apply self-calibration solutions on splited target scans

    Parameters
    ----------
    mslist: str
        Target measurement set list (comma separated)
    metafits : str
        Metafits file
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
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    int
        Succeeded gain solution ms number
    int
        Failed gain solution ms number
    int
        Succeeded polarisation solution ms number
    int
        Failed polarisation solution ms number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    applycal_basename = "apply_selfcal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applycal_basename}_{obsid}.log"
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
        ########################
        # Applying self-calibration
        ########################
        with get_dask_client() as dask_client:
            gain_succeed, gain_failed, pol_succeed, pol_failed = do_apply_selfcal.main(
                mslist,
                metafits,
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
                verbose=verbose,
            )
        if gain_failed == 0:
            msg = 0
        else:
            msg = 1
    finally:
        stop_event.set()
        log_thread_applyselfcal.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Applying self-calibration solutions is failed.")
    else:
        return msg, gain_succeed, gain_failed, pol_succeed, pol_failed


@task(
    name="imaging",
    log_prints=True,
)
def run_imaging_jobs(
    mslist,
    workdir,
    outdir,
    freqrange="",
    timerange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    pol="IQUV",
    freqres=1.28,
    timeres=10.0,
    threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    cutout_rsun=10.0,
    savemodel=False,
    saveres=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
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
    threshold : float, optional
        CLEAN threshold in sigma
    use_multiscale : bool, optional
        Use multiscale or not
    use_solar_mask : bool, optional
        Use solar mask or not
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 10.0 solar radii)
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for imaging
    int
        Succeeded ms number
    int
        Failed ms number
    int
        Total images
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    imaging_basename = "imaging_target"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{imaging_basename}_{obsid}.log"
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
        #######################
        # Performing imaging
        #######################
        with get_dask_client() as dask_client:
            msg, succeed, failed, total_images = do_imaging.main(
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
                cutout_rsun=float(cutout_rsun),
                use_multiscale=use_multiscale,
                use_solar_mask=use_solar_mask,
                savemodel=savemodel,
                saveres=saveres,
                start_remote_log=remote_log,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                jobid=jobid,
                logfile=logfile,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_imaging.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Imaging is failed.")
    else:
        return msg, succeed, failed, total_images


@task(
    name="apply_pbcor",
    log_prints=True,
)
def run_apply_pbcor(
    imagedir,
    metafits,
    workdir,
    leakage_dir="",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
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
    leakage_dir : str, optional
        Leakage dile directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for applying primary beam correction on all images
    int
        Succeeded image number
    int
        Failed image number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    applypbcor_basename = "apply_pbcor"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applypbcor_basename}_{obsid}.log"
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
        #####################
        # Applying primary beam correction
        #####################
        with get_dask_client() as dask_client:
            msg, succeed, failed = mwa_pbcor.main(
                imagedir,
                metafits,
                leakage_dir=leakage_dir,
                workdir=workdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_pbcor.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Primary beam correction is failed.")
    else:
        return msg, succeed, failed


@task(
    name="make_overlay",
    log_prints=True,
)
def run_make_overlay(
    imagedir,
    outdir,
    workdir="",
    all_overlay=False,
    jobid=0,
    cpu_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Making overlays of all images on EUV images

    Parameters
    ----------
    imagedir : str
        Image directory name
    outdir : str
        Output directory
    workdir : str, optional
        Work directory
    all_overlay : bool, optional
        Whether to make overlays for all images or not
    cpu_frac : float, optional
        CPU fraction to use
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for EUV overlays
    int
        Succeeded image number
    int
        Failed image number
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    overlay_basename = "do_overlay"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{overlay_basename}_{obsid}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_overlay = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    os.makedirs(outdir, exist_ok=True)
    try:
        #####################
        # Making overlays
        #####################
        with get_dask_client() as dask_client:
            msg, succeed, failed = make_mwa_overlay.main(
                imagedir,
                outdir,
                workdir=workdir,
                all_overlay=all_overlay,
                cpu_frac=float(cpu_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_overlay.join(timeout=5)
    if msg != 0:
        return 1, succeed, failed
    else:
        return msg, succeed, failed


@task(
    name="make_msplot",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_make_msplot(
    mslist,
    workdir,
    outdir,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    obsid=0,
    verbose=False,
):
    """
    Making diagnostic plots of measurement sets

    Parameters
    ----------
    mslist : str
        Measurement set list (comma separated)
    workdir : str, optional
        Work directory
    outdir : str
        Output directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    remote_log: bool, optional
        Start remote logger
    obsid : int, optional
        Observation ID
    verbose : bool, optional
        Verbose logs

    Returns
    -------
    int
        Success message for measurement set ploting
    """
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    msplot_basename = "do_msplot"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{msplot_basename}_{obsid}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id = str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread_overlay = start_log_task_saver(
        task_id, task_name, logfile, poll_interval=3, stop_event=stop_event
    )
    os.makedirs(outdir, exist_ok=True)
    try:
        #####################
        # Making plots
        #####################
        with get_dask_client() as dask_client:
            msg = make_ms_plot.main(
                mslist,
                workdir,
                outdir,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                logfile=logfile,
                jobid=jobid,
                start_remote_log=remote_log,
                dask_client=dask_client,
                verbose=verbose,
            )
    finally:
        stop_event.set()
        log_thread_overlay.join(timeout=5)
    if msg != 0:
        raise RuntimeError("Measurement set diagnostic ploting is failed.")
    else:
        return msg


def send_task_notification(
    emails, msg, jobid, obsid, logger_timestamp, flow_name="", attachments=[]
):
    """
    Send notification after each task is finished

    Parameters
    ----------
    emails : str
        E-mail ids
    msg : str
        Notification message
    jobid : int
        JobID
    obsid : int
        Observation ID
    logger_timestamp : str
        Logger timestamp
    flow_name : str, optional
        Flow name
    attachments : list, optional
        Any attachment to send
    """
    internet_on = internet_available()
    hostname = socket.gethostname()
    if internet_on:
        try:
            email_subject = (
                f"P-AIRCARS Log: {logger_timestamp}, OBSID: {obsid}, Hostname: {hostname}"
            )
            if flow_name == "":
                email_msg = f"{msg}"
            else:
                email_msg = f"From {flow_name}\n{msg}"
            success_msg, error_msg = send_notification(
                emails, email_subject, email_msg, attachments=attachments
            )
        except Exception:
            print("Could not send log emails.")
