import traceback
import glob
import os
import time
import numpy as np
from prefect import flow
from astropy.io import fits
from casatools import msmetadata
from prefect.futures import wait
from paircars.utils.basic_utils import (
    print_banner,
    internet_available,
)
from paircars.utils.calibration import (
    interpolate_bpass,
    interpolate_quartical,
    get_caltable_metadata,
    scale_bandpass,
)
from paircars.utils.flagging import get_chans_flag
from paircars.utils.mwa_ploting_utils import (
    plot_caltable_diagnostics,
    plot_quartical_tables,
    plot_hpc_collage,
)
from paircars.utils.mwa_utils import (
    freq_to_MWA_coarse,
    get_selfcal_ntimes,
)
from paircars.utils.ms_metadata import check_datacolumn_valid
from paircars.utils.image_utils import filter_images
from paircars.pipeline.tasks import (
    run_solar_phasecenter_jobs,
    run_ds_jobs,
    run_target_split_jobs,
    run_flag,
    run_import_model,
    run_basic_cal_jobs,
    run_apply_basiccal_sol,
    run_solar_siderealcor_jobs,
    run_selfcal_jobs,
    run_apply_selfcal_sol,
    run_imaging_jobs,
    run_apply_pbcor,
    run_make_overlay,
    send_task_notification,
)
from prefect.context import get_run_context
from multiprocessing import Event
from paircars.utils.prefect_logger_utils import start_flow_log_saver
from paircars.utils.logger_utils import (
    clean_shutdown,
    init_logger,
)


#########################
# Pre-processing subflow
#########################
@flow(
    name="Pre-processing target",
    description="Perform pre-processing on target measurement sets",
    log_prints=True,
)
def pre_process_subflow(
    # Core observational inputs
    target_mslist,
    target_metafits,
    target_obsid,
    solar_data,
    # I/O and workspace
    workdir,
    target_outdir,
    # Processing controls
    do_move_solarcenter,
    make_ds,
    # Resource management
    cpu_frac,
    mem_frac,
    # Logging / metadata
    jobid,
    timestamp,
    emails,
    remote_logger,
    verbose,
):
    """
    Pre-processing of target measurement set subflow

    Returns
    -------
    int
        Flow success message
    list
        Filtered target measurement set list
    """
    start_time = time.time()
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    pre_process_logfile = f"{logdir}/subflow_preprocess_{target_obsid}.log"
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id, flow_name, pre_process_logfile, poll_interval=3, stop_event=stop_event
    )
    observer = None
    if os.path.exists(f"{workdir}/.jobname_password.npy"):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if pre_process_logfile is not None and os.path.exists(pre_process_logfile):
            observer = init_logger(
                "preprocess_subflow_log",
                pre_process_logfile,
                log_type="subflow",
                jobname=jobname,
                password=password,
            )
    print_banner("Starting pre-processing subflow.")
    if observer is None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    try:
        ########################################
        # Flagging coarse channel edges
        ########################################
        if emails != "":
            email_msg = f"[{target_obsid}] Started flagging of targets."
            send_task_notification(
                emails,
                email_msg,
                jobid,
                target_obsid,
                timestamp,
                flow_name=f"subflow {flow_name}",
            )
        print_banner("Starting task: Flagging targets.")
        target_freqres_metafits = float(fits.getheader(target_metafits)["FINECHAN"])
        msmd = msmetadata()
        msmd.open(target_mslist[0])
        target_freqres_ms = float(round(msmd.chanres(0,unit="kHz")[0],0))
        print(f"Metafits frequency resolution: {target_freqres_metafits}kHz.")
        print(f"Measurement set frequency resolution: {target_freqres_ms}kHz.")
        if target_freqres_ms!=target_freqres_metafits:
            print("Measurement set is already frequency averaged. Not flagging coarse channel edges.")
            flag_bad_spw=False
        else:
            flag_bad_spw=True
        try:
            future_flag = run_flag.with_options(
                task_run_name=f"flag_target_initial_{target_obsid}"
            ).submit(
                ",".join(target_mslist),
                target_metafits,
                workdir,
                target_outdir,
                datacolumn="data",
                flag_calibrators=False,
                flag_bad_spw=flag_bad_spw,
                flag_quack=False,
                use_rflag=False,
                use_tfcrop=False,
                flagdimension="freqtime",
                flagdata_type="target",
                run_solarflagger=False,
                normalize=False,
                restore_flag=False,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                obsid=target_obsid,
                verbose=verbose,
            )
            wait([future_flag])
            msg, succeed, failed = future_flag.result()
            if emails != "":
                email_msg = f"[{target_obsid}] Initial flagging of targets is done.\nSucceeded: {succeed}, failed: {failed}."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            filtered_ms = []
            for c_ms in target_mslist:
                c_ms = c_ms.rstrip("/")
                if os.path.exists(f"{c_ms}/.flag_succeed"):
                    filtered_ms.append(c_ms)
                else:
                    print(f"Issue in flagging of measurement set: {c_ms}")
            target_mslist = filtered_ms  # Filtered target mslist
            print_banner("Finished task: Initial flagging of target is done.")
        except Exception:
            print_banner("!!!! WARNING: Initial flagging error for targets. !!!!")
            traceback.print_exc()
            if emails != "":
                email_msg = f"[{target_obsid}] Error in initial flagging of targets."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
    
        ########################################
        # Moving phasecenter to the solar center
        ########################################
        if solar_data and do_move_solarcenter:
            if emails != "":
                email_msg = (
                    f"[{target_obsid}] Started moving phasecenter to solar center."
                )
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Moving phasecenter to the Sun.")
            try:
                future_movecenter = run_solar_phasecenter_jobs.with_options(
                    task_run_name=f"move_solarcenter_{target_obsid}",
                ).submit(
                    ",".join(target_mslist),
                    workdir,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                wait([future_movecenter])
                msg, succeed, failed = future_movecenter.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Moving phasecenter to solar center is done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Moving phasecenter to solar center is done."
                )
                filtered_ms = []
                for t_ms in target_mslist:
                    t_ms = t_ms.rstrip("/")
                    if os.path.exists(f"{t_ms}/.solarcenter_move_succeed"):
                        filtered_ms.append(t_ms)
                    else:
                        print(f"Issue in moving phasecneter to solar center: {t_ms}")
                target_mslist = filtered_ms  # Filtered target mslist
            except Exception:
                print_banner(
                    "Error in moving phasecenter to solar center. P-AIRCARS has stopped."
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in moving phasecenter to solar center."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Pre-processing subflow failed.")
                return 1, []

        #######################################
        # Run dynamic spectra making
        #######################################
        if solar_data and make_ds:
            if emails != "":
                email_msg = f"[{target_obsid}] Started making solar dynamic spectra."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Making dynamic spectra of solar target.")
            try:
                future_maskms = run_ds_jobs.with_options(
                    task_run_name=f"make_ds_{target_obsid}",
                ).submit(
                    ",".join(target_mslist),
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
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Finished task: Making solar dynamic spectra are done.")
            except Exception:
                print_banner("!!! WARNING : Error in making dynamic spectra. !!!")
                traceback.print_exc()
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
                        flow_name=f"subflow {flow_name}",
                    )
        print_banner("Pre-processing subflow is successful.")
        return 0, target_mslist
    except Exception:
        print_banner("Pre-processing subflow failed.")
        traceback.print_exc()
        return 1, []
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Total run time: {run_time}")
        stop_event.set()
        time.sleep(60)
        log_thread_flow.join()
        if observer is not None:
            clean_shutdown(observer)


############################
# Basic calibration subflow
############################
@flow(
    name="Basic calibration",
    description="Perform basic calibration using calibrator observations",
    log_prints=True,
)
def basic_cal_subflow(
    # Core observational inputs
    cal_obsid,
    cal_datadir,
    cal_metafits,
    coarse_chans,
    target_obsid,
    target_metafits,
    # I/O and workspace
    workdir,
    cal_outdir,
    basic_caldir,
    # Calibration controls
    do_basic_cal,
    redo_basic_cal,
    do_cal_flag,
    do_import_model,
    do_polcal,
    keep_backup,
    # Data conditioning
    quack_timestamps,
    # Resource management
    cpu_frac,
    mem_frac,
    # Logging / metadata
    jobid,
    timestamp,
    emails,
    remote_logger,
    verbose,
):
    """
    Basic calibration sub flow
    """
    start_time = time.time()
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    basic_cal_logfile = f"{logdir}/subflow_basiccal_{cal_obsid}.log"
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id, flow_name, basic_cal_logfile, poll_interval=3, stop_event=stop_event
    )
    observer = None
    if os.path.exists(f"{workdir}/.jobname_password.npy"):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if basic_cal_logfile is not None and os.path.exists(basic_cal_logfile):
            observer = init_logger(
                "basic_cal_subflow_log",
                basic_cal_logfile,
                log_type="subflow",
                jobname=jobname,
                password=password,
            )
    print_banner("Starting basic-calibration subflow.")
    if observer is None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    try:
        ##########################################
        # Checking presence of basic caltables
        ##########################################
        if not redo_basic_cal:
            print(
                f"Searching for existing bandpass tables: {basic_caldir}/calibrator_{cal_obsid}*.bcal"
            )
            bandpass_tables = sorted(
                glob.glob(f"{basic_caldir}/calibrator_{cal_obsid}*.bcal")
            )
            print(
                f"Searching for existing crossphase tables: {basic_caldir}/calibrator_{cal_obsid}*.kcrossscal"
            )
            crossphase_tables = sorted(
                glob.glob(f"{basic_caldir}/calibrator_{cal_obsid}*.kcrosscal")
            )
            if len(bandpass_tables) < len(coarse_chans):
                bpass_coarse_chans = []
                for bpass in bandpass_tables:
                    cal_metadata = get_caltable_metadata(bpass)
                    freqMHz = cal_metadata["Channel 0 frequency (MHz)"]
                    bpass_coarse_chans.append(freq_to_MWA_coarse(freqMHz))
                coarse_chans = [
                    x for x in coarse_chans if x not in set(bpass_coarse_chans)
                ]
            else:
                print_banner(
                    f"Bandpass tables are already present. Calibration directory: {basic_caldir}"
                )
                for bpass in bandpass_tables:
                    print(f"{os.path.basename(bpass)}")
                if len(crossphase_tables) < len(coarse_chans):
                    kcross_coarse_chans = []
                    for kcross in crossphase_tables:
                        cal_metadata = get_caltable_metadata(kcross)
                        freqMHz = cal_metadata["Channel 0 frequency (MHz)"]
                        kcross_coarse_chans.append(freq_to_MWA_coarse(freqMHz))
                    coarse_chans = [
                        x for x in coarse_chans if x not in set(kcross_coarse_chans)
                    ]
                else:
                    print_banner(
                        f"Crosshand phase tables are already present. Calibration directory: {basic_caldir}"
                    )
                    for kcross in crossphase_tables:
                        print(f"{os.path.basename(kcross)}")
                    if emails != "":
                        email_msg = f"[{cal_obsid}] All gain solutions from calibrator are already present."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    print("All gain solutions are already present.")
                    print_banner("Basic calibration subflow is successful.")
                    return 0, bandpass_tables, crossphase_tables

        ############################
        # Calibrator ms list
        ############################
        cal_mslist = glob.glob(f"{cal_datadir}/*.ms")
        if len(cal_mslist) == 0 or len(coarse_chans) == 0:
            print(
                f"No calibrator measurement set present. Coarse channels: {coarse_chans}. Calibrator directory: {cal_datadir}"
            )
            print_banner("Basic calibration subflow failed.")
            if emails != "":
                email_msg = f"[{cal_obsid}] No calibrator measurement set with coarse channels: {coarse_chans} is present in: {cal_datadir}."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            return 1, [], []

        ##############################
        # Run spliting jobs
        ##############################
        # If basic calibration is requested and calibrator ms and metafits are present
        if do_cal_flag or do_import_model or do_basic_cal:
            prefix = "calibrator"
            if emails != "":
                email_msg = (
                    f"[{cal_obsid}] Started spliting of calibrator measurement sets."
                )
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Spliting of calibrator measurement sets.")
            try:
                future_cal_split = run_target_split_jobs.with_options(
                    task_run_name=f"split_{cal_obsid}"
                ).submit(
                    ",".join(cal_mslist),
                    cal_metafits,
                    workdir,
                    datacolumn="data",
                    split_coarse_chans=coarse_chans,
                    timeres=10.0,
                    freqres=0.16,
                    prefix=prefix,
                    force_split=False,
                    time_window=-1,
                    time_interval=-1,
                    quack_timestamps=quack_timestamps,
                    jobid=jobid,
                    cpu_frac=float(cpu_frac),
                    mem_frac=float(mem_frac),
                    remote_log=remote_logger,
                    obsid=cal_obsid,
                    verbose=verbose,
                )
                wait([future_cal_split])
                msg, expected, succeed = future_cal_split.result()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Spliting of calibrator measurement sets are done.\nExpected: {expected}, succeeded: {succeed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Spliting of calibrator measurement sets are done."
                )
            except Exception:
                print(
                    "!!!! WARNING: Error in spliting calibrator measurement sets. !!!!"
                )
                print_banner("Basic calibration subflow failed.")
                traceback.print_exc()
                if emails != "":
                    email_msg = (
                        f"[{cal_obsid}] Spliting calibrator measurement set failed."
                    )
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], []

        if do_cal_flag or do_import_model or do_basic_cal:
            split_cal_mslist = sorted(
                glob.glob(f"{workdir}/calibrator_{cal_obsid}*_ch_*.ms")
            )
            if len(split_cal_mslist) == 0:
                print("No splited measurement set is present for basic calibration.")
                print_banner("Basic calibration subflow failed.")
                if emails != "":
                    email_msg = f"[{cal_obsid}] No splited measurement set is present for basic calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], []

        ##################################
        # Run flagging jobs on calibrators
        ##################################
        # Only if basic calibration is requested
        if do_cal_flag:
            if emails != "":
                email_msg = f"[{cal_obsid}] Started flagging of calibrators."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Flagging calibrators.")
            cal_freqres_metafits = float(fits.getheader(cal_metafits)["FINECHAN"])
            msmd = msmetadata()
            msmd.open(split_cal_mslist[0])
            cal_freqres_ms = float(round(msmd.chanres(0,unit="kHz")[0],0))
            print(f"Metafits frequency resolution: {cal_freqres_metafits}kHz.")
            print(f"Measurement set frequency resolution: {cal_freqres_ms}kHz.")
            if cal_freqres_ms!=cal_freqres_metafits:
                print("Measurement set is already frequency averaged. Not flagging coarse channel edges.")
                flag_bad_spw=False
            else:
                flag_bad_spw=True
            try:
                future_flag = run_flag.with_options(
                    task_run_name=f"flag_cal_data_{cal_obsid}"
                ).submit(
                    ",".join(split_cal_mslist),
                    cal_metafits,
                    workdir,
                    cal_outdir,
                    datacolumn="data",
                    flag_calibrators=True,
                    flag_bad_spw=flag_bad_spw,
                    flag_quack=False,
                    use_rflag=False,
                    use_tfcrop=True,
                    flagdimension="freqtime",
                    flagdata_type="cal",
                    run_solarflagger=False,
                    normalize=False,
                    restore_flag=False,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=cal_obsid,
                    verbose=verbose,
                )
                wait([future_flag])
                msg, succeed, failed = future_flag.result()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Flagging of calibrator is done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                filtered_ms = []
                for c_ms in split_cal_mslist:
                    c_ms = c_ms.rstrip("/")
                    if os.path.exists(f"{c_ms}/.flag_succeed"):
                        filtered_ms.append(c_ms)
                    else:
                        print(f"Issue in flagging of measurement set: {c_ms}")
                split_cal_mslist = filtered_ms  # Filtered target mslist
                print_banner("Finished task: Flagging of calibrator is done.")
            except Exception:
                print_banner("!!!! WARNING: Flagging error for calibrator. !!!!")
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Error in flagging calibrators."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )

        #################################
        # Import model
        #################################
        if do_import_model:
            if emails != "":
                email_msg = f"[{cal_obsid}] Started importing sky model."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Importing model visibilities.")
            try:
                future_import_model = run_import_model.with_options(
                    task_run_name=f"model_{cal_obsid}"
                ).submit(
                    ",".join(split_cal_mslist),
                    cal_metafits,
                    workdir,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=cal_obsid,
                    verbose=verbose,
                )
                wait([future_import_model])
                msg, succeed, failed = future_import_model.result()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Model import for calibrator is done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Finished task: Model import for calibrator is done.")
                filtered_ms = []
                for c_ms in split_cal_mslist:
                    c_ms = c_ms.rstrip("/")
                    if os.path.exists(f"{c_ms}/.modeling_succeed"):
                        filtered_ms.append(c_ms)
                    else:
                        print(f"Issue in importing calibrator sky model: {c_ms}")
                split_cal_mslist = filtered_ms  # Filtered target mslist
            except Exception:
                print(
                    "!!!! WARNING: Error in importing calibrator models. Not continuing calibration. !!!!"
                )
                print_banner("Basic calibration subflow failed.")
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Error occured in importing model for calibrators.\nNot using calibrator solutions."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], []

        ###############################
        # Run basic calibration
        ###############################
        if do_basic_cal:
            if emails != "":
                email_msg = f"[{cal_obsid}] Started basic calibration."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Performing basic calibration.")
            try:
                future_basical = run_basic_cal_jobs.with_options(
                    task_run_name=f"calibration_{cal_obsid}"
                ).submit(
                    ",".join(split_cal_mslist),
                    cal_metafits,
                    workdir,
                    cal_outdir,
                    perform_polcal=do_polcal,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    keep_backup=keep_backup,
                    remote_log=remote_logger,
                    obsid=cal_obsid,
                    verbose=verbose,
                )
                wait([future_basical])
                msg, succeed, failed = future_basical.result()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Basic calibration is done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Finished task: Basic calibration is done.")
            except Exception:
                print("!!!! WARNING: Error in basic calibration. !!!!")
                print_banner("Basic calibration subflow failed.")
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{cal_obsid}] Error occured in basic calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], []

        ##################################################################
        # Checking and interpolating bandpass tables
        ##################################################################
        print(
            f"Searching for bandpass tables: {basic_caldir}/calibrator_{cal_obsid}*.bcal"
        )
        bandpass_tables = sorted(
            glob.glob(f"{basic_caldir}/calibrator_{cal_obsid}*.bcal")
        )
        if len(bandpass_tables) == 0:
            print(
                f"No bandpass table is present. Calibration directory : {basic_caldir}."
            )
            print_banner("Basic calibration subflow failed.")
            if emails != "":
                email_msg = f"[{cal_obsid}] No bandpass calibration table is found."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            return 1, [], []
        bandpass_tables = interpolate_bpass(bandpass_tables, overwrite=True)

        ################################
        # Scale bandpass for attenuators
        ################################
        calibrator_header = fits.getheader(cal_metafits)
        cal_attn = calibrator_header["ATTEN_DB"]
        target_header = fits.getheader(target_metafits)
        target_attn = target_header["ATTEN_DB"]
        for bpass_table in bandpass_tables:
            print(f"Scaling for attenuation: {bpass_table}")
            scale_bandpass(bpass_table, cal_attn, target_attn)

        print_banner(f"Bandpass tables in calibration directory: {basic_caldir}")
        for bpass in bandpass_tables:
            print(f"{os.path.basename(bpass)}")

        ######################################
        # Checking crossphase tables
        ######################################
        print(
            f"Searching for crossphase tables: {basic_caldir}/calibrator_{cal_obsid}*.kcrossscal"
        )
        crossphase_tables = sorted(
            glob.glob(f"{basic_caldir}/calibrator_{cal_obsid}*.kcrosscal")
        )
        if len(crossphase_tables) > 0:
            crossphase_tables = interpolate_bpass(crossphase_tables, overwrite=True)
            print_banner(
                f"Crosshand phase tables in calibration directory: {basic_caldir}"
            )
            for kcross in crossphase_tables:
                print(f"{os.path.basename(kcross)}")

        ###############################################
        # Making diagnostic plots
        ###############################################
        if len(bandpass_tables) > 0 and do_basic_cal:
            os.makedirs(f"{cal_outdir}/diagnostic_plots", exist_ok=True)
            msg, bpass_plots = plot_caltable_diagnostics(
                bandpass_tables,
                f"{cal_outdir}/diagnostic_plots/{cal_obsid}_bcal",
            )
            if msg == 0:
                print_banner(
                    f"Diagnostic plots for bandpass tables are saved in: {bpass_plots}."
                )
            else:
                print("Error in creating diagnostic plots for bandpass tables.")
        if len(crossphase_tables) > 0 and do_basic_cal:
            os.makedirs(f"{cal_outdir}/diagnostic_plots", exist_ok=True)
            msg, kcross_plots = plot_caltable_diagnostics(
                crossphase_tables,
                f"{cal_outdir}/diagnostic_plots/{cal_obsid}_kcrosscal",
                quantities=["phase"],
                plot_all_ants=False,
            )
            if msg == 0:
                print_banner(
                    f"Diagnostic plots for crosshand phase tables are saved in: {kcross_plots}."
                )
            else:
                print("Error in creating diagnostic plots for crosshand phase tables.")
        print_banner("Basic calibration subflow is successful.")
        return 0, bandpass_tables, crossphase_tables
    except Exception:
        print_banner("Basic calibration subflow failed.")
        traceback.print_exc()
        return 1, [], []
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Total run time: {run_time}")
        stop_event.set()
        time.sleep(60)
        log_thread_flow.join()
        if observer is not None:
            clean_shutdown(observer)


########################################################
# Self-calibration subflows
########################################################
@flow(
    name="Self-calibration",
    description="Perform self-calibration on target measurement sets",
    log_prints=True,
)
def selfcal_subflow(
    # Core observational inputs
    target_mslist,
    target_metafits,
    target_obsid,
    # I/O and workspace
    workdir,
    basic_caldir,
    selfcaldir,
    target_outdir,
    # Processing controls
    redo_selfcal,
    do_selfcal,
    has_cal,
    solar_selfcal,
    do_sidereal_cor,
    use_solarflagger,
    keep_backup,
    # Selfcal parameters
    solint,
    timeavg,
    freqavg,
    image_timeres,
    image_freqres,
    quack_timestamps,
    only_amplitude,
    do_ap_selfcal,
    do_polcal,
    uvrange,
    # Resource management
    cpu_frac,
    mem_frac,
    # Logging / metadata
    jobid,
    timestamp,
    emails,
    remote_logger,
    verbose,
):
    """
    Self-calibration subflow

    Returns
    -------
    int
        Flow success message
    list
        Self-calibration gaincal tables
    list
        Self-calibration bandpass tables
    list
        Self-calibration polcal leakage tables
    """
    start_time = time.time()
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    selfcal_subflow_logfile = f"{logdir}/subflow_selfcal_{target_obsid}.log"
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id,
        flow_name,
        selfcal_subflow_logfile,
        poll_interval=3,
        stop_event=stop_event,
    )
    observer = None
    if os.path.exists(f"{workdir}/.jobname_password.npy"):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if selfcal_subflow_logfile is not None and os.path.exists(
            selfcal_subflow_logfile
        ):
            observer = init_logger(
                "selfcal_subflow_log",
                selfcal_subflow_logfile,
                log_type="subflow",
                jobname=jobname,
                password=password,
            )
    print_banner("Starting self-calibration subflow.")
    if observer is None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    try:
        ###################################################
        # Checking if selfcal tables already exist or not
        ###################################################
        if not redo_selfcal:
            print("Checking pre-existing self-calibration solutions.")
            selfcal_gaincal = sorted(
                glob.glob(f"{selfcaldir}/selfcal_{target_obsid}*.gcal")
            )
            if len(selfcal_gaincal) > 0:
                selfcal_bandpass = sorted(
                    glob.glob(f"{selfcaldir}/selfcal_{target_obsid}*.bcal")
                )
                if len(selfcal_bandpass) > 0:
                    selfcal_bandpass = interpolate_bpass(
                        selfcal_bandpass, overwrite=True
                    )
                if do_polcal:
                    selfcal_leakage = sorted(
                        glob.glob(f"{selfcaldir}/selfcal_{target_obsid}*.dcal")
                    )
                    if len(selfcal_leakage) > 0:
                        selfcal_leakage = interpolate_quartical(
                            selfcal_leakage, overwrite=True
                        )
                        print(
                            "Self-calibration solutions exist including polarisation calibration. Not performing self-calibration"
                        )
                        print_banner("Self-calibration subflow is successful.")
                        if emails != "":
                            email_msg = f"[{target_obsid}] Self-calibration solutions including polarisation for target are already present."
                            send_task_notification(
                                emails,
                                email_msg,
                                jobid,
                                target_obsid,
                                timestamp,
                                flow_name=f"subflow {flow_name}",
                            )
                        return 0, selfcal_gaincal, selfcal_bandpass, selfcal_leakage
                    else:
                        print_banner(
                            "Self-calibration solutions exist without polarisation calibration. Hence, performing self-calibration"
                        )
                else:
                    print(
                        "Self-calibration solutions exist without polarisation calibration. Polarisation calibration is not requested."
                    )
                    print_banner("self-calibration subflow is successful.")
                    return 0, selfcal_gaincal, selfcal_bandpass, []

        ###################################################
        # Start spliting selfcal ms
        ###################################################
        if not do_selfcal:
            print(
                "Self-calibration is not requested and previous self-calibration tables are also not present."
            )
            print_banner("Self-calibrartion subflow failed.")
            if emails != "":
                email_msg = f"[{target_obsid}] Self-calibration is not requested and previous self-calibration tables are also not present."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            return 1, [], [], []
        else:
            ###############################################
            # Removing previous self-calibration artificats
            ###############################################
            print("Removing all previous self-calibration artificats.")
            os.system(
                f"rm -rf {workdir}/selfcal* {workdir}/.intselfcal* {workdir}/.polselfcal*"
            )
            prefix = "selfcal"
            try:
                time_interval = float(solint)
            except BaseException:
                if solint.endswith("s"):
                    time_interval = float(solint.split("s")[0])
                elif solint.endswith("min"):
                    time_interval = float(solint.split("min")[0]) * 60
                elif solint == "int":
                    time_interval = image_timeres
                else:
                    time_interval = 60.0

            ######################
            # Spliting
            ######################
            if emails != "":
                email_msg = f"[{target_obsid}] Started spliting of measurement sets for self-calibration."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner(f"Starting task: Spliting {prefix}.")
            ntime = get_selfcal_ntimes(target_mslist[0])
            msmd = msmetadata()
            msmd.open(target_mslist[0])
            times = msmd.timesforspws(0)
            timeres = np.nanmean(np.diff(times))
            msmd.close()
            time_window = min(10, round(ntime * timeres, 1))  # Maximum 10s
            print(f"Time window: {min(time_window, time_interval)}")
            print(f"Time interval: {time_interval}")
            try:
                future_selfcal_split = run_target_split_jobs.with_options(
                    task_run_name=f"split_{target_obsid}"
                ).submit(
                    ",".join(target_mslist),
                    target_metafits,
                    workdir,
                    datacolumn="data",
                    timeres=timeavg,
                    freqres=freqavg,
                    prefix=prefix,
                    force_split=True,
                    only_disk=False,
                    time_window=min(time_window, time_interval),
                    time_interval=time_interval,
                    quack_timestamps=quack_timestamps,
                    jobid=jobid,
                    cpu_frac=float(cpu_frac),
                    mem_frac=float(mem_frac),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, expected, succeed = future_selfcal_split.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Spliting of measurement sets for self-calibration is done.\nExpected: {expected}, succeeded: {succeed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Spliting of measurement sets for self-calibration is done."
                )
            except Exception:
                print(
                    "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
                )
                print_banner("Self-calibration subflow failed.")
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in spliting target measurement sets for self-calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], [], []

            ######################################
            # Checking status of self-cal split
            ######################################
            print("Checking measurement sets before spawning self-calibrations.")
            ####################################
            # Filtering any corrupted ms
            #####################################
            selfcal_target_mslist = sorted(glob.glob(workdir + "/selfcal*_ch_*.ms"))
            if (selfcal_target_mslist) == 0:
                print(
                    "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
                )
                print_banner("Self-calibration subflow failed.")
                if emails != "":
                    email_msg = f"[{target_obsid}] No splited measurement set is found for self-calibration. Not continuting for self-calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], [], []

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
                print_banner("Self-calibration subflow failed.")
                if emails != "":
                    email_msg = f"[{target_obsid}] No splited measurement set is found for self-calibration. Not continuting for self-calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], [], []

            print_banner("Selfcal measurement set list:")
            for ms in [os.path.basename(i) for i in selfcal_mslist]:
                print(ms)

            cal_applied = False
            ###################################
            # Apply basic calibration
            ###################################
            if has_cal:
                if emails != "":
                    email_msg = f"[{target_obsid}] Started applying basic calibration solution on self-calibration measurement sets."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Starting task: Applying basic calibration on self-calibration measurement sets."
                )
                try:
                    future_apply_basical_selfcal = run_apply_basiccal_sol.with_options(
                        task_run_name=f"apply_basic_cal_{target_obsid}"
                    ).submit(
                        ",".join(selfcal_mslist),
                        target_metafits,
                        workdir,
                        basic_caldir,
                        overwrite_datacolumn=False,
                        only_amplitude=only_amplitude,
                        applymode="calflag",
                        prefix="selfcal",
                        jobid=jobid,
                        cpu_frac=round(cpu_frac, 2),
                        mem_frac=round(mem_frac, 2),
                        remote_log=remote_logger,
                        obsid=target_obsid,
                        verbose=verbose,
                    )
                    msg, succeed, failed = future_apply_basical_selfcal.result()
                    cal_applied = True
                    if emails != "":
                        email_msg = f"[{target_obsid}] Applying basic calibration solution on self-calibration measurement sets are done.\nSucceeded: {succeed}, failed: {failed}."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    print_banner(
                        "Finished task: Applying basic calibration solution on self-calibration measurement sets are done."
                    )
                except Exception:
                    print_banner(
                        "!!!! WARNING: Error in applying basic calibration solutions on target. Continuing selfcal without basic calibration.!!!!"
                    )
                    traceback.print_exc()
                    if emails != "":
                        email_msg = f"[{target_obsid}] Error occured in applying basic calibration solutions on self-calibration measurement sets."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )

            ########################################
            # Filtering out for self-calibration
            ########################################
            if cal_applied:
                selfcal_applymode = "calonly"
                filtered_selfcalms_list = []
                for selfcalms in selfcal_mslist:
                    unflag_chans, flag_chans = get_chans_flag(
                        msname=selfcalms, n_threads=1
                    )
                    if len(flag_chans) / (len(flag_chans) + len(unflag_chans)) <= 0.8:
                        filtered_selfcalms_list.append(selfcalms)
                    else:
                        print(
                            f"More than 80% channels are flagged for ms: {selfcalms}. Not using for self-calibration."
                        )
                if len(filtered_selfcalms_list) == 0:
                    print(
                        "No measurement set is present with unflagged data for self-calibration after applying basic-calibration."
                    )
                    print_banner("Self-calibration subflow failed.")
                    if emails != "":
                        email_msg = f"[{target_obsid}] No measurement set is present with unflagged data for self-calibration after applying basic-calibration."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    return 1, [], [], []
                else:
                    selfcal_mslist = filtered_selfcalms_list
            else:
                selfcal_applymode = "calflag"

            ###############################################
            # Performing sidereal correction before selfcal
            ###############################################
            os.system(
                f"rm -rf {workdir}/*selfcal_int* {workdir}/*selfcal_pol* {workdir}/caltables/*selfcal*"
            )
            if do_sidereal_cor:
                if emails != "":
                    email_msg = f"[{target_obsid}] Started correcting for solar sidereal motion."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Starting task: Sidereal motion correction for self-calibration measurement sets."
                )
                try:
                    future_sidereal_cor_selfcal = (
                        run_solar_siderealcor_jobs.with_options(
                            task_run_name=f"sidereal_cor_{target_obsid}"
                        ).submit(
                            ",".join(selfcal_mslist),
                            workdir,
                            prefix="selfcal",
                            jobid=jobid,
                            cpu_frac=round(cpu_frac, 2),
                            mem_frac=round(mem_frac, 2),
                            remote_log=remote_logger,
                            obsid=target_obsid,
                            verbose=verbose,
                        )
                    )
                    msg, succeed, failed = future_sidereal_cor_selfcal.result()
                    if emails != "":
                        email_msg = f"[{target_obsid}] Correction for solar sidereal motion is done.\nSucceeded: {succeed}, failed: {failed}."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    print_banner(
                        "Finished task: Correction for solar sidereal motion is done."
                    )
                except Exception:
                    print_banner(
                        "!!! WARNING : Sidereal correction is not successful. !!!"
                    )
                    traceback.print_exc()
                    if emails != "":
                        email_msg = f"[{target_obsid}] Error occured in sidereal motion correction."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )

            #########################################################
            # Basic flagging beforr selfcal on corrected data column
            #########################################################
            if use_solarflagger:
                if emails != "":
                    email_msg = f"[{target_obsid}] Started flagging for self-calibration measurment sets corrected data columns."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Starting task: Flagging selfcal targets corrected data columns."
                )
                try:
                    future_flag = run_flag.with_options(
                        task_run_name=f"flag_selfcal_corrected_{target_obsid}"
                    ).submit(
                        ",".join(selfcal_mslist),
                        target_metafits,
                        workdir,
                        target_outdir,
                        datacolumn="corrected",
                        flag_calibrators=False,
                        flag_bad_spw=False,
                        flag_quack=False,
                        use_rflag=False,
                        use_tfcrop=False,
                        flagdimension="freqtime",
                        flagdata_type="selfcal",
                        run_solarflagger=use_solarflagger,
                        normalize=False,
                        restore_flag=False,
                        jobid=jobid,
                        cpu_frac=round(cpu_frac, 2),
                        mem_frac=round(mem_frac, 2),
                        remote_log=remote_logger,
                        obsid=target_obsid,
                        verbose=verbose,
                    )
                    msg, succeed, failed = future_flag.result()
                    if emails != "":
                        email_msg = f"[{target_obsid}] Flagging for self-calibration measurment sets corrected data columns are done.\nSucceeded: {succeed}, failed: {failed}."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    for s_ms in selfcal_mslist:
                        s_ms = s_ms.rstrip("/")
                        if os.path.exists(f"{s_ms}/.flag_failed"):
                            print(
                                f"Issue in flagging: {s_ms}. Check calibration solutions carefully."
                            )
                    print_banner(
                        "Finished task: Flagging for self-calibration measurment sets corrected data columns are done."
                    )
                except Exception:
                    print_banner(
                        "!!!! WARNING: Flagging error. Examine calibration solutions with caution. !!!!"
                    )
                    traceback.print_exc()
                    if emails != "":
                        email_msg = f"[{target_obsid}] Error occured in flagging self-calibration measurement sets corrected data columns."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )

            #############################
            # Self-calibration
            #############################
            if emails != "":
                email_msg = f"[{target_obsid}] Started self-calibration."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Self-calibrations.")
            if cal_applied:
                print("Calibrator solutions are applied.")
            else:
                print("Calibration solutions are not applied")
            try:
                future_selfcal = run_selfcal_jobs.with_options(
                    task_run_name=f"selfcal_{target_obsid}"
                ).submit(
                    ",".join(selfcal_mslist),
                    workdir,
                    selfcaldir,
                    target_metafits,
                    cal_applied,
                    solint=solint,
                    do_apcal=do_ap_selfcal,
                    do_polcal=do_polcal,
                    solar_selfcal=solar_selfcal,
                    keep_backup=keep_backup,
                    uvrange=uvrange,
                    weight="briggs",
                    robust=0.0,
                    applymode=selfcal_applymode,
                    use_solarflagger=use_solarflagger,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
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
                ) = future_selfcal.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Self-calibration is done.\nIntensity self-calibration, Succeeded: {int_succeed}, failed: {int_failed}\n"
                    email_msg = (
                        f"{email_msg}Average DR: {int_DR}, maximum DR: {max_int_DR}."
                    )
                    if do_polcal:
                        email_msg = f"{email_msg}\nPolarisation self-calibration, Succeeded: {pol_succeed}, failed: {pol_failed}\n"
                        email_msg = f"{email_msg}Average DR: {pol_DR}, maximum DR: {max_pol_DR}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Finished task: Self-calibration is done.")
            except Exception:
                print(
                    "!!!! WARNING: Error in self-calibration on targets. Not applying self-calibration. !!!!"
                )
                print_banner("Self-calibration subflow failed.")
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in self-calibration."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], [], []

            ########################################
            # Checking self-cal caltables
            ########################################
            print(
                f"Searching for self-calibration gaincal tables: {selfcaldir}/selfcal_{target_obsid}*.gcal"
            )
            selfcal_gaincal = sorted(
                glob.glob(f"{selfcaldir}/selfcal_{target_obsid}*.gcal")
            )
            if len(selfcal_gaincal) == 0:
                print(
                    "Self-calibration is not performed and no self-calibration caltable is available."
                )
                print_banner("Self-calibration subflown failed.")
                if emails != "":
                    email_msg = f"[{target_obsid}] Self-calibration is not performed and no self-calibration caltable is available."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, [], [], []
            print_banner(
                f"Self-calibration gaincal tables in calibration directory: {selfcaldir}"
            )
            for gcal in selfcal_gaincal:
                print(f"{os.path.basename(gcal)}")
            print(
                f"Searching for self-calibration bandpass tables: {selfcaldir}/selfcal_{target_obsid}*.bcal"
            )
            selfcal_bandpass = sorted(
                glob.glob(f"{selfcaldir}/selfcal_{target_obsid}*.bcal")
            )
            if len(selfcal_bandpass) > 0:
                print_banner(
                    f"Self-calibration bandpass tables in calibration directory: {selfcaldir}"
                )
                for bpass in selfcal_bandpass:
                    print(f"{os.path.basename(bpass)}")
                selfcal_bandpass = interpolate_bpass(selfcal_bandpass, overwrite=True)
            if do_polcal:
                print(
                    f"Searching for self-calibration polarisation leakage tables: {selfcaldir}/selfcal_{target_obsid}*.dcal"
                )
                selfcal_leakage = sorted(
                    glob.glob(f"{selfcaldir}/selfcal_{target_obsid}*.dcal")
                )
                if len(selfcal_leakage) > 0:
                    print_banner(
                        f"Self-calibration polarisation leakage tables in calibration directory: {selfcaldir}"
                    )
                    for dcal in selfcal_leakage:
                        print(f"{os.path.basename(dcal)}")
                    selfcal_leakage = interpolate_quartical(
                        selfcal_leakage, overwrite=True
                    )

            ###########################################
            # Plotting self-caltables
            ###########################################
            if len(selfcal_gaincal) > 0:
                os.makedirs(f"{target_outdir}/diagnostic_plots", exist_ok=True)
                msg, gcal_plots = plot_caltable_diagnostics(
                    selfcal_gaincal,
                    f"{target_outdir}/diagnostic_plots/{target_obsid}_gcal",
                )
                if msg == 0:
                    print(
                        f"Diagnostic plots for self-calibration gaincal tables are saved in: {gcal_plots}."
                    )
                else:
                    print(
                        "Error in creating diagnostic plots for self-calibration gaincal tables."
                    )

            if len(selfcal_bandpass) > 0:
                os.makedirs(f"{target_outdir}/diagnostic_plots", exist_ok=True)
                msg, bcal_plots = plot_caltable_diagnostics(
                    selfcal_bandpass,
                    f"{target_outdir}/diagnostic_plots/{target_obsid}_bcal",
                )
                if msg == 0:
                    print(
                        f"Diagnostic plots for self-calibration bandpass tables are saved in: {bcal_plots}."
                    )
                else:
                    print(
                        "Error in creating diagnostic plots for self-calibration bandpass tables."
                    )

            if do_polcal:
                if len(selfcal_leakage) > 0:
                    os.makedirs(f"{target_outdir}/diagnostic_plots", exist_ok=True)
                    msg, dcal_plots = plot_quartical_tables(
                        selfcal_leakage,
                        f"{target_outdir}/diagnostic_plots/{target_obsid}_dcal",
                    )
                    if msg == 0:
                        print(
                            f"Diagnostic plots for self-calibration leakage tables are saved in: {dcal_plots}."
                        )
                    else:
                        print(
                            "Error in creating diagnostic plots for self-calibration leakage tables."
                        )
            print_banner("Self-calibration subflow is successful.")
            return 0, selfcal_gaincal, selfcal_bandpass, selfcal_leakage
    except Exception:
        print_banner("Self-calibration subflow failed.")
        traceback.print_exc()
        return 1, [], [], []
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Total run time: {run_time}")
        stop_event.set()
        time.sleep(60)
        log_thread_flow.join()
        if observer is not None:
            clean_shutdown(observer)


############################
# Apply solutions subflow
############################
@flow(
    name="Apply solutions",
    description="Apply calibration solutions on target measurement sets",
    log_prints=True,
)
def applysol_subflow(
    # Core observational inputs
    target_mslist,
    target_metafits,
    target_obsid,
    # I/O and workspace
    workdir,
    basic_caldir,
    selfcaldir,
    target_outdir,
    # Processing controls
    do_applycal,
    do_apply_selfcal,
    has_cal,
    do_polcal,
    do_sidereal_cor,
    use_solarflagger,
    # Applysol
    freqavg,
    timeavg,
    quack_timestamps,
    only_amplitude,
    # Resource management
    cpu_frac,
    mem_frac,
    # Logging / metadata
    jobid,
    timestamp,
    emails,
    remote_logger,
    verbose,
):
    """
    Apply solutions subflow

    Returns
    -------
    int
        Flow success message
    list
        Calibrated measurement set list
    """
    start_time = time.time()
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    applysol_logfile = f"{logdir}/subflow_applysol_{target_obsid}.log"
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id, flow_name, applysol_logfile, poll_interval=3, stop_event=stop_event
    )
    observer = None
    if os.path.exists(f"{workdir}/.jobname_password.npy"):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if applysol_logfile is not None and os.path.exists(applysol_logfile):
            observer = init_logger(
                "applysol_subflow_log",
                applysol_logfile,
                log_type="subflow",
                jobname=jobname,
                password=password,
            )
    if observer is None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    try:
        #############################################
        # Spliting targets if not started already
        #############################################
        prefix = "target"
        if emails != "":
            email_msg = (
                f"[{target_obsid}] Started spliting target for final processing."
            )
            send_task_notification(
                emails,
                email_msg,
                jobid,
                target_obsid,
                timestamp,
                flow_name=f"subflow {flow_name}",
            )
        print_banner(f"Starting task: Spliting {prefix}.")
        try:
            future_split = run_target_split_jobs.with_options(
                task_run_name=f"split_{target_obsid}"
            ).submit(
                ",".join(target_mslist),
                target_metafits,
                workdir,
                datacolumn="data",
                force_split=True,
                freqres=freqavg,
                timeres=timeavg,
                quack_timestamps=quack_timestamps,
                prefix=prefix,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                obsid=target_obsid,
                verbose=verbose,
            )
            msg, expected, succeed = future_split.result()
            if emails != "":
                email_msg = f"[{target_obsid}] Spliting target for final processing is done.\nExpected: {expected}, succeeded: {succeed}."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Finished task: Spliting target for final processing is done.")
        except Exception:
            print_banner("!!!! WARNING: Error in spliting targets. !!!!")
            traceback.print_exc()
            if emails != "":
                email_msg = f"[{target_obsid}] Error occured in spliting target for final processing."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            return 1, []

        ################################
        # Checking splited final ms list
        ################################
        split_target_mslist = sorted(glob.glob(f"{workdir}/target*_ch_*.ms"))
        print(
            "Checking final valid measurement sets before applying solutions and spawning imaging."
        )
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in split_target_mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print(f"Issue in : {ms}")
                os.system(f"rm -rf {ms}")
        split_target_mslist = filtered_mslist
        if len(split_target_mslist) == 0:
            print_banner("No filtered target ms are available in work directory.")
            if emails != "":
                email_msg = f"[{target_obsid}] No un-corrupted target measurement is present for final processing."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            return 1, []
        print(f"Target mslist : {[os.path.basename(i) for i in split_target_mslist]}")

        ####################################
        # Applying basic calibration
        #####################################
        if (do_applycal or do_apply_selfcal) and has_cal:
            if emails != "":
                email_msg = f"[{target_obsid}] Started applying basic calibration solutions on final target measurement sets."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner(
                "Starting task: Applying basic calibration on final target measurement sets."
            )
            try:
                future_apply_basical = run_apply_basiccal_sol.with_options(
                    task_run_name=f"apply_basic_cal_{target_obsid}"
                ).submit(
                    ",".join(split_target_mslist),
                    target_metafits,
                    workdir,
                    basic_caldir,
                    overwrite_datacolumn=True,
                    only_amplitude=only_amplitude,
                    applymode="calflag",
                    prefix="target",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, succeed, failed = future_apply_basical.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Applying basic calibration solutions on final target measurement sets are done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Applying basic calibration solutions on final target measurement sets are done."
                )
            except Exception:
                print_banner(
                    "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further.!!!!"
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in applying basic calibration on final target measurement sets. P-AIRCARS has stopped."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                return 1, []

        ###################################
        # Correct sidereal motion
        ###################################
        if do_sidereal_cor:
            if emails != "":
                email_msg = f"[{target_obsid}] Start correcting sidereal motion of the Sun on final target measurement sets."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner(
                "Starting task: Sidereal motion correction for final target measurement sets."
            )
            try:
                future_sidereal_cor = run_solar_siderealcor_jobs.with_options(
                    task_run_name=f"sidereal_cor_{target_obsid}"
                ).submit(
                    ",".join(split_target_mslist),
                    workdir,
                    prefix="target",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, succeed, failed = future_sidereal_cor.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Sidereal motion correction of the Sun on final target measurement sets are done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Sidereal motion correction of the Sun on final target measurement sets are done."
                )
            except Exception:
                print_banner("!!!! WARNING: Error in applying sidereal correction.!!!!")
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in sidereal motion correction on final target measurement sets."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )

        ########################################
        # Apply self-calibration
        ########################################
        if do_apply_selfcal:
            selfcal_applymode = "calonly"
            for msname in split_target_mslist:
                if not os.path.exists(f"{msname}/.applied_sol"):
                    selfcal_applymode = "calflag"

            if emails != "":
                email_msg = f"[{target_obsid}] Started applying self-calibration on final target measurement sets."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner(
                "Starting task: Applying self-calibration solutions on final target measurement sets."
            )
            try:
                future_apply_selfcal = run_apply_selfcal_sol.with_options(
                    task_run_name=f"apply_selfcal_{target_obsid}"
                ).submit(
                    ",".join(split_target_mslist),
                    target_metafits,
                    workdir,
                    selfcaldir,
                    overwrite_datacolumn=False,
                    applymode=selfcal_applymode,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, gain_succeed, gain_failed, pol_succeed, pol_failed = (
                    future_apply_selfcal.result()
                )
                if emails != "":
                    email_msg = f"[{target_obsid}] Applying self-calibration are done.\nGain solutions applied: Succeeded: {gain_succeed}, failed: {gain_failed}."
                    if do_polcal:
                        email_msg += f"\nPolarisation solution applied: Succeeded: {pol_succeed}, failed: {pol_failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Applying self-calibration on final target measurement sets are done."
                )
            except Exception:
                print_banner(
                    "!!!! WARNING: Error in applying self-calibration solutions on targets. !!!!"
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in applying self-calibration solutions on final target measurement sets."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )

        ###################################
        # Basic flagging on corrected data
        ###################################
        if use_solarflagger:
            if emails != "":
                email_msg = f"[{target_obsid}] Started flagging of final target measurement sets corrected data column."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner(
                "Starting task: Flagging final target measurement sets corrected data column."
            )
            try:
                future_flag = run_flag.with_options(
                    task_run_name=f"flag_target_corrected_{target_obsid}"
                ).submit(
                    ",".join(split_target_mslist),
                    target_metafits,
                    workdir,
                    target_outdir,
                    datacolumn="corrected",
                    flag_calibrators=False,
                    flag_bad_spw=False,
                    flag_quack=False,
                    use_rflag=False,
                    use_tfcrop=False,
                    flagdimension="freqtime",
                    flagdata_type="target",
                    run_solarflagger=use_solarflagger,
                    normalize=False,
                    restore_flag=False,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, succeed, failed = future_flag.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Flagging of final target measurement sets corrected data columns are done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner(
                    "Finished task: Flagging of final target measurement sets corrected data columns are done."
                )
            except Exception:
                print_banner(
                    "!!!! WARNING: Flagging error. Examine calibration solutions with caution. !!!!"
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in flagging of final target measurement sets corrected data columns."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
        return 0, split_target_mslist
    except Exception:
        traceback.print_exc()
        return 1, []
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Total run time: {run_time}")
        stop_event.set()
        time.sleep(60)
        log_thread_flow.join()
        if observer is not None:
            clean_shutdown(observer)


############################
# Imaging subflow
############################
@flow(
    name="Imaging",
    description="Imaging target measurement sets",
    log_prints=True,
)
def imaging_subflow(
    # Core observational inputs
    split_target_mslist,
    target_metafits,
    target_obsid,
    # I/O and workspace
    workdir,
    selfcaldir,
    target_outdir,
    # Processing controls
    do_imaging,
    do_pbcor,
    do_polcal,
    keep_backup,
    make_overlay,
    # Imaging
    image_freqres,
    image_timeres,
    pol,
    freqrange,
    timerange,
    minuv,
    weight,
    robust,
    clean_threshold,
    use_multiscale,
    use_solar_mask,
    cutout_rsun,
    # Resource management
    cpu_frac,
    mem_frac,
    # Logging / metadata
    jobid,
    timestamp,
    emails,
    remote_logger,
    verbose,
):
    """
    Imaging subflow

    Returns
    -------
    int
        Flow success message
    """
    start_time = time.time()
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    imaging_subflow_logfile = f"{logdir}/subflow_imaging_{target_obsid}.log"
    ctx = get_run_context()
    flow_id = str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread_flow = start_flow_log_saver(
        flow_id,
        flow_name,
        imaging_subflow_logfile,
        poll_interval=3,
        stop_event=stop_event,
    )
    observer = None
    if os.path.exists(f"{workdir}/.jobname_password.npy"):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if imaging_subflow_logfile is not None and os.path.exists(
            imaging_subflow_logfile
        ):
            observer = init_logger(
                "imaging_subflow_log",
                imaging_subflow_logfile,
                log_type="subflow",
                jobname=jobname,
                password=password,
            )
    if observer is None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    try:
        if do_imaging:
            ######################################
            # Imaging
            ######################################
            if image_freqres > 0:
                print(f"Image frequency resolution: {image_freqres} MHz.")
            else:
                print("Image frequency resolution: entire corase channel.")
            if image_timeres > 0:
                print(f"Image time resolution: {image_timeres} s.")
            else:
                print("Imaging entire scan.")
            pol = pol.upper()
            if pol not in ["I", "IQUV"]:
                pol = "IQUV"

            if (
                not do_polcal
            ):  # Only if do_polcal is False, overwrite to make only Stokes I
                pol = "I"

            if emails != "":
                email_msg = f"[{target_obsid}] Started final imaging."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Final imaging.")
            try:
                future_imaging = run_imaging_jobs.with_options(
                    task_run_name=f"imaging_{target_obsid}"
                ).submit(
                    ",".join(split_target_mslist),
                    workdir,
                    target_outdir,
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
                    savemodel=keep_backup,
                    saveres=keep_backup,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, succeed, failed, total_images = future_imaging.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Final imaging is done.\nSucceeded: {succeed}, failed: {failed}.\nTotal images made: {total_images}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Finished task: Final imaging is done.")
            except Exception:
                print_banner(
                    "!!!! WARNING: Final imaging on all measurement sets is not successful. Check the image directory. !!!!"
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = "Error occured in final imaging. P-AIRCARS has stopped."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )

        ########################################
        # Naming of image directory
        ########################################
        if weight == "briggs":
            weight_str = f"{weight}_{robust}"
        else:
            weight_str = weight
        if image_freqres == -1 and image_timeres == -1:
            imagedir = target_outdir + f"/imagedir_f_all_t_all_pol_{pol}_w_{weight_str}"
        elif image_freqres != -1 and image_timeres == -1:
            imagedir = (
                target_outdir
                + f"/imagedir_f_{image_freqres}_t_all_pol_{pol}_w_{weight_str}"
            )
        elif image_freqres == -1 and image_timeres != -1:
            imagedir = (
                target_outdir
                + f"/imagedir_f_all_t_{image_timeres}_pol_{pol}_w_{weight_str}"
            )
        else:
            imagedir = (
                target_outdir
                + f"/imagedir_f_{image_freqres}_t_{image_timeres}_pol_{pol}_w_{weight_str}"
            )

        ##################################
        # Check presence of images
        ##################################
        images = sorted(glob.glob(f"{imagedir}/images/*.fits"))
        if len(images) == 0:
            print_banner(f"No image is present in image directory: {imagedir}/images")
            if emails != "":
                email_msg = f"[{target_obsid}] No image is present in image directory."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            return 1

        ###########################
        # Primary beam correction
        ###########################
        if do_pbcor:
            if emails != "":
                email_msg = f"[{target_obsid}] Started primary beam correction."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Primary beam correction.")
            try:
                future_pbcor = run_apply_pbcor.with_options(
                    task_run_name=f"apply_pbcor_{target_obsid}"
                ).submit(
                    f"{imagedir}/images",
                    target_metafits,
                    workdir,
                    leakage_dir=selfcaldir,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, succeed, failed = future_pbcor.result()
                if emails != "":
                    email_msg = f"[{target_obsid}] Primary beam correction is done.\nSucceeded: {succeed}, failed: {failed}."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )
                print_banner("Finished task: Primary beam correction is done.")
                print(f"Final image directory: {imagedir}/images")
            except Exception:
                print_banner(
                    "!!!! WARNING: Primary beam corrections of the final images are not successful. !!!!"
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in primary beam correction. P-AIRCARS has stopped."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )

        #################################################################
        # Filtering only coarse channel images for default overlay mode
        #################################################################
        if make_overlay is False and len(images) > 0:
            images = filter_images(images, min_time_sep=60.0)
        internet_on = internet_available()
        if not internet_on:
            print("Internet connection is not available. Can not make overlays")
        else:
            #################################
            # Start overlays
            #################################
            if emails != "":
                email_msg = f"[{target_obsid}] Started making overlays."
                send_task_notification(
                    emails,
                    email_msg,
                    jobid,
                    target_obsid,
                    timestamp,
                    flow_name=f"subflow {flow_name}",
                )
            print_banner("Starting task: Making overlay on EUV images.")
            try:
                future_overlay = run_make_overlay.with_options(
                    task_run_name=f"make_overlay_{target_obsid}"
                ).submit(
                    f"{imagedir}/images",
                    f"{imagedir}/overlay_pngs",
                    workdir=workdir,
                    all_overlay=make_overlay,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    remote_log=remote_logger,
                    obsid=target_obsid,
                    verbose=verbose,
                )
                msg, succeed, failed = future_overlay.result()
                if msg == 0:
                    if emails != "":
                        email_msg = f"[{target_obsid}] Making overlays are done.\nSucceeded: {succeed}, failed: {failed}."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    print_banner("Finished task: Making overlays are done.")
                    print(f"Final image directory: {imagedir}/overlay_pngs")
                else:
                    if emails != "":
                        email_msg = f"[{target_obsid}] Making overlays are not successful EUV images could not be download.\nSucceeded: {succeed}, failed: {failed}."
                        send_task_notification(
                            emails,
                            email_msg,
                            jobid,
                            target_obsid,
                            timestamp,
                            flow_name=f"subflow {flow_name}",
                        )
                    print_banner("Finished task: Making overlays are not successful.")
                    if len(glob.glob(f"{imagedir}/overlay_pngs/*.png")) == 0:
                        os.system(f"rm -rf {imagedir}/overlay_pngs")
                    else:
                        print(f"Final image directory: {imagedir}/overlay_pngs")
            except Exception:
                print_banner(
                    "!!!! WARNING: Overlay of the images are not successful. !!!!"
                )
                traceback.print_exc()
                if emails != "":
                    email_msg = f"[{target_obsid}] Error occured in making overlays."
                    send_task_notification(
                        emails,
                        email_msg,
                        jobid,
                        target_obsid,
                        timestamp,
                        flow_name=f"subflow {flow_name}",
                    )

        ##################################################################
        # Sending image collage and DR information
        ##################################################################
        if emails != "" and len(images) > 0:
            dyn_range_list = []
            for image in images:
                dr = fits.getheader(image)["RMSDYN"]
                dyn_range_list.append(dr)
            max_DR = np.nanmax(dyn_range_list)
            min_DR = np.nanmin(dyn_range_list)
            filtered_images = filter_images(images, min_time_sep=-1)
            outfile = plot_hpc_collage(
                filtered_images, outfile=f"{workdir}/{target_obsid}_collage.png"
            )
            email_msg = f"[{target_obsid}] Imaging is completed.\nMaximum dynamic range: {max_DR}\nMinimum dynamic range: {min_DR}."
            send_task_notification(
                emails,
                email_msg,
                jobid,
                target_obsid,
                timestamp,
                flow_name=f"subflow {flow_name}",
                attachments=[outfile],
            )
            os.system(f"rm -rf {outfile}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Total run time: {run_time}")
        stop_event.set()
        time.sleep(60)
        log_thread_flow.join()
        if observer is not None:
            clean_shutdown(observer)
