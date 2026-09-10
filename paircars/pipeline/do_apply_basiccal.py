import logging
import numpy as np
import argparse
import traceback
import time
import glob
import sys
import os
from casatools import table, msmetadata
from dask import delayed
from astropy.io import fits
from paircars.utils.basic_utils import (
    suppress_output,
    print_banner,
    capture_all_output,
)
from paircars.utils.calibration import get_nearest_bandpass_table, get_quartical_soltype
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.mwa_utils import get_ncoarse
from paircars.utils.ms_metadata import check_datacolumn_valid
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.resource_utils import drop_cache, limit_threads
from paircars.utils.udocker_utils import run_quartical

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def applysol_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = applysol(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def applysol(
    msname,
    workdir,
    gaintable=[],
    gainfield=[],
    interp=[],
    only_amplitude=False,
    applymode="calflag",
    quartical_table=[],
    overwrite_datacolumn=False,
    n_threads=1,
    mem_limit=1,
    force_apply=False,
    soltype="basic",
):
    """
    Apply calibration solutions

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    gaintable : list, optional
        Caltable list
    gainfield : list, optional
        Gain field list
    interp : list, optional
        Gain interpolation
    only_amplitude : bool, optional
        Apply only amplitude
    applymode : str, optional
        Apply mode
    quartical_table : list, optional
        Quartical caltables
    overwrite_datacolumn : bool, optional
        Overwrite data column with corrected solutions
    n_threads : int, optional
        Number of OpenMP threads
    mem_limit : float, optional
        Memory limit in GB
    force_apply : bool, optional
        Force to apply solutions if it is already applied
    soltype : str, optional
        Solution type

    Returns
    -------
    int
        Success message of gain solution
    int
        Success message of polarisation solution
    """
    n_threads = max(1, n_threads)
    mem_limit = abs(mem_limit)

    with limit_threads(n_threads=n_threads):
        from casatasks import applycal, flagdata, split, clearcal

    if soltype == "basic":
        check_file = "/.applied_sol"
    else:
        check_file = "/.applied_selfcalsol"
    try:
        if os.path.exists(msname + check_file) and not force_apply:
            print("Solutions are already applied.")
            gain_msg = 0
            if os.path.exists(f"{msname}/.nopolselfcal"):
                pol_msg = 1
            else:
                pol_msg = 0
            return gain_msg, pol_msg
        else:
            if os.path.exists(msname + check_file) and force_apply:
                print("Undo previous flagging.")
                print(f"clearcal(vis='{msname})")
                with suppress_output():
                    clearcal(vis=msname)
                print(
                    f"flagdata(vis='{msname}', mode='unflag', spw='0', flagbackup=False)"
                )
                with suppress_output():
                    flagdata(vis=msname, mode="unflag", spw="0", flagbackup=False)
                if os.path.exists(msname + ".flagversions"):
                    os.system(f"rm -rf {msname}.flagversions")
            filtered_gaintable = []
            only_ampcals = []
            for gtable in gaintable:
                if os.path.exists(gtable):
                    if only_amplitude and gtable.endswith(".bcal"):
                        print("Only amplitude part of the solution will be applied.")
                        os.system(f"cp -r {gtable} {gtable}.amp")
                        tb = table()
                        tb.open(f"{gtable}.amp", nomodify=False)
                        gain = tb.getcol("CPARAM")
                        gain = np.abs(gain)
                        tb.putcol("CPARAM", gain)
                        tb.flush()
                        tb.close()
                        gtable = f"{gtable}.amp"
                        only_ampcals.append(gtable)
                    filtered_gaintable.append(gtable)
            gaintable = filtered_gaintable
            print(
                f"Applying solution on ms: {msname} from gaintables: {','.join(gaintable)}."
            )
            try:
                has_kcross = False
                for g in gaintable:
                    if g.endswith("kcrosscal"):
                        has_kcross = True
                if not has_kcross and soltype == "basic":
                    print("No crosshand phase solutions applied.")
                    os.system(f"touch {msname}/.nokcross")
                applyal_cmd = (
                    f"applycal("
                    f"vis='{msname}',"
                    f"gaintable={gaintable},"
                    f"gainfield={gainfield},"
                    f"applymode={applymode},"
                    f"interp={interp},"
                    f"calwt={[False] * len(gaintable)},"
                    "flagbackup=False)"
                )
                print(applyal_cmd)
                with suppress_output():
                    applycal(
                        vis=msname,
                        gaintable=gaintable,
                        gainfield=gainfield,
                        applymode=applymode,
                        interp=interp,
                        calwt=[False] * len(gaintable),
                        flagbackup=False,
                    )
                if len(only_ampcals) > 0:
                    for ampcal in only_ampcals:
                        os.system(f"rm -rf {ampcal}")
                gain_msg = 0
            except Exception:
                traceback.print_exc()
                gain_msg = 1
            if gain_msg == 0 and soltype != "basic":
                os.system(f"rm -rf {msname}/.nopolselfcal")
                qc_success = False
                if len(quartical_table) > 0:
                    for qc in quartical_table:
                        if os.path.exists(qc) is False:
                            print(f"Quartical table: {qc} is not present.")
                        else:
                            print(
                                f"Applying solution on ms: {msname} from quartical table: {qc}."
                            )
                            temp_pol_caltable = (
                                f"{workdir}/{os.path.basename(qc)}.tempcal"
                            )
                            quartical_log = f"{workdir}/{os.path.basename(qc)}.log"
                            qc = qc.rstrip("/")
                            soltypes = get_quartical_soltype(qc)
                            if len(soltypes) == 0:
                                print(
                                    f"No solution is present in quartical table {qc}."
                                )
                                os.system(f"rm -rf {quartical_log}")
                                os.system(f"rm -rf {temp_pol_caltable}")
                            else:
                                soltype = soltypes[0]
                                quartical_args = [
                                    "goquartical",
                                    f"input_ms.path={msname}",
                                    "input_ms.data_column=CORRECTED_DATA",
                                    "output.log_to_terminal=True",
                                    f"output.log_directory={quartical_log}",
                                    f"output.gain_directory={temp_pol_caltable}",
                                    "output.overwrite=True",
                                    "output.products=[corrected_data]",
                                    "output.columns=[CORRECTED_DATA]",
                                    "output.flags=True",
                                    f"solver.terms=[{soltype}]",
                                    "solver.iter_recipe=[0]",
                                    "solver.propagate_flags=True",
                                    f"solver.threads={n_threads}",
                                    "dask.threads=1",
                                    f"{soltype}.type=complex",
                                    f"{soltype}.load_from={qc}/{soltype}",
                                ]
                                quartical_cmd = " ".join(quartical_args)
                                quartical_msg = run_quartical(
                                    quartical_cmd, "paircarsquartical", verbose=False
                                )
                                if quartical_msg != 0:
                                    print("Quartical solutions did not apply.")
                                else:
                                    print(
                                        f"Quartical solutions applied successfully from: {qc}."
                                    )
                                    qc_success = True
                            os.system(f"rm -rf {quartical_log}")
                            os.system(f"rm -rf {temp_pol_caltable}")
                if not qc_success:
                    print("No quartical solutions applied.")
                    os.system(f"touch {msname}/.nopolselfcal")
                    pol_msg = 1
                else:
                    pol_msg = 0
            elif gain_msg == 0 and soltype == "basic":
                pol_msg = 0
            else:
                os.system(f"touch {msname}/.nopolselfcal")
                pol_msg = 1
        if gain_msg == 0:
            os.system("touch " + msname + check_file)
        if overwrite_datacolumn:
            print(f"Over writing data column with corrected data for ms: {msname}.")
            outputvis = msname.split(".ms")[0] + "_cor.ms"
            if os.path.exists(outputvis):
                os.system(f"rm -rf {outputvis}")
            touch_file_names = glob.glob(f"{msname}/.*")
            if len(touch_file_names) > 0:
                touch_file_names = [os.path.basename(f) for f in touch_file_names]
            print(
                f"split(vis='{msname}', outputvis='{outputvis}', datacolumn='corrected')"
            )
            with suppress_output():
                split(vis=msname, outputvis=outputvis, datacolumn="corrected")
            if os.path.exists(outputvis):
                os.system(f"rm -rf {msname} {msname}.flagversions")
                os.system(f"mv {outputvis} {msname}")
            for t in touch_file_names:
                os.system(f"touch {msname}/{t}")
        return gain_msg, pol_msg
    except Exception:
        traceback.print_exc()
        return 1, 1


def run_all_applysol(
    mslist,
    target_metafits,
    dask_client,
    workdir,
    caldir,
    overwrite_datacolumn=False,
    applymode="calflag",
    only_amplitude=False,
    force_apply=False,
    n_threads=1,
    mem_limit=1,
    logger=None,
):
    """
    Apply basic-calibration solutions on all target scans

    Parameters
    ----------
    mslist : list
        Measurement set list
    target_metafits : str
        Target metafits file
    dask_client : dask.client
        Dask client
    workdir : str
        Working directory
    caldir : str
        Calibration directory
    overwrite_datacolumn : bool, optional
        Overwrite data column or not
    applymode : str, optional
        Apply mode
    only_amplitude: bool, optional
        Apply only amplitude
    force_apply : bool, optional
        Force to apply solutions even already applied
    n_threads : int, optional
        CPU threads to use
    mem_limit : float, optional
        Memory to use in GB

    Returns
    --------
    list
        Calibrated target scans
    int
        Succeeded ms number
    int
        Failed ms number
    """
    if logger is None:
        logger = get_logger_safe()

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        succeed = 0
        failed = len(mslist)

    try:
        os.chdir(workdir)
        logger.debug(f"Current working directory: {os.getcwd()}")
        if isinstance(mslist, list):
            mslist = np.array(mslist)
        mslist = np.unique(mslist).tolist()
        target_header = fits.getheader(target_metafits)
        target_attn = target_header["ATTEN_DB"]
        logger.debug(f"Target attenuation: {target_attn}dB")
        bandpass_table = glob.glob(caldir + f"/calibrator*.bcal.att{target_attn}")
        if len(bandpass_table) == 0:
            bandpass_table = glob.glob(caldir + "/calibrator*.bcal")
            logger.warning("No bandpass solution with target attenuation is found.")
            att_scaled = False
        else:
            att_scaled = True
        crossphase_table = glob.glob(caldir + "/calibrator*.kcrosscal")

        if len(bandpass_table) == 0:
            logger.error(
                f"No bandpass table is present in calibration directory : {caldir}."
            )
            return 1, 0, 0
        if len(crossphase_table) == 0:
            logger.warning(
                f"No crosshand phase solution is present in calibration directory : {caldir}. Applying only bandpass solutions."
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

        mslist = filtered_mslist
        if len(mslist) == 0:
            logger.error("No valid measurement set.")
            return 1, 0, 0

        ####################################
        # Applycal jobs
        ####################################
        logger.info(f"Total ms list: {len(mslist)}")
        tasks = []
        failed = 0
        for ms in mslist:
            if att_scaled:
                os.system(f"touch {ms}/.att")
            else:
                os.system(f"touch {ms}/.noatt")
            msmd = msmetadata()
            msmd.open(ms)
            ms_freq = msmd.meanfreq(0, unit="MHz")
            msmd.close()
            final_bpasstable = get_nearest_bandpass_table(bandpass_table, ms_freq)
            final_gaintable = [final_bpasstable]
            interp = ["linear,linear"]
            if len(crossphase_table) > 0:
                final_crossphasetable = get_nearest_bandpass_table(
                    crossphase_table, ms_freq
                )
                final_gaintable.append(final_crossphasetable)
                interp.append("linear, linear")
            if len(final_gaintable) > 0:
                tasks.append(
                    delayed(applysol_wrapper)(
                        ms,
                        workdir,
                        gaintable=final_gaintable,
                        overwrite_datacolumn=overwrite_datacolumn,
                        applymode=applymode,
                        interp=interp,
                        only_amplitude=only_amplitude,
                        n_threads=n_threads,
                        mem_limit=mem_limit,
                        force_apply=force_apply,
                    )
                )
            else:
                failed += 1
        result_wrapper = list(dask_client.gather(dask_client.compute(tasks)))
        results = []
        for r in result_wrapper:
            results.append(r[1])
            logger.debug("================")
            logger.debug(f"Worker log for: {os.path.basename(r[0])}")
            logger.debug("================")
            for line in r[2].splitlines():
                logger.debug(line)
            for line in r[3].splitlines():
                logger.debug(line)

        gain_msg = []
        for r in results:
            gain_msg.append(r[0])
        apply_failed = sum(gain_msg)
        succeed = len(mslist) - apply_failed - failed
        logger.info(f"Total measurement sets: {len(mslist)}")
        logger.info(f"Total success: {succeed}")
        logger.info(f"No solutions available for: {failed}")
        logger.info(f"Total solution apply failed: {apply_failed}")

        if failed + apply_failed == len(mslist):
            logger.error(
                "Applying basic calibration solutions for target scans are not done successfully."
            )
            return 1, succeed, failed + apply_failed
        else:
            logger.info(
                "Applying basic calibration solutions for target are done successfully."
            )
            return 0, succeed, failed + apply_failed
    except Exception:
        logger.exception(
            "Applying basic calibration solutions for target scans are not done successfully.",
            exc_info=True,
        )
        return 1, succeed, failed


def main(
    mslist,
    target_metafits,
    workdir,
    caldir,
    applymode="calflag",
    only_amplitude=False,
    overwrite_datacolumn=False,
    force_apply=False,
    do_post_flag=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    verbose=False,
    dask_client=None,
):
    """
    Apply calibration solutions to a list of measurement sets with optional post-flagging.

    Parameters
    ----------
    mslist : str
        Measurement set list (comma separated).
    target_metafits : str
        Target metafits file
    workdir : str
        Directory for logs, PID files, and temporary data products.
    caldir : str
        Path to directory containing calibration tables (e.g., bandpass, gain, polarization).
    applymode : str, optional
        CASA calibration application mode (e.g., "calonly", "calflag", "flagonly"). Default is "calflag".
    only_amplitude : bool, optional
        Apply only amplitude
    overwrite_datacolumn : bool, optional
        If True, overwrites the CORRECTED column during calibration. Default is False.
    force_apply : bool, optional
        If True, forces re-application of calibration even if it appears already applied. Default is False.
    start_remote_log : bool, optional
        Whether to enable remote logging using job credentials in `workdir`. Default is False.
    cpu_frac : float, optional
        Fraction of CPU resources to allocate per worker. Default is 0.8.
    mem_frac : float, optional
        Fraction of system memory to allocate per worker. Default is 0.8.
    logfile : str or None, optional
        Path to the logfile. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Identifier for tracking the job and saving PID. Default is 0.
    verbose : bool, optional
        Verbose logs
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
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
                "apply_basiccal", logfile, jobname=jobname, password=password
            )

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        succeed = 0
        failed = len(mslist)

    total_ncoarse = 0
    for msname in mslist:
        ncoarse = get_ncoarse(msname)
        total_ncoarse += ncoarse
    total_ncoarse = max(1, total_ncoarse)
    logger.debug(f"Total coarse channels: {total_ncoarse}")

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
            return 1, succeed, failed
        scale_worker_and_wait(dask_cluster, dask_client, nworker)

    try:
        for banner in print_banner(
            "Starting applying solutions.", no_print=True
        ).splitlines():
            logger.info(banner)
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
        if caldir == "" or not os.path.exists(caldir):
            logger.error("Provide existing caltable directory.")
            msg = 1
        else:
            msg, succeed, failed = run_all_applysol(
                mslist,
                target_metafits,
                dask_client,
                workdir,
                caldir,
                overwrite_datacolumn=overwrite_datacolumn,
                applymode=applymode,
                only_amplitude=only_amplitude,
                force_apply=force_apply,
                n_threads=n_threads,
                mem_limit=mem_limit,
                logger=logger,
            )
    except Exception:
        logger.exception("Exception in applying solutions.", exc_info=True)
        msg = 1
    finally:
        time.sleep(5)
        if observer is not None:
            clean_shutdown(observer)
        for msname in mslist:
            drop_cache(msname)
        if dask_cluster is not None:
            dask_client.shutdown()
            dask_client.close()
            dask_cluster.close()
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")
    return msg, succeed, failed


def cli():
    parser = argparse.ArgumentParser(
        description="Apply basic calibration solutions to target scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required)",
    )
    basic_args.add_argument(
        "--target_metafits",
        type=str,
        default="",
        required=True,
        help="Target metafits file",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory for intermediate files",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Directory containing calibration tables",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--applymode",
        type=str,
        default="calflag",
        help="Applycal mode (e.g. 'calonly', 'calflag')",
    )
    adv_args.add_argument(
        "--only_amplitude",
        action="store_true",
        help="Apply only amplitude",
    )
    adv_args.add_argument(
        "--overwrite_datacolumn",
        action="store_true",
        help="Overwrite corrected data column in MS",
    )
    adv_args.add_argument(
        "--force_apply",
        action="store_true",
        help="Force apply calibration even if already applied",
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose logs")
    adv_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and process tracking"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU fraction to use"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory fraction to use"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg, _, _ = main(
        args.mslist,
        args.target_metafits,
        args.workdir,
        args.caldir,
        applymode=args.applymode,
        overwrite_datacolumn=args.overwrite_datacolumn,
        only_amplitude=args.only_amplitude,
        force_apply=args.force_apply,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        verbose=args.verbose,
        jobid=args.jobid,
    )
    return msg
