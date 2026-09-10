import logging
import numpy as np
import argparse
import time
import sys
import os
from dask import delayed
from paircars.utils.basic_utils import (
    print_banner,
    capture_all_output,
)
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.mwa_utils import get_ncoarse
from paircars.utils.resource_utils import drop_cache
from paircars.utils.sunpos_utils import correct_solar_sidereal_motion
from paircars.utils.udocker_utils import (
    check_udocker_container,
    initialize_wsclean_container,
)

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def correct_solar_sidereal_motion_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = correct_solar_sidereal_motion(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def cor_sidereal_motion(
    mslist,
    dask_client,
    workdir,
    n_threads=1,
    logger=None,
):
    """
    Perform sidereal motion correction

    Parameters
    ----------
    mslist : list
        Measurement set list
    dask_client : dask.client
        Dask client
    workdir : str
        Work directory
    n_threads : int, optional
        Number of CPU threads to use

    Returns
    -------
    int
        Success message
    list
        List of sidereal motion corrected measurement sets
    int
        Succeeded ms number
    int
        Failed ms number
    """
    if logger is None:
        logger = get_logger_safe()

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, [], 0, 0
    else:
        succeed = 0
        failed = len(mslist)
    try:
        container_name = "paircarswsclean"
        container_present = check_udocker_container(container_name)
        if not container_present:
            logger.debug(f"Initializing {container_name}.")
            container_name = initialize_wsclean_container(
                name=container_name, verbose=True
            )
            if container_name is None:
                logger.critical(
                    "Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1, [], succeed, failed

        tasks = []
        for ms in mslist:
            tasks.append(
                delayed(correct_solar_sidereal_motion_wrapper)(ms, ncpu=n_threads)
            )

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

        splited_ms_list_phaserotated = []
        for i in range(len(results)):
            msg = results[i]
            ms = mslist[i]
            if msg == 0:
                if os.path.exists(ms + "/.sidereal_cor"):
                    splited_ms_list_phaserotated.append(ms)
        succeed = len(splited_ms_list_phaserotated)
        failed = len(mslist) - succeed

        if len(splited_ms_list_phaserotated) == 0:
            logger.error(
                "Sidereal motion correction is not successful for any measurement set."
            )
            return 1, [], succeed, failed
        else:
            logger.info(f"Total measurement sets: {len(mslist)}")
            logger.info(f"Total success: {len(splited_ms_list_phaserotated)}")
            logger.info(
                f"Total failure: {len(mslist)-len(splited_ms_list_phaserotated)}"
            )
            logger.info("Sidereal motion corrections are done successfully.")
            return 0, splited_ms_list_phaserotated, succeed, failed
    except Exception:
        logger.exception(
            "Sidereal motion correction is not successful for any measurement set.",
            exc_info=True,
        )
        return 1, [], succeed, failed


def main(
    mslist,
    workdir="",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    verbose=False,
    dask_client=None,
):
    """
    Run a parallel processing pipeline for solar sidereal motion correction

    Parameters
    ----------
    mslist : str
        Comma-separated list of paths to measurement sets to be processed.
    workdir : str, optional
        Directory for logs, intermediate files, and other outputs.
        If empty, defaults to the directory of the first MS with `/workdir` appended.
        Default is "".
    cpu_frac : float, optional
        Fraction of total CPU cores to allocate per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of total system memory to allocate per task. Default is 0.8.
    logfile : str or None, optional
        Path to the log file for capturing logs. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Unique job identifier used for PID tracking and task differentiation. Default is 0.
    start_remote_log : bool, optional
        Whether to enable remote logging based on credentials stored in the workdir. Default is False.
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
                "do_sidereal_cor", logfile, jobname=jobname, password=password
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
    logger.info(f"Total coarse channels: {total_ncoarse}")

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
            return 1
        scale_worker_and_wait(dask_cluster, dask_client, nworker)

    try:
        for banner in print_banner(
            "Starting sidereal motion correction.", no_print=True
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

        msg, final_target_mslist, succeed, failed = cor_sidereal_motion(
            mslist,
            dask_client,
            workdir,
            n_threads=n_threads,
            logger=logger,
        )
    except Exception:
        logger.exception(
            "Exception occured in sidereal motion correction", exc_info=True
        )
        msg = 1
    finally:
        time.sleep(5)
        if observer is not None:
            clean_shutdown(observer)
        for ms in mslist:
            drop_cache(ms)
        if dask_cluster is not None:
            dask_client.shutdown()
            dask_client.close()
            dask_cluster.close()
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")
    return msg, succeed, failed


def cli():
    parser = argparse.ArgumentParser(
        description="Correct measurement sets for sidereal motion",
        formatter_class=SmartDefaultsHelpFormatter,
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
    basic_args.add_argument("--workdir", type=str, default="", help="Working directory")

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration and imaging parameters\n###################"
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

    msg, _, _ = main(
        mslist=args.mslist,
        workdir=args.workdir,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
