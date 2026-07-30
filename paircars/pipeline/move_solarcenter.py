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
from paircars.utils.mwa_utils import get_ncoarse
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.resource_utils import drop_cache
from paircars.utils.sunpos_utils import move_to_sun
from paircars.utils.udocker_utils import (
    check_udocker_container,
    initialize_wsclean_container,
)

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def move_to_sun_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = move_to_sun(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def main(
    mslist,
    workdir="",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Run the flagging pipeline for a measurement set.

    Parameters
    ----------
    mslist : str
        List of measurement sets
    workdir : str, optional
        Working directory
    cpu_frac : float, optional
        Fraction of total CPU resources to use. Default is 0.8.
    mem_frac : float, optional
        Fraction of total memory to use. Default is 0.8.
    logfile : str or None, optional
        Path to the log file for saving logs. If None, logging to file is skipped.
    jobid : int, optional
        Numeric job ID used for PID tracking. Default is 0.
    verbose : bool, optional
        Verbose logs
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials in the workdir. Default is False.
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
        time.sleep(1)
        jobname, password = np.load(
            f"{workdir}/.jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "do_flagging", logfile, jobname=jobname, password=password
            )
            
    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        succeed = 0
        failed = len(mslist)

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
            return 1, succeed, failed

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
            "Starting to move phasecenter to solarcenters.", no_print=True
        ).splitlines():
            logger.info(banner)
        client_info = dask_client.scheduler_info()["workers"]
        njobs = len(client_info)
        worker_mem_list = []
        for addr, w in client_info.items():
            worker_mem_list.append(w["memory_limit"] / 1024**3)
        mem_limit = round(min(worker_mem_list), 3)
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

        tasks = [
            delayed(move_to_sun_wrapper)(msname, ncpu=n_threads) for msname in mslist
        ]
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

        failed = sum(results)
        succeed = len(mslist) - failed
        logger.info(f"Total measurement sets: {len(mslist)}")
        logger.info(f"Total success: {succeed}")
        logger.info(f"Total failure: {failed}")
        if len(mslist) == failed:
            msg = 1
        else:
            msg = 0
    except Exception:
        logger.exception(
            "Exception occured in moving phasecenter to solar center.", exc_info=True
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
    usage = "Move phasecenter of the measurement set to the Sun"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist", type=str, help="Measurement set list, comma separated"
    )
    basic_args.add_argument(
        "--workdir", type=str, default="", help="Name of work directory"
    )

    # Advanced switches
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose logs")
    adv_args.add_argument("--jobid", type=int, default=0, help="Job ID")

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
        workdir=args.workdir,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
