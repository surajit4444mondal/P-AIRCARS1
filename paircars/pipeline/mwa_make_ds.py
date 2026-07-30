import logging
import numpy as np
import argparse
import warnings
import time
import glob
import sys
import os
from dask import delayed
from paircars.utils.basic_utils import (
    print_banner,
    capture_all_output,
)
from paircars.utils.ds_utils import calc_dynamic_spectrum
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.mwa_ploting_utils import make_ds_plot
from paircars.utils.mwa_utils import get_MWA_OBSID, get_ncoarse
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.resource_utils import drop_cache

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def calc_dynamic_spectrum_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = calc_dynamic_spectrum(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def make_solar_DS(
    mslist,
    dask_client,
    metafits,
    workdir,
    outdir,
    plot_quantity="TB",
    extension="png",
    showgui=False,
    overwrite=False,
    n_threads=1,
    mem_limit=1,
    njobs=1,
    logger=None,
):
    """
    Make solar dynamic spectrum and plots

    Parameters
    ----------
    mslist : list
        Measurement set list (Provide only same obsid measurement set list)
    dask_client: dask.client
        Dask client
    metafits : str
        Metafits file
    workdir : str
        Work directory
    outdir : str
        Output directory
    plot_quantity : str, optional
        Plotting quantity (TB or flux)
    extension : str, optional
        Image file extension
    showgui : bool, optional
        Show GUI
    overwrite : bool, optional
        Overwrite plot
    n_threads : int, optional
        CPU threads to use
    mem_limit : float, optional
        Memory to use in GB
    njobs : int, optional
        Number of parallel jobs

    Returns
    -------
    int
        Success message
    str
        Plot file name
    int
        Succeeded ms number
    int
        Failed ms number
    """
    if logger is None:
        logger = get_logger_safe()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(f"{outdir}/dynamic_spectra", exist_ok=True)
    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        succeed = 0
        failed = len(mslist)

    obsid = get_MWA_OBSID(mslist[0])
    ds_file_name = f"{obsid}_ds"
    plot_file = f"{outdir}/dynamic_spectra/{ds_file_name}.{extension}"
    if not overwrite and os.path.exists(plot_file):
        logger.info("Dynamic spectrum already exists.")
        return 0, plot_file, len(mslist), 0

    try:
        ###########################################
        tasks = []
        for msname in mslist:
            tasks.append(
                delayed(calc_dynamic_spectrum_wrapper)(
                    msname,
                    metafits,
                    f"{outdir}/dynamic_spectra",
                    n_threads=n_threads,
                )
            )
        result_wrapper = []
        logger.info("Start making dynamic spectra.")
        for i in range(0, len(tasks), njobs):
            batch = tasks[i : i + njobs]
            futures = dask_client.compute(batch)
            result_wrapper.extend(dask_client.gather(futures))
        result_wrapper = list(result_wrapper)
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

        ds_files = []
        for r in results:
            ds_files.append(r[0])

        succeed = len(ds_files)
        failed = len(mslist) - succeed
        logger.info(f"Total measurement sets: {len(mslist)}")
        logger.info(f"Total success: {succeed}")
        logger.info(f"Total failure: {failed}")
        logger.info(f"DS files: {[os.path.basename(i) for i in ds_files]}")

        ###########################################
        # Plotting dynamic spectrum
        ###########################################
        logger.info("Making dynamic spectra plots.")
        plot_file = make_ds_plot(
            ds_files,
            plot_file=plot_file,
            plot_quantity=plot_quantity,
            showgui=showgui,
        )
        goes_files = glob.glob(f"{outdir}/dynamic_spectra/sci*.nc")
        for f in goes_files:
            os.system(f"rm -rf {f}")
        return 0, plot_file, succeed, failed
    except Exception:
        logger.exception("Exception occured in making dynamic spectra.", exc_info=True)
        return 1, "", succeed, failed


def main(
    mslist,
    metafits,
    workdir,
    outdir,
    plot_quantity="TB",
    extension="png",
    overwrite=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid="0",
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Make dynamic spectra

    Parameters
    ----------
    mslist : str
        Measurement set list (comma separated)
    metafits : str
        Metafits file
    workdir : str
        Work directory
    outdir : str
        Output directory
    plot_quantity : str
        Plotting quantity (TB or flux)
    extension : str, optional
        Plot extension
    overwrite : bool, optional
        Overwrite existing plot
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction
    logfile : str, optional
        Log file
    jobid : str, optional
        Job ID
    verbose : bool, optional
        Verbose logs
    start_remote_log : bool, optional
        Start remote log
    dask_client: dask.client, optional
        Dask client

    Returns
    -------
    int
        Success messsage
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

    if outdir == "":
        outdir = workdir
    os.makedirs(outdir, exist_ok=True)
    logger.debug(f"Output directory: {outdir}.")

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
                "ds_plot", logfile, jobname=jobname, password=password
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
    logger.debug(f"Total coarse channels: {total_ncoarse}.")

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
            "Starting making dynamic spectra.", no_print=True
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
        msg, ds_plot_file, succeed, failed = make_solar_DS(
            mslist,
            dask_client,
            metafits,
            workdir,
            outdir,
            overwrite=overwrite,
            plot_quantity=plot_quantity,
            extension=extension,
            n_threads=n_threads,
            mem_limit=mem_limit,
            njobs=njobs,
            logger=logger,
        )
    except Exception:
        logger.exception("Exception occured in making dynamic spectra.", exc_info=True)
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
        description="Make MWA dynamic spectra of the Sun",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    # === Essential parameters ===
    essential = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    essential.add_argument(
        "mslist", type=str, help="Measurement set list (comma separated)"
    )
    essential.add_argument("metafits", type=str, help="Metafits file")
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
        help="Output directory",
    )

    # === Advanced parameters ===
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--plot_quantity",
        type=str,
        default="TB",
        help="Plot quantity (TB ot flux)",
    )
    adv_args.add_argument(
        "--extension",
        type=str,
        default="png",
        help="Save file extension",
    )
    adv_args.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plot or not",
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose logs")
    adv_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and PID tracking"
    )

    # === Advanced local system/ per node hardware resource parameters ===
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="Fraction of CPU usuage per node",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Fraction of memory usuage per node",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg, _, _ = main(
        args.mslist,
        args.metafits,
        args.workdir,
        args.outdir,
        plot_quantity=args.plot_quantity,
        extension=args.extension,
        overwrite=args.overwrite,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
