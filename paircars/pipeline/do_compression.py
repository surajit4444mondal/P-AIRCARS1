import os
import glob
import logging
import time
import sys
import numpy as np
import argparse
from paircars.utils.image_utils import compress_fits, decompress_fits
from dask import delayed
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
from paircars.utils.resource_utils import drop_cache

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def main(
    imagedir,
    workdir,
    outputdir="",
    compress=True,
    keep_original=True,
    keep_compressed=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Function to run compression and decompression of all fits images in a directory

    Parameters
    ----------
    imagedir : str
        Image directory
    workdir : str
        Work directory
    outputdir : str
        Output directory
    compress : bool, optional
        Do compression or decompression
    keep_original : bool, optional
        Keep original fits images (only used for compression mode)
    keep_compressed : bool, optional
        Keep compressed fits images (only used for de-compression mode)
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
        Total succeeded images
    int
        Total failed images
    """
    logger = get_logger_safe()
    if verbose:
        logger.setLevel(logging.DEBUG)

    cpu_frac = min(0.8, abs(cpu_frac))
    mem_frac = min(0.8, abs(mem_frac))

    if workdir == "":
        workdir = imagedir + "/workdir"
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
                "all_selfcal", logfile, jobname=jobname, password=password
            )

    if not os.path.exists(imagedir):
        logger.error(f"{imagedir} does not exist.")
        return 1, 0, 0

    imagelist = glob.glob(f"{imagedir}/*.fits")
    if len(imagelist) == 0:
        logger.error(f"No image is present in: {imagedir}")
        return 1, 0, 0

    ##########################################
    # Creating local dask cluster if needed
    ##########################################
    largest_file = max(imagelist, key=os.path.getsize)
    largest_file_size_gb = os.path.getsize(largest_file) / 1024**3

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem=2 * largest_file_size_gb,
            max_worker=len(imagelist) + 1,
        )
        if dask_client is None:
            logger.critical("Error occured in creating local cluster.\n")
            return 1, 0, 0
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

    logger.info("#################################")
    logger.info(f"Total dask worker: {njobs}")
    logger.info(f"Memory per worker: {mem_limit} GB")
    logger.info("#################################\n")

    os.makedirs(f"{workdir}/logs", exist_ok=True)
    if outputdir != "":
        os.makedirs(f"{outputdir}", exist_ok=True)
    else:
        outputdir = imagedir

    ##############################################
    total_failed = 0
    total_succeed = 0
    try:
        if compress:
            tasks = [
                delayed(compress_fits)(
                    imagename, outputdir=outputdir, keep_original=keep_original
                )
                for imagename in imagelist
            ]
            logger.info(f"Starting compressing all images in: {imagedir}.\n")
            results = list(dask_client.gather(dask_client.compute(tasks)))
            logger.info(f"Compressed images saved in: {outputdir}.\n")
        else:
            tasks = [
                delayed(decompress_fits)(
                    imagename, outputdir=outputdir, keep_compressed=keep_compressed
                )
                for imagename in imagelist
            ]
            logger.info(f"Starting de-compressing all images in: {imagedir}.\n")
            results = list(dask_client.gather(dask_client.compute(tasks)))
            logger.info(f"De-compressed images saved in: {outputdir}.\n")
        for r in results:
            msg = r[0]
            if msg == 0:
                total_succeed += 1
            else:
                total_failed += 1
        return 0, total_succeed, total_failed
    except Exception:
        logger.exception(
            "Exception occured in compression/de-compression.", exc_info=True
        )
        return 1, 0, 0
    finally:
        time.sleep(5)
        clean_shutdown(observer)
        if dask_cluster is not None:
            dask_client.close()
            dask_cluster.close()
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")


def cli():
    parser = argparse.ArgumentParser(
        description="Compress/de-compress fits images",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "imagedir",
        type=str,
        help="Image directory name",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory",
    )
    basic_args.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory",
    )
    basic_args.add_argument(
        "--compress",
        action="store_true",
        help="Compress images (if not provided, de-compress images)",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--no_keep_original",
        action="store_false",
        dest="keep_original",
        help="Do not keep original fits files after compression",
    )
    adv_args.add_argument(
        "--no_keep_compressed",
        action="store_false",
        dest="keep_compressed",
        help="Do not keep compressed fits files after de-compression",
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
        args.imagedir,
        args.workdir,
        args.outdir,
        compress=args.compress,
        keep_original=args.keep_original,
        keep_compressed=args.keep_compressed,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
