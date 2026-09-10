import logging
import psutil
import numpy as np
import argparse
import traceback
import time
import glob
import sys
import os
import contextlib
from dask.distributed import as_completed
from paircars.utils.basic_utils import print_banner
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.mwa_ploting_utils import make_mwa_overlay, get_all_euv_maps
from paircars.utils.resource_utils import drop_cache
from paircars.utils.image_utils import filter_images
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def main(
    imagedir,
    outdir,
    wavelength=195,
    workdir="",
    all_overlay=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Make overlays of all images

    Parameters
    ----------
    imagedir : str
        Image directory
    outdir : str
        Overlay output directory
    workdir : str, optional
        Work directory
    wavelength : float, optional
        EUV wavelength
    all_overlay : bool, optional
        Make overlays of all images in image directory or not
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    logfile : str, optional
        Remote log file name
    jobid : str, optional
        Pipeline Job ID
    verbose : bool, optional
        Verbose logs
    start_remote_log : bool, optional
        Start logging to remote logger or not
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    int
        Total successful overlays
    int
        Total failed overlays
    """
    logger = get_logger_safe()
    if verbose:
        logger.setLevel(logging.DEBUG)

    cpu_frac = min(0.8, abs(cpu_frac))
    mem_frac = min(0.8, abs(mem_frac))

    if workdir == "":
        workdir = f"{imagedir}/workdir"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    logger.debug(f"Current working directory: {os.getcwd()}")

    if outdir == "":
        outdir = workdir
    os.makedirs(outdir, exist_ok=True)
    logger.debug(f"Output directory: {outdir}")

    ####################
    # Logger
    ####################
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
                "do_overlay", logfile, jobname=jobname, password=password
            )

    imagelist = glob.glob(f"{imagedir}/*.fits")

    if len(imagelist) == 0:
        logger.critical("No image in the image directory.")
        return 1, 0, 0

    succeed = 0
    failed = len(imagelist)

    ###############################################
    # Filter images
    ###############################################
    if not all_overlay:
        logger.debug("Filtering images for making overlays.")
        imagelist = filter_images(imagelist, min_time_sep=60.0)
        if len(imagelist) == 0:
            logger.critical("No filtered image is present in image directory.")
            return 1, 0, 0

    logger.info(f"Total images to overlay: {len(imagelist)}")

    try:
        nthreads = int(os.environ.get("OMP_NUM_THREADS", 1))
        ###########################
        # Download EUV maps
        ###########################
        euv_fits_images = get_all_euv_maps(
            imagelist,
            workdir,
            wavelength=wavelength,
            ncpu=nthreads,
        )
        if len(euv_fits_images) == 0:
            logger.critical("No EUV images downloaded.")
            return 1, succeed, failed
    except Exception:
        logger.exception("Error occured in EUV fits downloading.", exc_info=True)
        traceback.print_exc()
        return 1, succeed, failed

    ###############################
    # Dask cluster
    ###############################
    dask_cluster = None
    if dask_client is None:
        image_sizes = [os.stat(image).st_size / 1024**3 for image in imagelist]
        max_image_size = max(image_sizes)
        min_mem = max(1, round(5 * max_image_size, 2))
        dask_client, dask_cluster, dask_dir, njobs = get_local_dask_cluster(
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem=min_mem,
            max_worker=len(imagelist) + 1,
        )
        if dask_client is None:
            logger.critical("Error occured in creating local cluster.")
            return 1, succeed, failed
        scale_worker_and_wait(dask_cluster, dask_client, njobs)
        nthreads = int(psutil.cpu_count() * cpu_frac)
    else:
        ncpu = int(os.environ.get("OMP_NUM_THREADS", 1))
        client_info = dask_client.scheduler_info()["workers"]
        njobs = len(client_info)
        nthreads = ncpu * njobs

    try:
        ###########################
        # Overlay tasks
        ###########################
        for banner in print_banner("Starting overlays.", no_print=True).splitlines():
            logger.info(banner)
        results = []
        batch_size = max(1, njobs - 1)
        for i in range(0, len(imagelist), batch_size):
            batch_imgs = imagelist[i : i + batch_size]
            batch_euv = euv_fits_images[i : i + batch_size]
            futures = []
            for img, euv_fits in zip(batch_imgs, batch_euv):
                name_prefix = os.path.basename(img).replace(".fits", "")
                name_prefix = name_prefix.replace("IQUV", "I")
                futures.append(
                    dask_client.submit(
                        make_mwa_overlay,
                        img,
                        euv_fits,
                        workdir,
                        plot_mwa_colormap=False,
                        plot_file_prefix=name_prefix,
                        outdirs=[outdir],
                        verbose=True,
                        pure=False,
                        retries=2,
                    )
                )
            ###########################
            # Collect batch results
            ###########################
            for f in as_completed(futures):
                try:
                    r = f.result()
                    results.append(r)
                except Exception:
                    print("Overlay failed.")
                    traceback.print_exc()
            # free worker memory
            dask_client.cancel(futures)
            time.sleep(2)

        ###########################
        # Move outputs
        ###########################
        outimage_list = []
        for r in results:
            if r is not None:
                outimage_list.append(r[0])
                if os.path.dirname(os.path.abspath(r[0])) != os.path.abspath(outdir):
                    os.system(f"mv {r[0]} {outdir}")

        if len(outimage_list) == 0:
            logger.error("No overlay is made.")
            msg = 1
            succeed = 0
            failed = len(imagelist)
        else:
            logger.info(f"Total images: {len(imagelist)}")
            logger.info(f"Total overlays: {len(outimage_list)}")
            msg = 0
            succeed = len(outimage_list)
            failed = len(imagelist) - succeed
    except Exception:
        logger.exception("Exception occured in making overlays.", exc_info=True)
        msg = 1
    finally:
        os.system(f"rm -rf {imagedir}/*aia*.fits")
        os.system(f"rm -rf {imagedir}/*suvi*.fits")
        os.system(f"rm -rf {workdir}/aiamaps")
        time.sleep(5)
        drop_cache(imagedir)
        if observer is not None:
            clean_shutdown(observer)
        if dask_cluster is not None:
            with contextlib.suppress(Exception):
                dask_client.cancel(dask_client.futures)
            with contextlib.suppress(Exception):
                dask_client.close()
            with contextlib.suppress(Exception):
                dask_cluster.close()
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")
    return msg, succeed, failed


def cli():
    usage = "Overlay MWA images on EUV images"
    parser = argparse.ArgumentParser(
        description=usage,
        formatter_class=SmartDefaultsHelpFormatter,
    )
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument("imagedir", type=str, help="Image directory")
    basic_args.add_argument("outdir", type=str, help="Overlay output directory")
    basic_args.add_argument("--workdir", type=str, default="", help="Work directory")

    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--wavelength", type=float, default=195, help="EUV image wavelength in angstorm"
    )
    adv_args.add_argument(
        "--all_overlay", action="store_true", help="Make overlays of all images"
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose logs")
    adv_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    hard_args = parser.add_argument_group(
        "###################\nResource parameters\n###################"
    )
    hard_args.add_argument("--cpu_frac", type=float, default=0.8)
    hard_args.add_argument("--mem_frac", type=float, default=0.8)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg, _, _ = main(
        args.imagedir,
        args.outdir,
        workdir=args.workdir,
        wavelength=args.wavelength,
        all_overlay=args.all_overlay,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )

    return msg
