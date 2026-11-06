import logging
import numpy as np
import argparse
import warnings
import time
import glob
import sys
import os
from astropy.wcs import FITSFixedWarning
from paircars.utils.basic_utils import (
    print_banner,
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
from paircars.utils.resource_utils import drop_cache
from paircars.utils.mwa_ploting_utils import (
    make_gif_movie,
    plot_in_hpc,
    plot_in_hpc_full_stokes,
    save_in_hpc,
)
from paircars.utils.image_utils import get_image_npol, generate_tb_map

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
warnings.simplefilter("ignore", FITSFixedWarning)


def make_imaging_data_products(
    imagedir, save_hpc=True, make_plots=True, make_TB=True, logger=None
):
    """
    Make final imaging data products

    Parameters
    ----------
    imagedir : str
        Image directory
    save_hpc : bool, optional
        Save fits in helioprojective coordinates
    make_plots : bool, optional
        Make image plots in PNGs
    make_TB : bool, optional
        Make brightness temperature maps

    Returns
    -------
    int
        Success message
    int
        Total HPC fits files
    int
        Total PNG files
    int
        Total GIF files
    int
        Total brightness temperature fits files
    int
        Total TB HPC fits files
    int
        Total TB PNG files
    int
        Total TB GIF files
    """
    hpc_counts = 0
    png_counts = 0
    gif_counts = 0
    tb_counts = 0
    tb_hpc_counts = 0
    tb_png_counts = 0
    tb_gif_counts = 0
    if logger is None:
        logger = get_logger_safe()
    try:
        ############################################
        # Saving fits in helioprojective coordinates
        ############################################
        if save_hpc:
            hpcdir = f"{os.path.dirname(imagedir)}/final_hpcs"
            pbcor_images = glob.glob(f"{imagedir}/*.fits")
            os.makedirs(hpcdir, exist_ok=True)
            logger.info("Saving images in helioprojective coordinates.")
            for image in pbcor_images:
                save_in_hpc(image, outdir=hpcdir)
            hpc_counts = len(glob.glob(f"{hpcdir}/*.fits"))
            logger.debug(f"Total images saved in HPC coordinates: {hpc_counts}.")

        ################
        # Make png plots
        ################
        if make_plots:
            logger.info("Making plots of images in helioprojective coordinates.")
            pngdir = f"{os.path.dirname(imagedir)}/final_pngs"
            os.makedirs(pngdir, exist_ok=True)
            for image in pbcor_images:
                try:
                    npol = get_image_npol(image)
                    if npol == 4:
                        plot_in_hpc_full_stokes(
                            image,
                            draw_limb=True,
                            extensions=["png"],
                            outdirs=[pngdir],
                        )
                    else:
                        plot_in_hpc(
                            image,
                            draw_limb=True,
                            extensions=["png"],
                            outdirs=[pngdir],
                        )
                except BaseException:
                    junkpng = (
                        f"{pngdir}/{os.path.basename(image).split('.fits')[0]}.png.junk"
                    )
                    os.system(f"touch {junkpng}")
            png_counts = len(glob.glob(f"{pngdir}/*.png"))
            logger.debug(f"Total PNG images made: {png_counts}.")

            #########################
            # Making GIF
            #########################
            all_pngs = glob.glob(f"{pngdir}/*.png")
            png_freqs = np.unique(
                [float(a.split("freq_")[-1].split("_")[0]) for a in all_pngs]
            )
            gifdir = f"{os.path.dirname(imagedir)}/final_gifs"
            os.makedirs(gifdir, exist_ok=True)
            logger.info("Making GIFs per frequencies for all images.")
            for png_freq in png_freqs:
                sub_list = sorted(glob.glob(f"{pngdir}/*freq_{png_freq}*.png"))
                outfile = f"{gifdir}/freq_{png_freq}.gif"
                gif_file = make_gif_movie(sub_list, outfile)
                if os.path.exists(gif_file):
                    logger.debug(f"GIF for frequency: {png_freq} is {gif_file}.")
                else:
                    logger.warning(f"GIF for frequency: {png_freq} is failed.")
            gif_counts = len(glob.glob(f"{gifdir}/*.gif"))
            logger.debug(f"Total GIF images made: {gif_counts}.")

        ####################################
        # Making brightness temperature maps
        ####################################
        if make_TB:
            tb_dir = f"{os.path.dirname(imagedir)}/tb_images"
            os.makedirs(tb_dir, exist_ok=True)
            logger.info("Making brightness temperature maps.")
            for pbcor_image in pbcor_images:
                tb_image = (
                    tb_dir
                    + "/"
                    + os.path.basename(pbcor_image).split(".fits")[0]
                    + "_TB.fits"
                )
                generate_tb_map(pbcor_image, outfile=tb_image)
            tb_counts = len(glob.glob(f"{tb_dir}/*.fits"))
            logger.debug(f"Total brightness temperature maps made: {tb_counts}.")

            ############################################
            # Saving fits in helioprojective coordinates
            ############################################
            if save_hpc:
                hpcdir = f"{os.path.dirname(tb_dir)}/tb_hpcs"
                tb_images = glob.glob(f"{tb_dir}/*.fits")
                os.makedirs(hpcdir, exist_ok=True)
                logger.info(
                    "Saving brightness temperature maps helioprojective coordinates."
                )
                for image in tb_images:
                    save_in_hpc(image, outdir=hpcdir)
                tb_hpc_counts = len(glob.glob(f"{hpcdir}/*.fits"))
                logger.debug(f"Total TB maps in HPC coordinates saved: {tb_hpc_counts}.")

            if make_plots:
                logger.info("Making plots of brightness temperature maps.")
                pngdir = f"{os.path.dirname(tb_dir)}/tb_pngs"
                os.makedirs(pngdir, exist_ok=True)
                for image in tb_images:
                    try:
                        npol = get_image_npol(image)
                        if npol == 4:
                            plot_in_hpc_full_stokes(
                                image,
                                draw_limb=True,
                                extensions=["png"],
                                outdirs=[pngdir],
                            )
                        else:
                            plot_in_hpc(
                                image,
                                draw_limb=True,
                                extensions=["png"],
                                outdirs=[pngdir],
                            )
                    except BaseException:
                        junkpng = f"{pngdir}/{os.path.basename(image).split('.fits')[0]}.png.junk"
                        os.system(f"touch {junkpng}")
                tb_png_counts = len(glob.glob(f"{pngdir}/*.png"))
                logger.debug(f"Total PNG images of TB maps made: {tb_png_counts}.")

                #########################
                # Making GIF
                #########################
                all_pngs = glob.glob(f"{pngdir}/*.png")
                png_freqs = np.unique(
                    [float(a.split("freq_")[-1].split("_")[0]) for a in all_pngs]
                )
                gifdir = f"{os.path.dirname(tb_dir)}/tb_gifs"
                os.makedirs(gifdir, exist_ok=True)
                logger.info(
                    "Making GIFs per frequencies for brightness temperature maps."
                )
                for png_freq in png_freqs:
                    sub_list = sorted(glob.glob(f"{pngdir}/*freq_{png_freq}*.png"))
                    outfile = f"{gifdir}/freq_{png_freq}.gif"
                    gif_file = make_gif_movie(sub_list, outfile)
                    if os.path.exists(gif_file):
                        logger.debug(f"GIF for frequency: {png_freq} is {gif_file}.")
                    else:
                        logger.warning(f"GIF for frequency: {png_freq} is failed.")
                tb_gif_counts = len(glob.glob(f"{gifdir}/*.gif"))
                logger.debug(f"Total GIF images of TB maps made: {tb_gif_counts}.")
        return (
            0,
            hpc_counts,
            png_counts,
            gif_counts,
            tb_counts,
            tb_hpc_counts,
            tb_png_counts,
            tb_gif_counts,
        )
    except Exception:
        logger.exception(
            "Error occured in producing imaging data products.", exc_info=True
        )
        return (
            1,
            hpc_counts,
            png_counts,
            gif_counts,
            tb_counts,
            tb_hpc_counts,
            tb_png_counts,
            tb_gif_counts,
        )


def main(
    imagedir,
    workdir="",
    save_hpc=True,
    make_plots=True,
    make_TB=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Make final imaging data products

    Parameters
    ----------
    imagedir : str
        Image directory
    workdir : str, optinal
        Working directory
    save_hpc : bool, optional
        Save fits in helioprojective coordinates
    make_plots : bool, optional
        Make image plots in PNGs
    make_TB : bool, optional
        Make brightness temperature maps
    cpu_frac : float,optional
        CPU fraction
    mem_frac : float
        Memory fraction
    logfile : str, optional
        Log file
    jobid : str, optional
        Job ID
    verbose : bool, optional
        Verbose logs
    start_remote_log : bool, optional
        Start remote logger
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    int
        Succeeded image number
    int
        Failed image number
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
                "all_imaging_products", logfile, jobname=jobname, password=password
            )

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        if dask_client is None:
            logger.critical("Error occured in creating local cluster.")
            return 1
        scale_worker_and_wait(dask_cluster, dask_client, nworker)

    try:
        for banner in print_banner(
            "Start making final imaging data products.", no_print=True
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

        if os.path.exists(imagedir):
            (
                msg,
                hpc_counts,
                png_counts,
                gif_counts,
                tb_counts,
                tb_hpc_counts,
                tb_png_counts,
                tb_gif_counts,
            ) = make_imaging_data_products(
                imagedir, save_hpc=True, make_plots=True, make_TB=False, logger=None
            )
        else:
            logger.critical("Please provide correct image directory path.")
            msg = 1
    except Exception:
        logger.exception(
            "Exception occured in making final imaging data products.", exc_info=True
        )
        msg = 1
    finally:
        time.sleep(5)
        if observer is not None:
            clean_shutdown(observer)
        if dask_cluster is not None:
            dask_client.shutdown()
            dask_client.close()
            dask_cluster.close()
            drop_cache(imagedir)
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Make final imaging data products",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument("imagedir", help="Path to image directory")
    basic_args.add_argument("--workdir", default="", help="Path to work directory")

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--no_save_hpc",
        action="store_false",
        dest="save_hpc",
        help="Do not save HPC fits",
    )
    adv_args.add_argument(
        "--no_make_plot",
        action="store_false",
        dest="make_plots",
        help="Do not make PNG and GIF plots",
    )
    adv_args.add_argument(
        "--no_make_TB",
        action="store_false",
        dest="make_TB",
        help="Do not make brightness temperature maps",
    )
    adv_args.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs",
    )
    adv_args.add_argument(
        "--jobid", type=int, default=0, help="Job ID for logging and PID tracking"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU usage fraction"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory usage fraction"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        args.imagedir,
        workdir=args.workdir,
        save_hpc=args.save_hpc,
        make_plots=args.make_plots,
        make_TB=args.make_TB,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
