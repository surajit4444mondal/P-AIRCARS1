import logging
import psutil
import numpy as np
import argparse
import traceback
import time
import glob
import sys
import os
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def main(
    imagedir,
    outdir,
    workdir="",
    cpu_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
):
    """
    Run the flagging pipeline for a measurement set.

    Parameters
    ----------
    imagedir : str
        Image directory
    outdir : str
        Output directory
    workdir : str, optional
        Working directory
    cpu_frac : float, optional
        Fraction of total CPU resources to use. Default is 0.8.
    logfile : str or None, optional
        Path to the log file for saving logs. If None, logging to file is skipped.
    jobid : int, optional
        Numeric job ID used for PID tracking. Default is 0.
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials in the workdir. Default is False.

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    if workdir == "":
        workdir = f"{imagedir}/workdir"
    os.makedirs(workdir, exist_ok=True)

    os.makedirs(outdir, exist_ok=True)

    ############
    # Logger
    ############
    observer = None
    if (
        start_remote_log
        and os.path.exists(f"{workdir}/jobname_password.npy")
        and logfile is not None
    ):
        time.sleep(1)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "do_flagging", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")

    imagelist = glob.glob(f"{imagedir}/*.fits")
    if len(imagelist) == 0:
        print("No image in the image directory.")
        return 1

    try:
        ncpu = max(1, int(psutil.cpu_count() * cpu_frac))
        outimage_list = []
        for image in imagelist:
            outimage = make_mwa_overlay(
                image,
                plot_file_prefix=os.path.basename(image).split(".fits")[0]
                + "_euv_mwa_overlay",
                extensions=["png"],
                outdirs=[outdir],
                keep_euv_fits=True,
                ncpu=ncpu,
                verbose=False,
            )
        outimage_list.append(outimage)
        if len(outimage_list) == 0:
            print("No overlay is made.")
            msg = 1
        else:
            print(f"Total images: {len(imagelist)}")
            print(f"Total overlays: {len(outimage_list)}")
            os.system(f"rm -rf {imagedir}/images/aia.lev1_euv*.fits")
            os.system(f"rm -rf {imagedir}/images/*suvi-l2*.fits")
            msg = 0
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(1)
        drop_cache(imagedir)
        drop_cache(workdir)
        drop_cache(outdir)
        clean_shutdown(observer)
    return msg


def cli():
    usage = "Overlay MWA images on EUV images"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument("imagedir", type=str, help="Image directory")
    basic_args.add_argument("outdir", type=str, help="Output directory")
    basic_args.add_argument(
        "--workdir", type=str, default="", help="Name of work directory"
    )

    # Advanced switches
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU fraction to use"
    )
    hard_args.add_argument("--logfile", type=str, default=None, help="Log file")
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        args.imagedir,
        args.outdir,
        workdir=args.workdir,
        cpu_frac=args.cpu_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print("\n###################\nOverlay of images are done.\n###################\n")
    os._exit(result)
