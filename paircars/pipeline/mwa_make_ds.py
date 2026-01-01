import logging
import dask
import numpy as np
import argparse
import traceback
import warnings
import time
import glob
import sys
import os
from dask import delayed
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


def make_solar_DS(
    mslist,
    metafits,
    workdir,
    outdir,
    plot_quantity="TB",
    extension="png",
    showgui=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Make solar dynamic spectrum and plots

    Parameters
    ----------
    mslist : list
        Measurement set list (Provide only same obsid measurement set list)
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
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    str
        Plot file name
    """
    if cpu_frac > 0.8:
        cpu_frac = 0.8
    total_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
    if mem_frac > 0.8:
        mem_frac = 0.8
    total_mem = (psutil.virtual_memory().available * mem_frac) / (1024**3)  # In GB

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(f"{outdir}/dynamic_spectra", exist_ok=True)
    print("##############################################")
    print(f"Start making dynamic spectra for ms: {msname}")
    print("##############################################")

    ########################################
    # Number of worker limit based on memory
    ########################################
    mem_limit = min(total_mem, max(mslist))
    njobs = min(len(mslist), int(total_mem / mem_limit))
    njobs = max(1, min(total_cpu, njobs))
    n_threads = max(1, int(total_cpu / njobs))

    print("#################################")
    print(f"Total dask worker: {njobs}")
    print(f"CPU per worker: {n_threads}")
    print(f"Memory per worker: {round(mem_limit,2)} GB")
    print("#################################")

    try:
        ###########################################
        tasks = []
        for msname in mslist:
            tasks.append(
                delayed(calc_dynamic_spectrum)(
                    msname, metafits, f"{outdir}/dynamic_spectra"
                )
            )
        results = []
        print("Start making dynamic spectra...")
        for i in range(0, len(tasks), njobs):
            batch = tasks[i : i + njobs]
            futures = dask_client.compute(batch)
            results.extend(dask_client.gather(futures))
        results = list(results)
        ds_files = []
        for r in results:
            ds_files.append(r[0])
        print(f"DS files: {ds_files}")

        ###########################################
        # Plotting dynamic spectrum
        ###########################################
        obsid = get_MWA_OBSID(mslist[0])
        ds_file_name = f"{obsid}_ds"
        plot_file = f"{outdir}/dynamic_spectra/{ds_file_name}.{extension}"
        plot_file = make_ds_plot(
            ds_files,
            plot_file=plot_file,
            plot_quantity=plot_quantity,
            showgui=showgui,
        )
        goes_files = glob.glob(f"{outdir}/dynamic_spectra/sci*.nc")
        for f in goes_files:
            os.system(f"rm -rf {f}")
        return plot_file
    except Exception as e:
        traceback.print_exc()
        return
    finally:
        time.sleep(5)
        for msname in mslist:
            drop_cache(msname)
        drop_cache(outdir)


def main(
    mslist,
    metafits,
    workdir,
    outdir,
    plot_quantity="TB",
    extension="png",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid="0",
    start_remote_log=False,
    dask_client=None,
):
    """
    Make dynamic spectra

    Parameters
    ----------
    mslist : str
        Measurement set list (comma seperated)
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
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction
    logfile : str, optional
        Log file
    jobid : str, optional
        Job ID
    start_remote_log : bool, optional
        Start remote log
    dask_client: dask.client, optional
        Dask client

    Returns
    -------
    int
        Success messsage
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    mslist = mslist.split(",")

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
    os.makedirs(workdir, exist_ok=True)
    if outdir == "":
        outdir = workdir
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
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "ds_plot", logfile, jobname=jobname, password=password
            )

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir = get_local_dask_cluster(
            2,
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        nworker = max(2, int(psutil.cpu_count() * cpu_frac))
        scale_worker_and_wait(dask_cluster, nworker)

    try:
        if len(mslist) > 0:
            ds_plot_file = make_solar_DS(
                mslist,
                metafits,
                outdir,
                plot_quantity=plot_quantity,
                extension=extension,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
            if ds_plot_file is not None:
                msg = 0
            else:
                msg = 1
        else:
            print("Please provide a valid measurement set list.")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for msname in mslist:
            drop_cache(msname)
        drop_cache(workdir)
        clean_shutdown(observer)
        if dask_cluster is not None:
            dask_client.close()
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
    return msg


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
        "mslist", type=str, help="Measurement set list (comma seperated)"
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
        "--start_remote_log", action="store_true", help="Start remote logging"
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
    hard_args.add_argument(
        "--logfile", type=str, default=None, help="Optional path to log file"
    )
    hard_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and PID tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        args.mslist,
        args.metafits,
        args.workdir,
        plot_quantity=args.plot_quantity,
        extension=args.extension,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nDynamic spectra are produced successfully.\n###################\n"
    )
    os._exit(result)
