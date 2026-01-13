import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import sys
import os
from casatools import msmetadata
from dask import delayed
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


def chanlist_to_str(lst):
    lst = sorted(lst)
    ranges = []
    start = lst[0]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1] + 1:
            if lst[i - 1] > start:
                ranges.append(f"{start}~{lst[i - 1]}")
            elif lst[i - 1] == start:
                ranges.append(f"{start}")
            start = lst[i]
    if lst[-1] > start:
        ranges.append(f"{start}~{lst[-1]}")
    elif lst[-1] == start:
        ranges.append(f"{start}")
    return ";".join(ranges)


def split_target_scans(
    msname,
    dask_client,
    workdir,
    timeres,
    freqres,
    datacolumn,
    scan=1,
    prefix="targets",
    time_interval=-1,
    time_window=-1,
    quack_timestamps=-1,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Split target scans

    Parameters
    ----------
    msname : str
        Measurement set
    dask_client : dask.client
        Dask client
    workdir : str
        Work directory
    timeres : float
        Time resolution in seconds
    freqres : float
        Frequency resolution in MHz
    datacolumn : str
        Data column to split
    scan : int
        Scan to split
    prefix : str, optional
        Splited ms prefix
    time_interval : float
        Time interval in seconds
    time_window : float
        Time window in seconds
    quack_timestamps : int, optional
        Number of timestamps ignored at the start and end of each scan
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    list
        Splited ms list
    """
    try:
        if cpu_frac > 0.8:
            cpu_frac = 0.8
        total_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
        if mem_frac > 0.8:
            mem_frac = 0.8
        total_mem = (psutil.virtual_memory().available * mem_frac) / (1024**3)  # In GB

        os.chdir(workdir)
        #######################################
        # Extracting time frequency information
        #######################################
        msmd = msmetadata()
        msmd.open(msname)
        chanres = msmd.chanres(0, unit="MHz")[0]
        freqs = msmd.chanfreqs(0, unit="MHz")
        bw = max(freqs) - min(freqs)
        nchan = msmd.nchan(0)
        msmd.close()
        if freqres > 0:  # Image resolution is in MHz
            chanwidth = int(freqres / chanres)
            if chanwidth < 1:
                chanwidth = 1
        else:
            chanwidth = 1
        if timeres > 0:  # Image resolution is in seconds
            timebin = str(timeres) + "s"
        else:
            timebin = ""

        #############################
        # Making spectral chunks
        #############################
        coarse_channel_bands = get_MWA_coarse_bands(msname)
        chanlist = []
        for chan in coarse_channel_bands:
            chanlist.append(f"{chan[0]}~{chan[1]-1}")

        ##################################
        # Parallel spliting
        ##################################
        if len(chanlist) > 0:
            total_chunks = len(chanlist)
        else:
            total_chunks = 1

        njobs = max(1, min(total_cpu, total_chunks))
        n_threads = max(1, int(total_cpu / njobs))

        tasks = []
        splited_ms_list = []
        timerange_list = get_timeranges(
            msname,
            time_interval,
            time_window,
            quack_timestamps=quack_timestamps,
        )
        timerange = ",".join(timerange_list)
        for chanrange in chanlist:
            outputvis = f"{workdir}/{prefix}_{os.path.basename(msname).split('.ms')[0]}_spw_{chanrange}.ms"
            if os.path.exists(f"{outputvis}/.splited"):
                print(f"{outputvis} is already splited successfully.")
                splited_ms_list.append(outputvis)
            else:
                task = delayed(single_mstransform)(
                    msname=msname,
                    outputms=outputvis,
                    width=chanwidth,
                    timebin=timebin,
                    datacolumn=datacolumn,
                    spw="0:" + chanrange,
                    corr="",
                    timerange=timerange,
                    n_threads=n_threads,
                )
                tasks.append(task)
        if len(tasks):
            futures = dask_client.compute(tasks)
            results = list(dask_client.gather(futures))
            for splited_ms in results:
                splited_ms_list.append(splited_ms)
        print("##################")
        print("Spliting of target scans are done successfully.")
        print("##################")
        return 0, splited_ms_list
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print("Spliting of target scans are unsuccessful.")
        print("##################")
        return 1, []
    finally:
        time.sleep(1)
        drop_cache(msname)


def main(
    mslist,
    workdir="",
    datacolumn="data",
    scan=1,
    time_window=-1,
    time_interval=-1,
    quack_timestamps=-1,
    freqres=-1,
    timeres=-1,
    prefix="targets",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_client=None,
):
    """
    Split target scans from a measurement set into smaller chunks for parallel processing.

    Parameters
    ----------
    mslist : str
        Measurement sets (comma separated).
    workdir : str, optional
        Working directory for intermediate and output products. If empty, defaults to `<msname>/workdir`.
    datacolumn : str, optional
        Column of the MS to use for splitting (e.g., "DATA", "CORRECTED"). Default is "data".
    scan : int, optional
        Scan numbers to split.
    time_window : float, optional
        Time window in seconds for a single time chunk. Set -1 to disable. Default is -1.
    time_interval : float, optional
        Time interval in seconds between two time chunks. Set -1 to disable. Default is -1.
    quack_timestamps : int, optional
       Number of timestamps to flag at the beginning and end of each scan ("quack"). -1 to disable. Default is -1.
    freqres : float, optional
        Frequency resolution in MHz for spectral averaging. Set -1 to disable. Default is -1.
    timeres : float, optional
        Time resolution in seconds for time averaging. Set -1 to disable. Default is -1.
    prefix : str, optional
        Prefix for the output split MS files. Default is "targets".
    cpu_frac : float, optional
        Fraction of available CPUs to allocate per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of available memory to allocate per task. Default is 0.8.
    logfile : str or None, optional
        Path to log file. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Job identifier for tracking and PID storage. Default is 0.
    start_remote_log : bool, optional
        If True, enables remote logging using credentials stored in workdir. Default is False.
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    mslist = mslist.split(",")

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

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
                "do_target_split", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")

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
            print("###################################")
            print(f"Start spliting measurement sets in coarse frequency bands.")
            print("###################################")
            for msname in mslist:
                msg, final_target_mslist = split_target_scans(
                    msname,
                    dask_client,
                    workdir,
                    float(timeres),
                    float(freqres),
                    datacolumn,
                    time_window=float(time_window),
                    time_interval=float(time_interval),
                    quack_timestamps=int(quack_timestamps),
                    scan=scan,
                    prefix=prefix,
                    cpu_frac=float(cpu_frac),
                    mem_frac=float(mem_frac),
                )
        else:
            print("Please provide correct measurement set list.")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
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
        description="Split measurement set into coarse channels",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Name of measurement sets (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        help="Name of work directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--datacolumn",
        type=str,
        default="data",
        help="Data column to split",
    )
    adv_args.add_argument(
        "--scan",
        type=int,
        default=1,
        help="Target scan to split",
    )
    adv_args.add_argument(
        "--time_window",
        type=float,
        default=-1,
        help="Time window in seconds of a single time chunk",
    )
    adv_args.add_argument(
        "--time_interval",
        type=float,
        default=-1,
        help="Time interval in seconds between two time chunks",
    )
    adv_args.add_argument(
        "--quack_timestamps",
        type=int,
        default=-1,
        help="Time stamps to ignore at the start and end of the each scan",
    )
    adv_args.add_argument(
        "--freqres",
        type=float,
        default=-1,
        help="Frequency to average in MHz",
        metavar="Float",
    )
    adv_args.add_argument(
        "--timeres",
        type=float,
        default=-1,
        help="Time bin to average in seconds",
        metavar="Float",
    )
    adv_args.add_argument(
        "--prefix",
        type=str,
        default="targets",
        help="Splited ms prefix name",
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

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
    hard_args.add_argument("--logfile", type=str, default=None, help="Log file")
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        mslist=args.mslist,
        workdir=args.workdir,
        datacolumn=args.datacolumn,
        scan=args.scan,
        time_window=args.time_window,
        time_interval=args.time_interval,
        quack_timestamps=args.quack_timestamps,
        freqres=args.freqres,
        timeres=args.timeres,
        prefix=args.prefix,
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
        "\n###################\nSpliting measurement set into coarse channels are done.\n###################\n"
    )
    os._exit(result)
