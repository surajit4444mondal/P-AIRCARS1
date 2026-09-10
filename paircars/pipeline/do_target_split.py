import logging
import numpy as np
import argparse
import time
import sys
import os
from casatools import msmetadata
from dask import delayed
from astropy.io import fits
from paircars.utils.basic_utils import print_banner, capture_all_output
from paircars.utils.casatasks import single_mstransform
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.ms_metadata import get_timeranges, get_total_time
from paircars.utils.mwa_utils import (
    get_MWA_coarse_bands,
    get_MWA_coarse_chan,
)
from paircars.utils.flagging import flag_badchan
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

logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


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


def get_timerange_str(timerange):
    timerange_list = timerange.split(",")
    s = timerange_list[0].split("~")[0]
    e = timerange_list[-1].split("~")[-1]
    s_string = "".join(s.split(".")[0].split("/")[:-1]) + "".join(
        s.split(".")[0].split("/")[-1].split(":")
    )
    e_string = "".join(e.split(".")[0].split("/")[:-1]) + "".join(
        e.split(".")[0].split("/")[-1].split(":")
    )
    return f"{s_string}_{e_string}"


def single_mstransform_wrapper(**kwargs):
    with capture_all_output() as (out, err):
        result = single_mstransform(**kwargs)
        return result, out.getvalue(), err.getvalue()


def move_to_sun_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = move_to_sun(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def split_target_scans(
    mslist,
    metafits,
    dask_client,
    workdir,
    timeres,
    freqres,
    datacolumn,
    split_coarse_chans=[],
    scan=1,
    prefix="targets",
    time_interval=-1,
    time_window=-1,
    quack_timestamps=-1,
    max_time_chunk=-1,
    single_chan_split=False,
    force_split=False,
    move_solarcenter=False,
    n_threads=-1,
    logger=None,
):
    """
    Split target scans

    Parameters
    ----------
    mslist : list
        Measurement set list
    metafits : str
        Metafits file
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
    split_coarse_chans : list, optional
        Split coarse channels
    scan : int, optional
        Scan to split
    prefix : str, optional
        Splited ms prefix
    time_interval : float
        Time interval in seconds
    time_window : float
        Time window in seconds
    quack_timestamps : int, optional
        Number of timestamps ignored at the start and end of each scan
    max_time_chunk : float, optional
        Maximum time chunk of each spliting ms in seconds (default: -1, full time chunk)
    single_chan_split: bool, optional
        Split only a single good channel
    force_split : bool, optional
        Force split
    move_solarcenter : bool, optional
        Move phasecenter to solar center after spliting
    n_threads : int, optional
        Number of threads to use

    Returns
    -------
    int
        Success message
    list
        Splited ms list
    """
    if logger is None:
        logger = get_logger_safe()
    n_threads = max(1, n_threads)
    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, []
    try:
        os.chdir(workdir)
        logger.debug(f"Current working directory: {os.getcwd()}")
        #######################################
        # Extracting time frequency information
        #######################################
        header = fits.getheader(metafits)
        obsid = header["GPSTIME"]
        mode = header["MODE"]
        meta_chanres = header["FINECHAN"]
        if "MWAX" in mode:
            flag_central_chan = False
        else:
            flag_central_chan = True

        tasks = []
        splited_ms_list = []

        for msname in mslist:
            msmd = msmetadata()
            msmd.open(msname)
            chanres_MHz = msmd.chanres(0, unit="MHz")[0]
            if flag_central_chan:
                chanres_kHz = msmd.chanres(0, unit="kHz")[0]
                if chanres_kHz > meta_chanres:
                    flag_central_chan = False
            times = msmd.timesforspws(0)
            diff = np.diff(times)
            ms_timeres = abs(np.nanmax(diff))
            msmd.close()
            if freqres > 0:  # Image resolution is in MHz
                chanwidth = int(freqres / chanres_MHz)
                if chanwidth < 1:
                    chanwidth = 1
            else:
                chanwidth = 1
            if timeres > 0:  # Image resolution is in seconds
                timebin = str(max(ms_timeres, timeres)) + "s"
            else:
                timebin = ""

            #############################
            # Making spectral chunks
            #############################
            coarse_channel_bands = get_MWA_coarse_bands(msname)
            coarse_chans = get_MWA_coarse_chan(msname)
            if len(split_coarse_chans) == 0:
                use_coarse_chans = coarse_chans
            else:
                use_coarse_chans = []
                for coarse_chan in coarse_chans:
                    if coarse_chan in split_coarse_chans:
                        use_coarse_chans.append(coarse_chan)
            if len(use_coarse_chans) > 0:
                logger.debug("##################################")
                logger.debug(
                    f"Using coarse channels for {msname} are: {use_coarse_chans}"
                )
                logger.debug("##################################")
                coarse_chlist = []
                good_spwlist = []
                for c in range(len(coarse_channel_bands)):
                    coarse_chan = coarse_chans[c]
                    if coarse_chan in use_coarse_chans:
                        chan = coarse_channel_bands[c]
                        start_chan = chan[0]
                        end_chan = chan[1]
                        good_chan_list = chan[2]
                        if single_chan_split:
                            good_spwlist.append(f"0:{min(good_chan_list)}")
                        else:
                            good_spwlist.append(
                                f"0:{min(good_chan_list)}~{max(good_chan_list)}"
                            )
                            if flag_central_chan:
                                central_chan = int((start_chan + end_chan) / 2)
                                logger.debug(f"Flag central channel: {central_chan}.")
                                logger.debug(
                                    f"flag_badchan('{msname}', spw='0:{central_chan}')"
                                )
                                flag_badchan(msname, spw=f"0:{central_chan}")
                        coarse_chlist.append(f"{coarse_chan}")

                ##################################
                # Making temporal chunks
                ##################################
                all_timerange_list = get_timeranges(
                    msname,
                    time_interval,
                    time_window,
                    quack_timestamps=quack_timestamps,
                    max_time_chunk=max_time_chunk,
                )
                for timerange_list in all_timerange_list:
                    timerange = ",".join(timerange_list)
                    for i in range(len(coarse_chlist)):
                        good_spw = good_spwlist[i]
                        coarse_chan = coarse_chlist[i]
                        t_range = get_timerange_str(timerange)
                        outputvis = f"{workdir}/{prefix}_{obsid}_ch_{coarse_chan}_t_{t_range}.ms"
                        if (
                            os.path.exists(f"{outputvis}/.splited")
                            and force_split is False
                        ):
                            logger.info(f"{outputvis} is already splited successfully.")
                            splited_ms_list.append(outputvis)
                        else:
                            if os.path.exists(outputvis):
                                logger.debug(
                                    f"Deleteing pre-existing output ms: {outputvis}"
                                )
                                os.system(f"rm -rf {outputvis}")
                            if os.path.exists(f"{outputvis}.flagversions"):
                                logger.debug(
                                    f"Deleteing pre-existing output ms flags: {outputvis}.flagversions"
                                )
                                os.system(f"rm -rf {outputvis}.flagversions")
                            logger.debug("Spliting parameters:")
                            logger.debug(
                                f"Channel width: {chanwidth}, timebin: {timebin}, datacolumn: {datacolumn}, spectral window: {good_spw}, time range: {timerange}"
                            )
                            tasks.append(
                                delayed(single_mstransform_wrapper)(
                                    msname=msname,
                                    outputms=outputvis,
                                    width=chanwidth,
                                    timebin=timebin,
                                    datacolumn=datacolumn,
                                    spw=good_spw,
                                    corr="",
                                    timerange=timerange,
                                    n_threads=n_threads,
                                )
                            )
        if len(tasks) == 0:
            logger.error("No task to split measurement sets.")
            return 1, []
        future = dask_client.compute(tasks)
        result_wrapper = dask_client.gather(future)
        result = []
        for r in result_wrapper:
            result.append(r[0][1])
            logger.debug("================")
            logger.debug(f"Worker log for: {os.path.basename(r[0][1])}")
            logger.debug("================")
            for line in r[1].splitlines():
                logger.debug(line)
            for line in r[2].splitlines():
                logger.debug(line)
        splited_ms_list = splited_ms_list + result
        if len(splited_ms_list) == 0:
            logger.error(f"Spliting of measurement set: {msname} is unsuccessful.")
            return 1, []
        else:
            logger.info(f"Spliting of measurement set: {msname} is done successfully.")
            if move_solarcenter:
                try:
                    tasks = [
                        delayed(move_to_sun_wrapper)(msname, ncpu=n_threads)
                        for msname in splited_ms_list
                    ]
                    result_wrapper = list(
                        dask_client.gather(dask_client.compute(tasks))
                    )
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
                    logger.info(
                        "Moving phasecenter to solarcenter has done successfully."
                    )
                    logger.info(f"Total success: {succeed}")
                    logger.info(f"Total failure: {failed}")
                except Exception:
                    logger.warning("Moving phasecenter to solarcenter has failed.")
                finally:
                    for splited_ms in splited_ms_list:
                        drop_cache(splited_ms)
            return 0, splited_ms_list
    except Exception:
        logger.exception(
            f"Spliting of measurement set: {msname} is unsuccessful.", exc_info=True
        )
        return 1, []


def main(
    mslist,
    metafits,
    workdir="",
    datacolumn="data",
    split_coarse_chans=[],
    scan=1,
    time_window=-1,
    time_interval=-1,
    quack_timestamps=-1,
    max_time_chunk=-1,
    freqres=-1,
    timeres=-1,
    prefix="targets",
    single_chan_split=False,
    force_split=False,
    move_solarcenter=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    verbose=False,
    dask_client=None,
):
    """
    Split target scans from a measurement set into smaller chunks for parallel processing.

    Parameters
    ----------
    mslist : str
        Measurement sets (comma separated).
    metafits : str
        Metafits file
    workdir : str, optional
        Working directory for intermediate and output products. If empty, defaults to `<msname>/workdir`.
    datacolumn : str, optional
        Column of the MS to use for splitting (e.g., "DATA", "CORRECTED"). Default is "data".
    split_coarse_chans : list, optional
        Split coarse channels
    scan : int, optional
        Scan numbers to split.
    time_window : float, optional
        Time window in seconds for a single time chunk. Set -1 to disable. Default is -1.
    time_interval : float, optional
        Time interval in seconds between two time chunks. Set -1 to disable. Default is -1.
    quack_timestamps : int, optional
       Number of timestamps to flag at the beginning and end of each scan ("quack"). -1 to disable. Default is -1.
    max_time_chunk : float, optional
        Maximum time chunk of each spliting ms in seconds (default: -1, full time chunk)
    freqres : float, optional
        Frequency resolution in MHz for spectral averaging. Set -1 to disable. Default is -1.
    timeres : float, optional
        Time resolution in seconds for time averaging. Set -1 to disable. Default is -1.
    prefix : str, optional
        Prefix for the output split MS files. Default is "targets".
    single_chan_split : bool, optional
        Split only a songle good channel.
    force_split : bool, optional
        Force to split
    move_solarcenter : bool, optional
        Move phasecenter to solar center after spliting
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
    verbose : bool, optional
        Verbose logs
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    int
        Expected splited ms
    int
        Succeeded splited ms
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
                "do_target_split", logfile, jobname=jobname, password=password
            )

    ###########################
    # WSClean container
    ###########################
    if move_solarcenter:
        container_name = "paircarswsclean"
        container_present = check_udocker_container(container_name)
        if not container_present:
            logger.debug(f"Initializing {container_name}.")
            container_name = initialize_wsclean_container(
                name=container_name, verbose=True
            )
            if container_name is None:
                logger.critical(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1, 0, 0

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        total_ncoarse = 0
        for msname in mslist:
            ms_coarse_chans = get_MWA_coarse_chan(msname)
            if len(split_coarse_chans) > 0:
                ms_coarse_chans = list(set(ms_coarse_chans) & set(split_coarse_chans))
            ncoarse = len(ms_coarse_chans)
            if max_time_chunk > 0:
                total_time = get_total_time(msname)
                t_chunks = int(total_time / max_time_chunk) + 1
                total_ncoarse += ncoarse * t_chunks
            else:
                total_ncoarse += ncoarse
        total_ncoarse = max(1, total_ncoarse)
        logger.debug(f"Total usable coarse channels: {total_ncoarse}")
        expected = total_ncoarse
        succeed = 0

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir, nworker = get_local_dask_cluster(
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            max_worker=total_ncoarse + 1,
        )
        if dask_client is None:
            logger.critical("Error occured in creating local cluster.")
            return 1, expected, succeed
        scale_worker_and_wait(dask_cluster, dask_client, nworker)

    try:
        for banner in print_banner(
            "Starting spliting measurement sets.", no_print=True
        ).splitlines():
            logger.info(banner)
        ##################################
        # Parallel spliting
        ##################################
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

        msg, splited_mslist = split_target_scans(
            mslist,
            metafits,
            dask_client,
            workdir,
            float(timeres),
            float(freqres),
            datacolumn,
            split_coarse_chans=split_coarse_chans,
            time_window=float(time_window),
            time_interval=float(time_interval),
            quack_timestamps=int(quack_timestamps),
            max_time_chunk=max_time_chunk,
            single_chan_split=single_chan_split,
            force_split=force_split,
            move_solarcenter=move_solarcenter,
            scan=scan,
            prefix=prefix,
            n_threads=n_threads,
            logger=logger,
        )
        succeed = len(splited_mslist)
        logger.info(f"Total measurement sets: {len(mslist)}")
        logger.info(f"Total expected splited ms: {total_ncoarse}")
        logger.info(f"Total splited ms: {succeed}")
        if len(splited_mslist) == 0:
            logger.debug("No splited measurement sets.")
            msg = 1
        else:
            logger.debug("List of splited measurement sets:")
            logger.debug(f"{splited_mslist}")
            msg = 0
        return msg, expected, succeed
    except Exception:
        logger.exception("Exception occured in spliting.", exc_info=True)
        msg = 1
        return msg, expected, succeed
    finally:
        if observer is not None:
            clean_shutdown(observer)
        if len(mslist) > 0:
            for msname in mslist:
                if os.path.exists(msname):
                    drop_cache(msname)
        if dask_cluster is not None:
            dask_client.shutdown()
            dask_client.close()
            dask_cluster.close()
            drop_cache(workdir)
            os.system(f"rm -rf {dask_dir}")


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
        "metafits",
        type=str,
        help="Metafits file (required positional argument)",
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
        "--max_time_chunk",
        type=float,
        default=-1,
        help="Maximum time chunk of each spliting ms in seconds (default: full time chunk)",
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
    adv_args.add_argument("--force_split", action="store_true", help="Force to split")
    adv_args.add_argument(
        "--move_solarcenter",
        action="store_true",
        help="Move phasecenter to solar center",
    )
    adv_args.add_argument(
        "--single_chan_split", action="store_true", help="Single channel to split"
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
        args.mslist,
        args.metafits,
        workdir=args.workdir,
        datacolumn=args.datacolumn,
        scan=args.scan,
        time_window=args.time_window,
        time_interval=args.time_interval,
        quack_timestamps=args.quack_timestamps,
        max_time_chunk=args.max_time_chunk,
        single_chan_split=args.single_chan_split,
        force_split=args.force_split,
        move_solarcenter=args.move_solarcenter,
        freqres=args.freqres,
        timeres=args.timeres,
        prefix=args.prefix,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
