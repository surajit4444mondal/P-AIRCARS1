import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import glob
import sys
import os
from casatools import msmetadata
from dask import delayed
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def single_ms_flag(
    msname="",
    badspw="",
    bad_ants_str="",
    datacolumn="data",
    use_tfcrop=True,
    use_rflag=False,
    flagdimension="freqtime",
    flag_autocorr=True,
    flag_quack=True,
    threshold=5.0,
    n_threads=-1,
    memory_limit=-1,
):
    """
    Flag on a single ms

    Parameters
    ----------
    msname : str
        Measurement set name
    badspw : str, optional
        Bad spectral window
    bad_ants_str : str, optional
        Bad antenna string
    datacolumn : str, optional
        Data column
    use_tfcrop : str, optional
        Use tfcrop or not
    use_rflag : str, optional
        Use rflag or not
    flagdimension : str, optional
        Flag dimension (only applicable for tfcrop)
    flag_autocorr : bool, optional
        Flag autocorrelations or not
    flag_quack : bool, optional
        Flag quack timestamps
    threshold : float, optional
        Flagging threshold
    n_threads : int, optional
        Number of OpenMP threads
    memory_limit : float, optional
        Memory limit in GB

    Returns
    -------
    int
        Success message
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    msname = msname.rstrip("/")
    try:
        ##############################
        # Flagging bad channels
        ##############################
        if badspw != "":
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        spw=badspw,
                        cmdreason="badchan",
                        flagbackup=False,
                    )
            except BaseException:
                pass

        ##############################
        # Flagging bad antennas
        ##############################
        if bad_ants_str != "":
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        antenna=bad_ants_str,
                        cmdreason="badant",
                        flagbackup=False,
                    )
            except BaseException:
                pass

        #################################
        # Flag quack timestamps
        #################################
        if flag_quack:
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="quack",
                        quackmode="beg",
                        quackinterval=4.0,
                        datacolumn=datacolumn,
                        flagbackup=False,
                    )
                    flagdata(
                        vis=msname,
                        mode="quack",
                        quackmode="endb",
                        quackinterval=4.0,
                        datacolumn=datacolumn,
                        flagbackup=False,
                    )
            except BaseException:
                pass

        #################################
        # Clip zero amplitude data points
        #################################
        try:
            with suppress_output():
                flagdata(
                    vis=msname,
                    mode="clip",
                    clipzeros=True,
                    datacolumn=datacolumn,
                    autocorr=flag_autocorr,
                    flagbackup=False,
                )
        except BaseException:
            pass

        #################################
        # Flag auto-correlations
        #################################
        if flag_autocorr:
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        autocorr=True,
                        datacolumn=datacolumn,
                        flagbackup=False,
                    )
            except BaseException:
                pass

        ####################################################
        # Check if required columns are present for residual
        ####################################################
        if datacolumn == "residual" or datacolumn == "RESIDUAL":
            modelcolumn_present = check_datacolumn_valid(
                msname, datacolumn="MODEL_DATA"
            )
            corcolumn_present = check_datacolumn_valid(
                msname, datacolumn="CORRECTED_DATA"
            )
            if modelcolumn_present == False or corcolumn_present == False:
                datacolumn = "corrected"
        elif datacolumn == "RESIDUAL_DATA":
            modelcolumn_present = check_datacolumn_valid(
                msname, datacolumn="MODEL_DATA"
            )
            datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
            if modelcolumn_present == False or datacolumn_present == False:
                datacolumn = "corrected"

        #################################################
        # Whether corrected data column is present or not
        #################################################
        if datacolumn == "corrected" or datacolumn == "CORRECTED_DATA":
            corcolumn_present = check_datacolumn_valid(
                msname, datacolumn="CORRECTED_DATA"
            )
            if not corcolumn_present:
                print(
                    "Corrected data column is chosen for flagging, but it is not present."
                )
                return 1
            else:
                datacolumn = "corrected"

        #################################################
        # Whether data column is present or not
        #################################################
        if datacolumn == "data" or datacolumn == "DATA":
            datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
            if not datacolumn_present:
                print("Data column is chosen for flagging, but it is not present.")
                return 1
            else:
                datacolumn = "data"

        ##############
        # Tfcrop flag
        ##############
        if use_tfcrop:
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="tfcrop",
                        timefit="line",
                        freqfit="poly",
                        extendflags=True,
                        flagdimension=flagdimension,
                        timecutoff=max(4.0, threshold - 1),
                        freqcutoff=max(3.0, threshold - 2),
                        growaround=False,
                        action="apply",
                        flagbackup=False,
                        overwrite=True,
                        writeflags=True,
                        datacolumn=datacolumn,
                    )
            except BaseException:
                pass

        #############
        # Rflag flag
        #############
        if use_rflag:
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="rflag",
                        extendflags=True,
                        timedevscale=threshold,
                        freqdevscale=threshold,
                        growaround=False,
                        action="apply",
                        flagbackup=False,
                        overwrite=True,
                        writeflags=True,
                        datacolumn=datacolumn,
                    )
            except BaseException:
                pass

        ##############
        # Extend flag
        ##############
        if use_tfcrop or use_rflag:
            try:
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="extend",
                        datacolumn=datacolumn,
                        clipzeros=True,
                        extendflags=True,
                        extendpols=True,
                        growtime=80.0,
                        growfreq=80.0,
                        growaround=False,
                        flagneartime=False,
                        flagnearfreq=False,
                        action="apply",
                        flagbackup=False,
                        overwrite=True,
                        writeflags=True,
                    )
            except BaseException:
                pass
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def do_flagging(
    mslist,
    metafits,
    dask_client,
    workdir,
    outdir,
    datacolumn="data",
    flag_bad_ants=True,
    flag_bad_spw=True,
    use_tfcrop=True,
    use_rflag=False,
    flagdimension="freqtime",
    flag_autocorr=True,
    flag_quack=True,
    flag_backup=True,
    restore_flag=True,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Function to perform initial flagging

    Parameters
    ----------
    mslist : list
        List of the ms
    metafits : str
        MWA metafits
    dask_client : dask.client
        Dask client
    workdir : str
        Work directory
    outdir : str
        Output directory
    datacolumn : str, optional
        Data column
    flag_bad_ants : bool, optional
        Flag bad antennas
    flag_bad_spw : bool, optional
        Flag bad channels
    use_tfcrop : bool, optional
        Use tfcrop or not
    use_rflag : bool, optional
        Use rflag or not
    flagdimension : str, optional
        Flag dimension (only for tfcrop)
    flag_autocorr : bool,optional
        Flag auto-correlations
    flag_quack : bool, optional
        Flag quack timestamps
    flag_backup : bool, optional
        Flag backup
    restore_flag : bool, optional
        Restore previous flags
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Success message
    """
    try:
        if cpu_frac > 0.8:
            cpu_frac = 0.8
        total_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
        if mem_frac > 0.8:
            mem_frac = 0.8
        total_mem = (psutil.virtual_memory().available * mem_frac) / (1024**3)  # In GB

        from casatasks import flagdata

        njobs = max(1, min(total_cpu, len(mslist)))
        n_threads = max(1, int(total_cpu / njobs))
        mem_limit = total_mem / njobs

        print("#################################")
        print(f"Total dask worker: {njobs}")
        print(f"CPU per worker: {n_threads}")
        print(f"Memory per worker: {round(mem_limit,2)} GB")
        print("#################################")

        ###########################################
        tasks = []
        for msname in mslist:
            msname = msname.rstrip("/")
            mspath = os.path.dirname(os.path.abspath(msname))
            os.chdir(mspath)
            print("###########################")
            print("Flagging measurement set : ", msname)
            print("###########################")
            correct_missing_col_subms(msname)
            if restore_flag:
                print("Restoring all previous flags...")
                with suppress_output():
                    flagdata(vis=msname, mode="unflag", spw="0", flagbackup=False)
            if flag_bad_spw:
                badspw = get_bad_chans(msname)
                print(f"Flagging bad spws: {badspw}.")
            else:
                badspw = ""
            if flag_bad_ants:
                bad_ants_str = get_mwa_bad_ants(metafits)
                print(f"Flagging bad antennas: {bad_ants_str}.")
            else:
                bad_ants_str = ""

            if flag_backup:
                do_flag_backup(msname, flagtype="flagdata")
            tasks.append(
                delayed(single_ms_flag)(
                    msname,
                    badspw=badspw,
                    bad_ants_str=bad_ants_str,
                    datacolumn=datacolumn,
                    use_tfcrop=use_tfcrop,
                    use_rflag=use_rflag,
                    flagdimension=flagdimension,
                    flag_autocorr=flag_autocorr,
                    flag_quack=flag_quack,
                    threshold=5.0,
                    n_threads=n_threads,
                    memory_limit=mem_limit,
                )
            )
        print(f"Flagging mslist: {','.join(mslist)}")
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        for msname in mslist:
            ###############
            # Flag summary
            ###############
            summary_file = (
                f"{outdir}/{os.path.basename(msname).split('.ms')[0]}_basicflag.summary"
            )
            print(f"Flag summary: {summary_file}")
            flagsummary(msname, summary_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def main(
    mslist,
    metafits,
    workdir="",
    outdir="",
    datacolumn="DATA",
    flag_bad_ants=True,
    flag_bad_spw=True,
    use_tfcrop=False,
    use_rflag=False,
    flag_autocorr=True,
    flag_quack=True,
    flagbackup=True,
    flagdimension="freqtime",
    restore_flag=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_client=None,
):
    """
    Run the flagging pipeline for a measurement set.

    Parameters
    ----------
    mslist : str
        Measurement set list (comma separated)
    metafits : str
        Metafits file
    workdir : str, optional
        Working directory to store logs and temporary files. If empty, defaults to
        `<msname>/workdir`. Default is "".
    outdir : str, optional
        Output directory. Default is: workdir
    datacolumn : str, optional
        Data column to be flagged (e.g., "DATA", "CORRECTED"). Default is "DATA".
    flag_bad_ants : bool, optional
        If True, flags known bad antennas using pre-defined heuristics. Default is True.
    flag_bad_spw : bool, optional
        If True, flags bad spectral windows based on statistics. Default is True.
    use_tfcrop : bool, optional
        If True, applies the `tfcrop` automated flagging algorithm. Default is False.
    use_rflag : bool, optional
        If True, applies the `rflag` automated flagging algorithm. Default is False.
    flag_autocorr : bool, optional
        If True, flags auto-correlations. Default is True.
    flag_quack : bool, optional
        If True, flag quack timestamps. Default is True.
    flagbackup : bool, optional
        If True, saves a flag backup before applying new flags. Default is True.
    flagdimension : str, optional
        Dimension over which to apply automated flagging (e.g., "freqtime"). Default is "freqtime".
    restore_flag : bool, optional
        Restore previous flags
    cpu_frac : float, optional
        Fraction of total CPU resources to use. Default is 0.8.
    mem_frac : float, optional
        Fraction of total memory resources to use. Default is 0.8.
    logfile : str or None, optional
        Path to the log file for saving logs. If None, logging to file is skipped.
    jobid : int, optional
        Numeric job ID used for PID tracking. Default is 0.
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials in the workdir. Default is False.
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

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir = get_local_dask_cluster(
            2,
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        nworker = min(len(mslist), int(psutil.cpu_count() * cpu_frac))
        scale_worker_and_wait(dask_cluster, nworker + 1)

    try:
        if len(mslist) > 0:
            msg = do_flagging(
                mslist,
                metafits,
                dask_client,
                workdir,
                outdir,
                datacolumn=datacolumn,
                flag_bad_ants=flag_bad_ants,
                flag_bad_spw=flag_bad_spw,
                use_tfcrop=use_tfcrop,
                use_rflag=use_rflag,
                flagdimension=flagdimension,
                flag_autocorr=flag_autocorr,
                flag_quack=flag_quack,
                restore_flag=restore_flag,
                flag_backup=flagbackup,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
        else:
            print("Please provide correct measurement set.")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(1)
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
    usage = "Initial flagging"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist", type=str, help="Name of measurement sets (Comma seperated)"
    )
    basic_args.add_argument("metafits", type=str, help="Metafits file")
    basic_args.add_argument(
        "--workdir", type=str, default="", help="Name of work directory"
    )
    basic_args.add_argument(
        "--outdir", type=str, default="", help="Name of output directory"
    )
    basic_args.add_argument(
        "--datacolumn", type=str, default="DATA", help="Name of the datacolumn"
    )

    # Advanced switches
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--no_flag_bad_ants",
        dest="flag_bad_ants",
        action="store_false",
        help="Do not flag bad antennas",
    )
    adv_args.add_argument(
        "--no_flag_bad_spw",
        dest="flag_bad_spw",
        action="store_false",
        help="Do not flag bad spectral windows",
    )
    adv_args.add_argument(
        "--use_tfcrop", action="store_true", help="Use tfcrop flagging"
    )
    adv_args.add_argument("--use_rflag", action="store_true", help="Use rflag flagging")
    adv_args.add_argument(
        "--no_flag_autocorr",
        dest="flag_autocorr",
        action="store_false",
        help="Do not flag auto-correlations",
    )
    adv_args.add_argument(
        "--no_flag_quack",
        dest="flag_quack",
        action="store_false",
        help="Do not flag quack timestamps",
    )
    adv_args.add_argument(
        "--no_flagbackup",
        dest="flagbackup",
        action="store_false",
        help="Do not backup flags",
    )
    adv_args.add_argument(
        "--no_restore",
        dest="restore_flag",
        action="store_false",
        help="Do not restore flags",
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--flagdimension", type=str, default="freqtime", help="Flag dimension"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU fraction to use"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory fraction to use"
    )
    hard_args.add_argument("--logfile", type=str, default=None, help="Log file")
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        args.mslist,
        args.metafits,
        workdir=args.workdir,
        outdir=args.outdir,
        datacolumn=args.datacolumn,
        flag_bad_ants=args.flag_bad_ants,
        flag_bad_spw=args.flag_bad_spw,
        use_tfcrop=args.use_tfcrop,
        use_rflag=args.use_rflag,
        flag_autocorr=args.flag_autocorr,
        flag_quack=args.flag_quack,
        flagbackup=args.flagbackup,
        flagdimension=args.flagdimension,
        restore_flag=args.restore_flag,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print("\n###################\nFlagging is done.\n###################\n")
    os._exit(result)
