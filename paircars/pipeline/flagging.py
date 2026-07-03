import logging
import numpy as np
import argparse
import traceback
import time
import sys
import os
from dask import delayed
from astropy.io import fits
from paircars.utils.basic_utils import (
    suppress_output,
    print_banner,
    capture_all_output,
)
from paircars.utils.flagging import (
    flagsummary, 
    flag_badchan,
    do_flag_backup,
)
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.ms_metadata import check_datacolumn_valid
from paircars.utils.mwa_utils import get_bad_chans, get_mwa_bad_ants, get_ncoarse
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.solarflagger import flagger
from paircars.utils.resource_utils import drop_cache, limit_threads

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def single_ms_flag_wrapper(**kwargs):
    with capture_all_output() as (out, err):
        result = single_ms_flag(**kwargs)
        return kwargs.get("msname"), result, out.getvalue(), err.getvalue()


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
    run_solarflagger=False,
    normalize=False,
    threshold=5.0,
    force_flag=False,
    n_threads=1,
    mem_limit=1,
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
    run_solarflagger : bool, optional
        Run solar flagger or not
    normalize : bool, optional
        Use normalization in solar flagger
    threshold : float, optional
        Flagging threshold
    force_flag : bool, optional
        Force flag
    n_threads : int, optional
        Number of OpenMP threads
    mem_limit : float, optional
        Memory limit in GB

    Returns
    -------
    int
        Success message
    """
    n_threads = max(1, n_threads)
    mem_limit = abs(mem_limit)

    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    msname = msname.rstrip("/")
    if os.path.exists(f"{msname}/.flag_succeed") and not force_flag:
        print(f"Flagging of ms: {msname} is already done.")
        return 0
    os.system(f"rm -rf {msname}/.flag_*")
    print(f"Flagging ms: {msname}")
    try:
        ##############################
        # Flagging bad channels
        ##############################
        if badspw != "":
            try:
                flag_cmd = f"flag_badchan(\'{msname}\',\'{badspw}\')"
                print(flag_cmd)
                with suppress_output():
                    flag_badchan(msname,badspw)
            except Exception:
                traceback.print_exc()
                pass

        ##############################
        # Flagging bad antennas
        ##############################
        if bad_ants_str != "":
            try:
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='manual',"
                    f"antenna='{bad_ants_str}',"
                    f"cmdreason='badant',"
                    f"flagbackup=False)"
                )
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        antenna=bad_ants_str,
                        cmdreason="badant",
                        flagbackup=False,
                    )
            except Exception:
                traceback.print_exc()
                pass

        #################################
        # Flag quack timestamps
        #################################
        if flag_quack:
            try:
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='quack',"
                    f"quackmode='beg',"
                    f"quackinterval=4.0,"
                    f"flagbackup=False)"
                )
                print(flag_cmd)
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="quack",
                        quackmode="beg",
                        quackinterval=4.0,
                        flagbackup=False,
                    )
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='quack',"
                    f"quackmode='endb',"
                    f"quackinterval=4.0,"
                    f"flagbackup=False)"
                )
                print(flag_cmd)
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="quack",
                        quackmode="endb",
                        quackinterval=4.0,
                        flagbackup=False,
                    )
            except Exception:
                traceback.print_exc()
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
            if not modelcolumn_present and corcolumn_present:
                print(
                    "Residual column is requested, but model column is not present. Using corrected data instead"
                )
                datacolumn = "corrected"
            elif not modelcolumn_present and not corcolumn_present:
                print(
                    "Residual column is requested, but model and corrected columns are not present. Using data instead"
                )
                datacolumn = "data"
        elif datacolumn == "RESIDUAL_DATA":
            modelcolumn_present = check_datacolumn_valid(
                msname, datacolumn="MODEL_DATA"
            )
            datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
            if not modelcolumn_present:
                print(
                    "Residual column is requested, but model column is not present. Using data instead."
                )
                datacolumn = "data"

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
                datacolumn = "data"
            else:
                datacolumn = "corrected"

        #################################################
        # Whether data column is present or not
        #################################################
        if datacolumn == "data" or datacolumn == "DATA":
            datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
            if not datacolumn_present:
                print("Data column is chosen for flagging, but it is not present.")
                os.system(f"touch {msname}/.flag_failed")
                return 1
            else:
                datacolumn = "data"

        #################################
        # Clip zero amplitude data points
        #################################
        try:
            flag_cmd = (
                f"flagdata("
                f"vis='{msname}',"
                f"mode='clip',"
                f"clipzeros=True,"
                f"datacolumn='{datacolumn},"
                f"autocorr={flag_autocorr},"
                f"flagbackup=False)"
            )
            print(flag_cmd)
            with suppress_output():
                flagdata(
                    vis=msname,
                    mode="clip",
                    clipzeros=True,
                    datacolumn=datacolumn,
                    autocorr=flag_autocorr,
                    flagbackup=False,
                )
        except Exception:
            traceback.print_exc()
            pass

        #################################
        # Flag auto-correlations
        #################################
        if flag_autocorr:
            try:
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='manual',"
                    f"autocorr=True,"
                    f"datacolumn='{datacolumn}',"
                    f"flagbackup=False)"
                )
                print(flag_cmd)
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        autocorr=True,
                        datacolumn=datacolumn,
                        flagbackup=False,
                    )
            except Exception:
                traceback.print_exc()
                pass

        ##############
        # Tfcrop flag
        ##############
        if use_tfcrop:
            print("Using tfcrop flagging.")
            try:
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='tfcrop',"
                    f"timefit='line',"
                    f"freqfit='poly',"
                    f"extendflags=True,"
                    f"flagdimension='{flagdimension}',"
                    f"timecutoff={max(4.0, threshold)},"
                    f"freqcutoff={max(3.0, threshold)},"
                    f"growaround=False,"
                    f"action='apply',"
                    f"flagbackup=False,"
                    f"overwrite=True,"
                    f"writeflags=True,"
                    f"datacolumn='{datacolumn}')"
                )
                print(flag_cmd)
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="tfcrop",
                        timefit="line",
                        freqfit="poly",
                        extendflags=True,
                        flagdimension=flagdimension,
                        timecutoff=max(4.0, threshold),
                        freqcutoff=max(3.0, threshold),
                        growaround=False,
                        action="apply",
                        flagbackup=False,
                        overwrite=True,
                        writeflags=True,
                        datacolumn=datacolumn,
                    )
            except Exception:
                traceback.print_exc()
                pass

        #############
        # Rflag flag
        #############
        if use_rflag:
            print("Using rflag flagging.")
            try:
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='rflag',"
                    f"extendflags=True,"
                    f"timedevscale={max(5.0, threshold)},"
                    f"freqdevscale={max(5.0, threshold)},"
                    f"growaround=False,"
                    f"action='apply',"
                    f"flagbackup=False,"
                    f"overwrite=True,"
                    f"writeflags=True,"
                    f"datacolumn='{datacolumn}')"
                )
                print(flag_cmd)
                with suppress_output():
                    flagdata(
                        vis=msname,
                        mode="rflag",
                        extendflags=True,
                        timedevscale=max(5.0, threshold),
                        freqdevscale=max(5.0, threshold),
                        growaround=False,
                        action="apply",
                        flagbackup=False,
                        overwrite=True,
                        writeflags=True,
                        datacolumn=datacolumn,
                    )
            except Exception:
                traceback.print_exc()
                pass

        ##############
        # Extend flag
        ##############
        if use_tfcrop or use_rflag:
            try:
                flag_cmd = (
                    f"flagdata("
                    f"vis='{msname}',"
                    f"mode='extend',"
                    f"datacolumn='{datacolumn}',"
                    f"clipzeros=True,"
                    f"extendflags=True,"
                    f"extendpols=True,"
                    f"growtime=80.0,"
                    f"growfreq=80.0,"
                    f"growaround=False,"
                    f"flagneartime=False,"
                    f"flagnearfreq=False,"
                    f"action='apply',"
                    f"flagbackup=False,"
                    f"overwrite=True,"
                    f"writeflags=True)"
                )
                print(flag_cmd)
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
            except Exception:
                traceback.print_exc()
                pass

        ######################
        # Solar flagger
        ######################
        if run_solarflagger:
            print(f"Using solar flagger. Normalization used: {normalize}")
            do_flag_backup(msname, flagtype="solarflag")
            for th in range(10, int(threshold), 2):
                count = 0
                while count < 10:
                    result, n_final_flagged, n_additional_flagged = flagger(
                        msname,
                        datacolumn,
                        threshold=max(5.0, th),
                        normalize=normalize,
                        num_processes=n_threads,
                        flagbackup=False,
                    )
                    if n_additional_flagged == 0:
                        break
                    else:
                        count += 1
        os.system(f"touch {msname}/.flag_succeed")
        return 0
    except Exception:
        traceback.print_exc()
        os.system(f"touch {msname}/.flag_failed")
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
    run_solarflagger=False,
    normalize=False,
    threshold=5.0,
    restore_flag=False,
    force_flag=False,
    n_threads=1,
    mem_limit=1,
    logger=None,
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
    run_solarflagger : bool, optional
        Run solar flagger or not
    normalize : bool, optional
        Use normalization in solar flagger
    threshold : float, optional
        Flag threshold
    restore_flag : bool, optional
        Restore previous flags
    force_flag : bool, optional
        Force flag
    n_threads : int, optional
        CPU threads to use
    mem_limit : float, optional
        Memory to use in GB

    Returns
    -------
    int
        Success message
    int
        Succeeded ms number
    int
        Failed ms number
    """
    if logger is None:
        logger = get_logger_safe()

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        succeed = 0
        failed = len(mslist)

    try:
        limit_threads(n_threads=n_threads)
        from casatasks import flagdata

        header = fits.getheader(metafits)
        mode = header["MODE"]
        if "MWAX" in mode:
            flag_central_chan = False
        else:
            flag_central_chan = True
        logger.debug(f"Flag central channel: {flag_central_chan} for {mode}")

        tasks = []
        test_msname = os.path.abspath(mslist[0].rstrip("/"))
        if flag_bad_spw:
            badspw = get_bad_chans(test_msname, flag_central_chan=flag_central_chan)
            if badspw != "":
                logger.info(f"Bad spws: {badspw}.")
            else:
                logger.info("No bad spectral window.")
        else:
            badspw = ""
        if flag_bad_ants:
            bad_ants_str = get_mwa_bad_ants(metafits)
            if bad_ants_str != "":
                logger.info(f"Bad antennas: {bad_ants_str}.")
            else:
                logger.info("No bad antennas.")
        else:
            bad_ants_str = ""

        for msname in mslist:
            msname = os.path.abspath(msname.rstrip("/"))
            if restore_flag:
                logger.info(f"Restoring all previous flags for ms: {msname}")
                with suppress_output():
                    flagdata(vis=msname, mode="unflag", spw="0", flagbackup=False)
            if flag_backup:
                do_flag_backup(msname, flagtype="flagdata")
            tasks.append(
                delayed(single_ms_flag_wrapper)(
                    msname=msname,
                    badspw=badspw,
                    bad_ants_str=bad_ants_str,
                    datacolumn=datacolumn,
                    use_tfcrop=use_tfcrop,
                    use_rflag=use_rflag,
                    flagdimension=flagdimension,
                    flag_autocorr=flag_autocorr,
                    flag_quack=flag_quack,
                    threshold=threshold,
                    run_solarflagger=run_solarflagger,
                    normalize=normalize,
                    force_flag=force_flag,
                    n_threads=n_threads,
                    mem_limit=mem_limit,
                )
            )
        logger.info(f"Flagging mslist: {','.join(mslist)}")
        futures = dask_client.compute(tasks)
        result_wrapper = list(dask_client.gather(futures))
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

        for msname in mslist:
            ###############
            # Flag summary
            ###############
            summary_file = (
                f"{outdir}/{os.path.basename(msname).split('.ms')[0]}_basicflag.summary"
            )
            logger.info(f"Flag summary: {summary_file}")
            flagsummary(msname, summary_file)
        failed = sum(results)
        succeed = len(mslist) - failed
        logger.info(f"Total measurement set: {len(mslist)}")
        logger.info(f"Total success: {succeed}")
        logger.info(f"Total failure: {failed}")
        if len(mslist) == failed:
            return 1, succeed, failed
        else:
            return 0, succeed, failed
    except Exception:
        logger.exception("Exception occured during flagging.", exc_info=True)
        return 1, 0, len(mslist)


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
    run_solarflagger=False,
    normalize=False,
    threshold=5.0,
    restore_flag=False,
    force_flag=False,
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
    run_solarflagger : bool, optional
        Run solar flagger or not
    normalize : bool, optional
        Use normalization in solar flagger
    threshold : float, optional
        Flagging threshold
    restore_flag : bool, optional
        Restore previous flags
    force_flag : bool, optional
        Force flag
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
    flag_summary_dir = f"{outdir}/flag_summary"
    os.makedirs(flag_summary_dir, exist_ok=True)
    logger.debug(f"Flag summary directory: {flag_summary_dir}")

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
    if observer is None:
        logger.info(
            "Remote link or jobname is blank. Not transmiting to remote logger."
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
        for banner in print_banner("Starting flagging.", no_print=True).splitlines():
            logger.info(banner)
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

        msg, succeed, failed = do_flagging(
            mslist,
            metafits,
            dask_client,
            workdir,
            flag_summary_dir,
            datacolumn=datacolumn,
            flag_bad_ants=flag_bad_ants,
            flag_bad_spw=flag_bad_spw,
            use_tfcrop=use_tfcrop,
            use_rflag=use_rflag,
            flagdimension=flagdimension,
            flag_autocorr=flag_autocorr,
            flag_quack=flag_quack,
            run_solarflagger=run_solarflagger,
            normalize=normalize,
            threshold=threshold,
            restore_flag=restore_flag,
            force_flag=force_flag,
            flag_backup=flagbackup,
            n_threads=n_threads,
            mem_limit=mem_limit,
            logger=logger,
        )
    except Exception:
        logger.exception("Exception occured in flagging.", exc_info=True)
        msg = 1
    finally:
        time.sleep(5)
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
    usage = "Initial flagging"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist", type=str, help="Name of measurement sets (Comma separated)"
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
        "--run_solarflagger",
        dest="run_solarflagger",
        action="store_true",
        help="Run solar flagger or not",
    )
    adv_args.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Use normalization in solar flagger or not",
    )
    adv_args.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Flag threshold",
    )
    adv_args.add_argument(
        "--flagdimension", type=str, default="freqtime", help="Flag dimension"
    )
    adv_args.add_argument(
        "--restore",
        dest="restore_flag",
        action="store_true",
        help="Restore flags",
    )
    adv_args.add_argument(
        "--force_flag",
        dest="force_flag",
        action="store_true",
        help="Force flag",
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
        run_solarflagger=args.run_solarflagger,
        normalize=args.normalize,
        threshold=args.threshold,
        restore_flag=args.restore_flag,
        force_flag=args.force_flag,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        verbose=args.verbose,
    )
    return msg
