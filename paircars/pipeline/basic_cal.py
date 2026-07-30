import logging
import numpy as np
import argparse
import time
import sys
import os
import glob
import traceback
from casatools import msmetadata
from dask import delayed
from astropy.io import fits
from paircars.utils.basic_utils import (
    suppress_output,
    print_banner,
    capture_all_output,
)
from paircars.utils.calibration import (
    get_caltable_metadata,
)
from paircars.utils.crossphasecal import crossphasecal
from paircars.utils.flagging import (
    flagsummary,
    do_flag_backup,
    get_unflagged_antennas,
    get_chans_flag,
)
from paircars.utils.logger_utils import (
    SmartDefaultsHelpFormatter,
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.ms_metadata import get_uvrange_exclude
from paircars.utils.mwa_utils import freq_to_MWA_coarse, get_ncoarse, get_gleam_uvrange
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
)
from paircars.utils.resource_utils import drop_cache, limit_threads
from paircars.pipeline.flagging import single_ms_flag

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def single_ms_cal_and_flag_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = single_ms_cal_and_flag(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def filtered_final_caltables(caltables, workdir):
    """
    Filter last round bandpass and crossphase caltables

    Parameters
    ----------
    caltables : list
        Caltable list
    workdir : str
        Work directory

    Returns
    -------
    list
        Bandpass list
    list
        Crossphase list
    """
    groups = {}
    for f in caltables:
        name = os.path.basename(f)
        if name.endswith(".bcal"):
            ext = "bcal"
        elif name.endswith(".kcrosscal"):
            ext = "kcrosscal"
        else:
            continue
        prefix = name.split("_round_")[0]
        round_part = name.split("_round_")[1]
        round_num = int(round_part.split(".")[0])
        key = (prefix, ext)
        if key not in groups or round_num > groups[key][0]:
            groups[key] = (round_num, f)
    bcals = []
    kcrosscals = []
    for (prefix, ext), (_, fname) in groups.items():
        if ext == "bcal":
            bcals.append(fname)
        else:
            kcrosscals.append(fname)
    final_bcals = []
    final_kcrosscals = []
    if len(bcals) > 0:
        for bcal in bcals:
            out_bcal = f"{workdir}/{os.path.basename(bcal).split('_round')[0]}.bcal"
            os.system(f"cp -r {bcal} {out_bcal}")
            final_bcals.append(out_bcal)
    if len(kcrosscals) > 0:
        for kcrosscal in kcrosscals:
            out_kcrosscal = (
                f"{workdir}/{os.path.basename(kcrosscal).split('_round')[0]}.kcrosscal"
            )
            os.system(f"cp -r {kcrosscal} {out_kcrosscal}")
            final_kcrosscals.append(out_kcrosscal)
    print(f"Filtered bandpass tables: {final_bcals}")
    print(f"Filtered crosshand phase tables: {final_kcrosscals}")
    return final_bcals, final_kcrosscals


def run_bandpass(
    msname,
    workdir,
    uvrange="",
    refant="",
    solint="inf",
    combine="",
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=1,
):
    """
    Perform bandpass calibration
    """
    n_threads = max(1, n_threads)
    with limit_threads(n_threads=n_threads):
        from casatasks import bandpass

    caltable_prefix = f"{workdir}/{os.path.basename(msname).split('.ms')[0]}"
    bpass_cmd = (
        f"bandpass("
        f"vis='{msname}', "
        f"caltable='{caltable_prefix}.bcal', "
        f"uvrange='{uvrange}', "
        f"refant='{refant}', "
        f"solint='{solint}', "
        f"combine='{combine}', "
        f"gaintable={gaintable}, "
        f"gainfield={gainfield}, "
        f"interp={interp}"
        f")"
    )
    print(f"{bpass_cmd}")
    with suppress_output():
        bandpass(
            vis=msname,
            caltable=f"{caltable_prefix}.bcal",
            uvrange=uvrange,
            refant=refant,
            solint=solint,
            combine=combine,
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
        )
    if os.path.exists(caltable_prefix + ".bcal"):
        print(f"Bandpass table: {caltable_prefix}.bcal")
        return caltable_prefix + ".bcal"
    else:
        print("Bandpass table is not made.")
        return


def run_crossphasecal(
    msname,
    workdir,
    uvrange="",
    gaintable=[],
    n_threads=1,
):
    """
    Perform crosshand phase calibration
    """
    n_threads = max(1, n_threads)
    caltable_prefix = f"{workdir}/{os.path.basename(msname).split('.ms')[0]}"
    kcross_cmd = (
        f"crossphasecal("
        f"'{msname}', "
        f"'{caltable_prefix}.kcrosscal', "
        f"uvrange='{uvrange}', "
        f"gaintable={gaintable}, "
        f"n_threads={n_threads}"
        f")"
    )
    print(f"{kcross_cmd}")
    with suppress_output():
        if len(gaintable) != 0:
            gaintable = gaintable[0]
        else:
            gaintable = ""
        crossphasecal(
            msname,
            f"{caltable_prefix}.kcrosscal",
            uvrange=uvrange,
            gaintable=gaintable,
            n_threads=n_threads,
        )
    if os.path.exists(caltable_prefix + ".kcrosscal"):
        print(f"Crosshand phase table: {caltable_prefix}.kcrosscal")
        return caltable_prefix + ".kcrosscal"
    else:
        print("Crosshand phase table is not made.")
        return


def run_applycal(
    msname,
    applymode="",
    flagbackup=True,
    gaintable=[],
    gainfield=[],
    interp=[],
    calwt=[],
    n_threads=1,
):
    """
    Perform apply calibration
    """
    n_threads = max(1, n_threads)
    with limit_threads(n_threads=n_threads):
        from casatasks import applycal

    applycal_cmd = (
        f"applycal("
        f"vis='{msname}', "
        f"gaintable={gaintable},"
        f"gainfield={gainfield}, "
        f"interp={interp}, "
        f"calwt={calwt}, "
        f"flagbackup={flagbackup}"
        f")"
    )
    print(f"{applycal_cmd}")
    with suppress_output():
        applycal(
            vis=msname,
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
            calwt=calwt,
            applymode=applymode,
            flagbackup=flagbackup,
        )
    return


def run_postcal_flag(
    msname="",
    datacolumn="residual",
    threshold=5.0,
    n_threads=1,
    mem_limit=1,
):
    """
    Perform apply calibration
    """
    n_threads = max(1, n_threads)
    mem_limit = abs(mem_limit)
    flag_cmd = (
        f"single_ms_flag("
        f"msname='{msname}',"
        f"badspw='',"
        f"bad_ants_str='',"
        f"datacolumn='{datacolumn}',"
        "use_tfcrop=True,"
        "use_rflag=True,"
        "flagdimension='freqtime',"
        "flag_autocorr=False,"
        f"threshold={threshold},"
        f"n_threads={n_threads},"
        f"mem_limit={mem_limit})"
    )
    print(flag_cmd)
    msg = single_ms_flag(
        msname=msname,
        badspw="",
        bad_ants_str="",
        datacolumn=datacolumn,
        use_tfcrop=True,
        use_rflag=True,
        flagdimension="freqtime",
        flag_autocorr=False,
        threshold=threshold,
        n_threads=n_threads,
        mem_limit=mem_limit,
    )
    if msg > 0:
        print(f"Issue in post-calibration flagging in ms: {msname}")
    return


def single_ms_cal_and_flag(
    msname,
    workdir,
    cal_round,
    refant,
    uvrange,
    do_polcal=True,
    applysol=True,
    do_postcal_flag=True,
    flag_threshold=5.0,
    n_threads=1,
    mem_limit=1,
):
    """
    Single ms calibration and post-calibration flagging

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory
    cal_round : int
        Calibration round number
    refant : str
        Reference antenna
    uvrange :str
        UV-range
    do_polcal : bool, optional
        Perform polarisation calibration
    applysol : bool, optional
        Apply solutions for post-calibration flagging
    do_postcal_flag : bool, optional
        Peform post-calibration flagging
    flag_threshold : float, optional
        Flag threshold
    n_threads : int, optional
        Number of OpenMP threads
    mem_limit : float, optional
        Memory limit in GB

    Returns
    -------
    str
        Caltables
    bool
        Whether postcal flagging is successful or not
    """
    n_threads = max(1, n_threads)
    mem_limit = abs(mem_limit)
    with limit_threads(n_threads=n_threads):
        from casatasks import flagmanager

    succeed_postcal_flag = True
    try:
        caltable_prefix = (
            f"{workdir}/{os.path.basename(msname).split('.ms')[0]}_caltable"
        )
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        ######################################
        # Removing previous rounds caltables
        ######################################
        bpass_caltable = caltable_prefix + ".bcal"
        crossphase_caltable = caltable_prefix + ".kcrosscal"

        if os.path.exists(bpass_caltable):
            os.system("rm -rf " + bpass_caltable)
        if os.path.exists(crossphase_caltable):
            os.system("rm -rf " + crossphase_caltable)

        #######################################
        # Calibration on calibrator fields
        #######################################
        print_banner(f"Calibrating calibrator field ms: {msname}")
        applycal_gaintable = []
        applycal_gainfield = []
        applycal_interp = []

        ##############################
        # Bandpass calibration
        ##############################
        print(f"Performing bandpass calibrations on: {msname}")
        bpass_caltable = run_bandpass(
            msname,
            workdir,
            uvrange=uvrange,
            refant=refant,
            solint="inf",
            n_threads=n_threads,
        )
        if bpass_caltable is not None and os.path.exists(bpass_caltable):
            applycal_gaintable.append(bpass_caltable)
            applycal_gainfield.append("")
            applycal_interp.append("linear,linear")
        else:
            print(f"Bandpass calibration is not successful for ms: {msname}.")
            return [], False

        ##############################
        # Crossphase calibration
        ##############################
        if do_polcal:
            if npol != 4:
                print(
                    f"Measurement set: {msname} is not full-polar. Not performing crosshand phase calibration."
                )
            else:
                print(f"Performing crosshand phase calibrations on: {msname}")
                crossphase_caltable = run_crossphasecal(
                    msname,
                    workdir,
                    uvrange=uvrange,
                    gaintable=applycal_gaintable,
                    n_threads=n_threads,
                )
                if crossphase_caltable is not None and os.path.exists(
                    crossphase_caltable
                ):
                    applycal_gaintable.append(crossphase_caltable)
                    applycal_gainfield.append("")
                    applycal_interp.append("linear,linear")

        ##############################
        # Apply calibration
        ##############################
        if applysol:
            print(f"Applying calibrations on: {msname} from {applycal_gaintable}.")
            run_applycal(
                msname,
                flagbackup=False,
                gaintable=applycal_gaintable,
                gainfield=applycal_gainfield,
                interp=applycal_interp,
                calwt=[False] * len(applycal_gainfield),
                n_threads=n_threads,
            )

            ##############################
            # Post calibration flagging
            ##############################
            if do_postcal_flag:
                do_flag_backup(msname, flagtype="postcal")
                print(
                    f"Performing post-calibration flagging - MS: {msname}, threshold: {flag_threshold}"
                )
                run_postcal_flag(
                    msname,
                    datacolumn="residual",
                    threshold=flag_threshold,
                    n_threads=n_threads,
                    mem_limit=mem_limit,
                )
                unflag_chans, flag_chans = get_chans_flag(msname, n_threads=n_threads)
                if len(flag_chans) / (len(unflag_chans) + len(flag_chans)) > 0.5:
                    print(
                        "Restoring flags because of large number of channels flagged."
                    )
                    with suppress_output():
                        flagmanager(vis=msname, mode="restore", versionname="postcal_1")
                    succeed_postcal_flag = False
                with suppress_output():
                    flagmanager(vis=msname, mode="delete", versionname="postcal_1")

        ###############################
        # Finished calibration round
        ###############################
        bpass_caltable = (
            bpass_caltable
            if (bpass_caltable is not None and os.path.exists(bpass_caltable))
            else None
        )
        crossphase_caltable = (
            crossphase_caltable
            if (crossphase_caltable is not None and os.path.exists(crossphase_caltable))
            else None
        )
        return [
            bpass_caltable,
            crossphase_caltable,
        ], succeed_postcal_flag
    except Exception:
        print(f"Calibration round {cal_round} failed.")
        traceback.print_exc()
        return [], False


def single_round_cal_and_flag(
    mslist,
    dask_client,
    workdir,
    cal_round,
    refant=1,
    uvrange="",
    do_polcal=True,
    applysol=True,
    do_postcal_flag=True,
    flag_threshold=5.0,
    n_threads=1,
    mem_limit=1,
    logger=None,
):
    """
    Single round calibration and flagging for a set of measurement sets in parallel

    Parameters
    ----------
    mslist : list
        Measurement set list
    dask_client : dask.client
        Dask client
    workdir : str
        Working directory
    cal_round : int
        Calibration round
    refant : str, optional
        Reference antenna
    uvrange : str, optional
        UV-range
    do_polcal : bool, optional
        Perform polarisation calibration
    applysol : bool, optional
        Apply solutions
    do_postcal_flag : bool or list, optional
        Perform post-calibration flagging for each measurement set
    flag_threashold : float, optional
        Flagging threshold
    n_threads : int, optional
        CPU threads to use
    mem_limit : float, optional
        Memory to use in GB

    Returns
    -------
    dict
        A python dictionary cotaining measurement set name and its caltables
    int
        Succeeded ms number
    int
        Failed ms number
    list
        List whether postcal flag is successful or not
    """
    if logger is None:
        logger = get_logger_safe()

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return {}, 0, 0, []
    else:
        succeed = 0
        failed = len(mslist)

    if isinstance(do_postcal_flag, bool):
        do_postcal_flag = [do_postcal_flag] * len(mslist)
    elif len(do_postcal_flag) < len(mslist):
        do_postcal_flag = [do_postcal_flag[0]] * len(mslist)

    tasks = []
    for i in range(len(mslist)):
        msname = mslist[i]
        postcal_flag = do_postcal_flag[i]
        tasks.append(
            delayed(single_ms_cal_and_flag_wrapper)(
                msname,
                workdir,
                cal_round,
                refant,
                uvrange,
                do_polcal=do_polcal,
                applysol=applysol,
                do_postcal_flag=postcal_flag,
                flag_threshold=flag_threshold,
                n_threads=n_threads,
                mem_limit=mem_limit,
            )
        )
    result_wrapper = list(dask_client.gather(dask_client.compute(tasks)))
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

    caltable_dic = {}
    postcal_flags = []
    succeed = 0
    failed = 0
    for i in range(len(mslist)):
        msname = mslist[i]
        caltables = results[i][0]
        postcal_flag = results[i][1]
        caltables_clean = [x for x in caltables if x is not None]
        if len(caltables_clean) == 0:
            logger.error(f"Basic calibration is not succssful for ms : {msname}")
            failed += 1
        else:
            succeed += 1
        caltable_dic[msname] = caltables_clean
        postcal_flags.append(postcal_flag)
    return caltable_dic, succeed, failed, postcal_flags


def run_basic_cal_rounds(
    mslist,
    dask_client,
    workdir,
    outdir="",
    refant="",
    uvrange="",
    keep_backup=False,
    perform_polcal=False,
    n_threads=1,
    mem_limit=1,
    logger=None,
):
    """
    Perform basic calibration rounds

    Parameters
    ----------
    mslist : str
        List of measurement sets
    dask_client : dask.client
        Dask client
    workdir : str
        Warking directory
    outdir : str
        Output directory
    refant : str, optional
        Reference antenna
    uvrange : str, optional
        UV-range
    perform_polcal : bool, optional
        Perform polarization calibration for fullpolar data
    keep_backup : bool, optional
        Keep backup of ms after each calibration rounds
    n_threads : int, optional
        CPU threads to use
    mem_limit : float, optional
        Memory to use in GB

    Returns
    -------
    int
        Success message
    list
        Bandpass caltables
    list
        Crossphase caltables
    int
        Succeeded ms number
    int
        Failed ms number
    """
    if logger is None:
        logger = get_logger_safe()

    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, [], [], 0, 0
    else:
        succeed = 0
        failed = len(mslist)

    try:
        from casatasks import flagdata

        trial_ms = mslist[0]
        msmd = msmetadata()
        msmd.open(trial_ms)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        if npol == 4:
            n_rounds = 3
        else:
            n_rounds = 2
            perform_polcal = False
        logger.info(f"Calibration for ms list: {mslist}.")
        logger.info(f"Total calibration rounds: {n_rounds}")

        #################
        # Initial values
        #################
        do_polcal = False
        do_postcal_flag = [True] * len(mslist)
        applysol = True
        flag_threshold = 6.0
        if refant == "":
            logger.debug(
                "Reference antenna is not specified. Determining reference antenna"
            )
            unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(trial_ms)
            refant = unflagged_antenna_names[0]
            msmd = msmetadata()
            msmd.open(trial_ms)
            refant = str(msmd.antennaids(refant)[0])
            msmd.close()
        logger.debug(f"Reference antenna: {refant}")
        for msname in mslist:
            if uvrange == "":
                logger.debug("UV-range is not specified. Determining it.")
                uvrange = get_gleam_uvrange(msname)
            logger.debug(f"UV-range for ms {msname} is {uvrange}")
            if uvrange != "":
                flag_uvranges = get_uvrange_exclude(uvrange)
                logger.debug(f"Flagging outside uv-range: {flag_uvranges}")
                for flag_uvrange in flag_uvranges:
                    try:
                        with suppress_output():
                            flagdata(
                                vis=msname,
                                mode="manual",
                                uvrange=flag_uvrange,
                                flagbackup=False,
                            )
                    except Exception:
                        logger.info(
                            f"No data to flag outside UV-range: {flag_uvrange}."
                        )
                        pass

        for cal_round in range(1, n_rounds + 1):
            for banner in print_banner(
                f"Calibration round: {cal_round}", no_print=True
            ).splitlines():
                logger.info(banner)
            if cal_round > 1:
                if perform_polcal:
                    do_polcal = True
                    logger.debug("Performing polarisation calibration.")
                flag_threshold = 5.0
            if cal_round == n_rounds + 1:
                do_postcal_flag = [False] * len(mslist)
            caltable_dic, succeed, failed, postcal_flags = single_round_cal_and_flag(
                mslist,
                dask_client,
                workdir,
                cal_round,
                refant,
                uvrange,
                do_polcal=do_polcal,
                applysol=applysol,
                do_postcal_flag=do_postcal_flag,
                flag_threshold=flag_threshold,
                n_threads=n_threads,
                mem_limit=mem_limit,
                logger=logger,
            )
            do_postcal_flag = postcal_flags
            caltables = list(caltable_dic.values())
            caltables = [x for sub in caltables for x in sub]

            ###################################
            # Backup
            ###################################
            os.makedirs(workdir + "/backup", exist_ok=True)
            for caltable in caltables:
                if caltable is not None and os.path.exists(caltable):
                    cal_ext = os.path.basename(caltable).split(".")[-1]
                    outputname = (
                        workdir
                        + "/backup/"
                        + os.path.basename(caltable).split(f".{cal_ext}")[0]
                        + "_round_"
                        + str(cal_round)
                        + f".{cal_ext}"
                    )
                    os.system("mv " + caltable + " " + outputname)

            ###############
            # Flag summary
            ###############
            tasks = []
            os.makedirs(f"{outdir}/flag_summary", exist_ok=True)
            for msname in mslist:
                summary_file = f"{outdir}/flag_summary/{os.path.basename(msname).split('.ms')[0]}_calflag_{cal_round}.summary"
                tasks.append(delayed(flagsummary)(msname, summary_file))
            dask_client.gather(dask_client.compute(tasks))
        all_caltables = glob.glob(f"{workdir}/backup/calibrator*cal")
        final_bcals, final_kcrosscals = filtered_final_caltables(all_caltables, workdir)
        if keep_backup:
            os.system(f"mv {workdir}/backup/* {workdir}")
        os.system(f"rm -rf {workdir}/backup")
        logger.info("Basic calibrations are done.")
        return 0, final_bcals, final_kcrosscals, succeed, failed
    except Exception:
        logger.exception("Exception in basic calibration.", exc_info=True)
        return 1, [], [], succeed, failed


def main(
    mslist,
    metafits,
    workdir,
    outdir,
    refant="",
    uvrange="",
    perform_polcal=True,
    keep_backup=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    verbose=False,
    dask_client=None,
):
    """
    Main function to perform basic calibration

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
    refant : str, optional
        Reference antenna
    uvrange : str, optional
        UV-range
    perform_polcal : bool, optional
        Perform polarization calibration
    keep_backup : bool, optional
        Keep backup
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction
    logfile : str, optional
        Log file name
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
    """
    logger = get_logger_safe()
    if verbose:
        logger.setLevel(logging.DEBUG)

    cpu_frac = min(0.8, abs(cpu_frac))
    mem_frac = min(0.8, abs(mem_frac))

    header = fits.getheader(metafits)
    obsid = header["GPSTIME"]

    mslist = mslist.split(",")
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    if outdir == "":
        outdir = workdir
    os.makedirs(outdir, exist_ok=True)
    caldir = f"{outdir}/caltables"
    os.makedirs(caldir, exist_ok=True)

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
                "basic_cal", logfile, jobname=jobname, password=password
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
        for banner in print_banner(
            "Starting basic calibration.", no_print=True
        ).splitlines():
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

        logger.info("##################################")
        logger.info(f"Total dask worker: {njobs}")
        logger.info(f"CPU per worker: {n_threads}")
        logger.info(f"Memory per worker: {mem_limit} GB")
        logger.info("#################################")
        msg, bcals, kcrosscals, succeed, failed = run_basic_cal_rounds(
            mslist,
            dask_client,
            workdir,
            outdir,
            refant=refant,
            uvrange=uvrange,
            perform_polcal=perform_polcal,
            keep_backup=keep_backup,
            n_threads=int(n_threads),
            mem_limit=float(mem_limit),
            logger=logger,
        )
        if len(bcals) == 0:
            logger.error("No bandpass caltable is made.")
            msg = 1
        else:
            logger.info(
                f"All bandpass caltables: {[os.path.basename(i) for i in bcals]}."
            )
            if len(kcrosscals) > 0:
                logger.info(
                    f"All cross-phase caltables: {[os.path.basename(i) for i in kcrosscals]}."
                )
            caltables = bcals + kcrosscals
            for caltable in caltables:
                if caltable is not None and os.path.exists(caltable):
                    cal_metadata = get_caltable_metadata(caltable)
                    freq_start = cal_metadata["Channel 0 frequency (MHz)"]
                    bw = cal_metadata["Bandwidth (MHz)"]
                    freq_end = freq_start + bw
                    ch_start = freq_to_MWA_coarse(freq_start)
                    ch_end = freq_to_MWA_coarse(freq_end)
                    if ch_end > ch_start:
                        coarse_chan = f"{ch_start}-{ch_end}"
                    else:
                        coarse_chan = f"{ch_start}"
                    if caltable.endswith(".bcal"):
                        final_caltable = (
                            caldir + f"/calibrator_{obsid}_ch_{coarse_chan}.bcal"
                        )
                    elif caltable.endswith(".kcrosscal"):
                        final_caltable = (
                            caldir + f"/calibrator_{obsid}_ch_{coarse_chan}.kcrosscal"
                        )
                    else:
                        final_caltable = (
                            caldir + f"/calibrator_{obsid}_ch_{coarse_chan}.cal"
                        )
                    os.system(f"rm -rf {final_caltable}")
                    os.system(f"cp -r {caltable} {final_caltable}")
                    os.system("rm -rf " + caltable)
        logger.info("Basic calibration runs are done successfully.")
    except Exception:
        logger.exception("Basic calibration runs failed.", exc_info=True)
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
            drop_cache(caldir)
            os.system(f"rm -rf {dask_dir}")
    return msg, succeed, failed


def cli():
    parser = argparse.ArgumentParser(
        description="Basic calibration using calibrator fields",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Name of measurement sets (comma separated)",
    )
    basic_args.add_argument(
        "metafits",
        type=str,
        help="Metafits file",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory for calibration outputs (default: auto-created next to MS)",
    )
    basic_args.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory (default: auto-created in the workdir)",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration parameters\n###################"
    )
    adv_args.add_argument("--refant", type=str, default="", help="Reference antenna")
    adv_args.add_argument(
        "--uvrange",
        type=str,
        default="",
        help="UV range for calibration (e.g. '>100lambda')",
    )
    adv_args.add_argument(
        "--no_perform_polcal",
        dest="perform_polcal",
        action="store_false",
        help="Disable polarization calibration",
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of measurement set after each calibration round",
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose logs")
    adv_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and PID tracking"
    )

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

    (
        msg,
        _,
        _,
    ) = main(
        args.mslist,
        args.metafits,
        args.workdir,
        args.outdir,
        refant=args.refant,
        uvrange=args.uvrange,
        perform_polcal=args.perform_polcal,
        keep_backup=args.keep_backup,
        verbose=args.verbose,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        jobid=args.jobid,
    )
    return msg
