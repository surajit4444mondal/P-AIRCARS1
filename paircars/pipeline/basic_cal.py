import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import sys
import os
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass
from casatools import msmetadata
from dask import delayed
from paircars.utils import *
from paircars.pipeline.flagging import single_ms_flag

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def run_bandpass(
    msname="",
    field="",
    scan="",
    uvrange="",
    refant="",
    solint="inf",
    solnorm=False,
    combine="",
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=-1,
):
    """
    Perform bandpass calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import bandpass, flagdata

    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_output():
        bandpass(
            vis=msname,
            caltable=f"{caltable_prefix}.bcal",
            field=str(field),
            scan=str(scan),
            uvrange=uvrange,
            refant=refant,
            solint=solint,
            solnorm=solnorm,
            combine=combine,
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
        )
        flagdata(
            vis=f"{caltable_prefix}.bcal",
            mode="rflag",
            datacolumn="CPARAM",
            flagbackup=False,
        )
    return caltable_prefix + ".bcal"


def run_crossphasecal(
    msname="",
    uvrange="",
    gaintable=[],
    n_threads=-1,
):
    """
    Perform crosshand phase calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import polcal, flagdata

    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_output():
        crossphasecal(
            msname,
            f"{caltable_prefix}.kcrosscal",
            uvrange=uvrange,
            gaintable=gaintable[0],
        )
    return caltable_prefix + ".kcrosscal"


def run_applycal(
    msname="",
    field="",
    scan="",
    applymode="",
    flagbackup=True,
    gaintable=[],
    gainfield=[],
    interp=[],
    calwt=[],
    n_threads=-1,
):
    """
    Perform apply calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import applycal

    with suppress_output():
        applycal(
            vis=msname,
            field=str(field),
            scan=str(scan),
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
    n_threads=-1,
    memory_limit=-1,
):
    """
    Perform apply calibration
    """
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
        memory_limit=memory_limit,
    )
    if msg > 0:
        print(f"Issue in post-calibration flagging in ms: {msname}")
    return


def single_ms_cal_and_flag(
    msname,
    cal_round,
    refant,
    uvrange,
    do_polcal=True,
    applysol=True,
    do_postcal_flag=True,
    flag_threshold=5.0,
    n_threads=-1,
    memory_limit=-1,
):
    """
    Single ms calibration and post-calibration flagging

    Parameters
    ----------
    msname : str
        Name of the measurement set
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
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    str
        Caltables
    """
    try:
        caltable_prefix = msname.split(".ms")[0] + "_caltable"
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
        print("##############################")
        print(f"Calibrating calibrator field ms: {msname}")
        print("###############################")
        applycal_gaintable = []
        applycal_gainfield = []
        applycal_interp = []

        ##############################
        # Bandpass calibration
        ##############################
        print(f"Performing bandpass calibrations on: {msname}")
        bpass_caltable = run_bandpass(
            msname,
            uvrange=uvrange,
            refant=refant,
            solint="inf",
            n_threads=n_threads,
        )
        if bpass_caltable is not None and os.path.exists(bpass_caltable):
            applycal_gaintable.append(bpass_caltable)
            applycal_gainfield.append("")
            applycal_interp.append("nearest,nearestflag")
        else:
            print(f"Bandpass calibration is not successful for ms: {msname}.")
            return []

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
                    uvrange=uvrange,
                    gaintable=applycal_gaintable,
                    n_threads=n_threads,
                )
                if crossphase_caltable is not None and os.path.exists(
                    crossphase_caltable
                ):
                    applycal_gaintable.append(crossphase_caltable)
                    applycal_gainfield.append("")
                    applycal_interp.append("nearest,nearestflag")

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
                do_flag_backup(msname, flagtype="flagdata")
                print(
                    f"Performing post-calibration flagging - MS: {msname}, threshold: {flag_threshold}"
                )
                run_postcal_flag(
                    msname,
                    datacolumn="residual",
                    threshold=flag_threshold,
                    n_threads=n_threads,
                    memory_limit=memory_limit,
                )

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
        ]
    except Exception as e:
        traceback.print_exc()
        return []
    finally:
        time.sleep(1)
        drop_cache(msname)


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
    cpu_frac=0.8,
    mem_frac=0.8,
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
    do_postcal_flag : bool, optional
        Perform post-calibration flagging
    flag_threashold : float, optional
        Flagging threshold
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction

    Returns
    -------
    dict
        A python dictionary cotaining measurement set name and its caltables
    """
    if cpu_frac > 0.8:
        cpu_frac = 0.8
    total_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
    if mem_frac > 0.8:
        mem_frac = 0.8
    total_mem = (psutil.virtual_memory().available * mem_frac) / (1024**3)  # In GB
    n_threads = max(1, int(total_cpu / len(mslist)))
    memory_limit = total_mem / len(mslist)
    tasks = [
        delayed(single_ms_cal_and_flag)(
            msname,
            cal_round,
            refant,
            uvrange,
            do_polcal=do_polcal,
            applysol=applysol,
            do_postcal_flag=do_postcal_flag,
            flag_threshold=flag_threshold,
            n_threads=n_threads,
            memory_limit=memory_limit,
        )
        for msname in mslist
    ]
    results = list(dask_client.gather(dask_client.compute(tasks)))
    caltable_dic = {}
    for i in range(len(mslist)):
        msname = mslist[i]
        caltables = results[i]
        caltables_clean = [x for x in caltables if x is not None]
        if len(caltables_clean) == 0:
            print(f"Basic calibration is not succssful for ms : {msname}")
        caltable_dic[msname] = caltables_clean
    return caltable_dic


def run_basic_cal_rounds(
    mslist,
    dask_client,
    workdir,
    outdir="",
    refant="",
    uvrange="",
    keep_backup=False,
    perform_polcal=False,
    cpu_frac=0.8,
    mem_frac=0.8,
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
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Success message
    list
        Caltables
    """
    try:
        from casatasks import flagdata

        os.chdir(workdir)
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
        print(f"Calibration for ms list: {mslist}.")
        print(f"Total calibration rounds: {n_rounds}")

        #################
        # Initial values
        #################
        do_polcal = False
        do_postcal_flag = True
        applysol = True
        flag_threshold = 6.0
        if refant == "":
            refant = get_refant(trial_ms)
        for msname in mslist:
            if uvrange == "":
                uvrange = get_gleam_uvrange(msname)
            flag_uvranges = get_uvrange_exclude(uvrange)
            for flag_uvrange in flag_uvranges:
                flagdata(
                    vis=msname, mode="manual", uvrange=flag_uvrange, flagbackup=False
                )

        for cal_round in range(1, n_rounds + 1):
            print("#################################")
            print(f"Calibration round: {cal_round}")
            print("#################################")
            if cal_round > 1:
                if perform_polcal:
                    do_polcal = True
                flag_threshold = 5.0
            caltable_dic = single_round_cal_and_flag(
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
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
            caltables = list(caltable_dic.values())
            caltables = [x for sub in caltables for x in sub]
            if keep_backup:
                print(f"Backup directory: {workdir}/backup")
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
                        os.system("cp -r " + caltable + " " + outputname)
            ###############
            # Flag summary
            ###############
            tasks = []
            for msname in mslist:
                summary_file = f"{outdir}/{os.path.basename(msname).split('.ms')[0]}_calflag_{cal_round}.summary"
                tasks.append(delayed(flagsummary)(msname, summary_file))
            results = list(dask_client.gather(dask_client.compute(tasks)))
        print("##################")
        print("Basic calibration is done successfully.")
        print("##################")
        return 0, caltables
    except Exception as e:
        traceback.print_exc()
        return 1, []


def main(
    mslist,
    workdir,
    outdir,
    refant="",
    uvrange="",
    perform_polcal=True,
    keep_backup=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    dask_client=None,
):
    """
    Main function to perform basic calibration

    Parameters
    ----------
    mslist : str
        Measurement set list (comma separated)
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
    start_remote_log : bool, optional
        Start logging to remote logger or not
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
        obsid = get_MWA_OBSID(mslist[0])
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

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
        and os.path.exists(f"{workdir}/jobname_password.npy")
        and logfile is not None
    ):
        time.sleep(1)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "basic_cal", logfile, jobname=jobname, password=password
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
            print("###################################")
            print("Starting initial calibration.")
            print("###################################")
            msg, caltables = run_basic_cal_rounds(
                mslist,
                dask_client,
                workdir,
                outdir,
                refant=refant,
                uvrange=uvrange,
                perform_polcal=perform_polcal,
                keep_backup=keep_backup,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
            )
            print(f"Caltables: {caltables}")
            for caltable in caltables:
                if caltable is not None and os.path.exists(caltable):
                    dest = caldir + "/" + os.path.basename(caltable)
                    if os.path.exists(dest):
                        os.system("rm -rf " + dest)
                    os.system("mv " + caltable + " " + caldir)
        else:
            print("Please provide a valid measurement set.")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(1)
        for msname in mslist:
            drop_cache(msname)
        drop_cache(workdir)
        drop_cache(caldir)
        clean_shutdown(observer)
        if dask_cluster is not None:
            dask_client.close()
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
    return msg


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
        required=True,
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
        "--no-perform_polcal",
        dest="perform_polcal",
        action="store_false",
        help="Disable polarization calibration",
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of measurement set after each calibration round",
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
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory fraction to use"
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
        args.workdir,
        args.outdir,
        refant=args.refant,
        uvrange=args.uvrange,
        perform_polcal=args.perform_polcal,
        keep_backup=args.keep_backup,
        start_remote_log=args.start_remote_log,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        logfile=args.logfile,
        jobid=args.jobid,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nBasic calibration is finished.\n###################\n"
    )
    os._exit(result)
