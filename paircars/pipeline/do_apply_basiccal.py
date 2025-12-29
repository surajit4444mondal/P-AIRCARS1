import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import warnings
import copy
import time
import glob
import sys
import os
from casatools import table
from dask import delayed
from astropy.io import fits
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


def scale_bandpass(bandpass_table, cal_attn, target_attn):
    """
    Scale a bandpass calibration table using attenuation data.

    Parameters
    ----------
    bandpass_table : str
        Input bandpass calibration table.
    cal_attn : float
        Calibrator attenuation
    target_attn : float
        Target attenuation

    Returns
    -------
    str
        Name of the output table.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    output_table = bandpass_table.split(".bcal")[0] + "_att.bcal"
    # Prepare output table
    if os.path.exists(output_table):
        os.system(f"rm -rf {output_table}")
    os.system(f"cp -r {bandpass_table} {output_table}")
    if cal_attn == target_attn:
        return output_table
    else:
        tb = table()
        tb.open(output_table, nomodify=False)
        gain = tb.getcol("CPARAM")
        flag = tb.getcol("FLAG")
        scaling = 10 ** (-(target_attn - cal_attn) / 20.0)
        gain *= scaling
        tb.putcol("CPARAM", gain)
        tb.flush()
        tb.close()
        return output_table


def applysol(
    msname="",
    gaintable=[],
    gainfield=[],
    interp=[],
    parang=False,
    applymode="calflag",
    overwrite_datacolumn=False,
    n_threads=-1,
    memory_limit=-1,
    force_apply=False,
    soltype="basic",
):
    """
    Apply flux calibrated and attenuation calibrated solutions

    Parameters
    ----------
    msname : str
        Measurement set
    gaintable : list, optional
        Caltable list
    gainfield : list, optional
        Gain field list
    interp : list, optional
        Gain interpolation
    parang : bool, optional
        Parallactic angle apply or not
    applymode : str, optional
        Apply mode
    overwrite_datacolumn : bool, optional
        Overwrite data column with corrected solutions
    n_threads : int, optional
        Number of OpenMP threads
    memory_limit : float, optional
        Memory limit in GB
    force_apply : bool, optional
        Force to apply solutions if it is already applied
    soltype : str, optional
        Solution type

    Returns
    -------
    int
        Success message
    """
    limit_threads(n_threads=n_threads)
    from casatasks import applycal, flagdata, split, clearcal

    if soltype == "basic":
        check_file = "/.applied_sol"
    else:
        check_file = "/.applied_selfcalsol"
    try:
        if os.path.exists(msname + check_file) and force_apply == False:
            print("Solutions are already applied.")
            return 0
        else:
            if os.path.exists(msname + check_file) and force_apply:
                with suppress_output():
                    clearcal(vis=msname)
                    flagdata(vis=msname, mode="unflag", spw="0", flagbackup=False)
                if os.path.exists(msname + ".flagversions"):
                    os.system("rm -rf " + msname + ".flagversions")
            with suppress_output():
                applycal(
                    vis=msname,
                    gaintable=gaintable,
                    gainfield=gainfield,
                    applymode=applymode,
                    interp=interp,
                    calwt=[False] * len(gaintable),
                    parang=parang,
                    flagbackup=False,
                )
        if overwrite_datacolumn:
            outputvis = msname.split(".ms")[0] + "_cor.ms"
            if os.path.exists(outputvis):
                os.system(f"rm -rf {outputvis}")
            touch_file_names = glob.glob(f"{msname}/.*")
            if len(touch_file_names) > 0:
                touch_file_names = [os.path.basename(f) for f in touch_file_names]
            with suppress_output():
                split(vis=msname, outputvis=outputvis, datacolumn="corrected")
            if os.path.exists(outputvis):
                os.system(f"rm -rf {msname} {msname}.flagversions")
                os.system(f"mv {outputvis} {msname}")
            for t in touch_file_names:
                os.system(f"touch {msname}/{t}")
        os.system("touch " + msname + check_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def run_all_applysol(
    mslist,
    target_metafits,
    calibrator_metafits,
    dask_client,
    workdir,
    caldir,
    overwrite_datacolumn=False,
    applymode="calflag",
    force_apply=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply basic-calibration solutions on all target scans

    Parameters
    ----------
    mslist : str
        Measurement set list
    target_metafits : str
        Target metafits file
    calibrator_metafits : str
        Calibrator metafits file
    dask_client : dask.client
        Dask client
    workdir : str
        Working directory
    caldir : str
        Calibration directory
    overwrite_datacolumn : bool, optional
        Overwrite data column or not
    applymode : str, optional
        Apply mode
    force_apply : bool, optional
        Force to apply solutions even already applied
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    --------
    list
        Calibrated target scans
    """
    try:
        if cpu_frac > 0.8:
            cpu_frac = 0.8
        total_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
        if mem_frac > 0.8:
            mem_frac = 0.8
        total_mem = (psutil.virtual_memory().available * mem_frac) / (1024**3)  # In GB
        os.chdir(workdir)
        mslist = np.unique(mslist).tolist()

        calibrator_header = fits.getheader(calibrator_metafits)
        target_header = fits.getheader(target_metafits)
        calibrator_obsid = calibrator_header["GPSTIME"]
        cal_attn = calibrator_header["ATTEN_DB"]
        target_attn = target_header["ATTEN_DB"]

        bandpass_table = glob.glob(caldir + f"/{calibrator_obsid}_caltable.bcal")
        crossphase_table = glob.glob(caldir + f"/{calibrator_obsid}_caltable.kcrosscal")

        if len(bandpass_table) == 0:
            print(f"No bandpass table is present in calibration directory : {caldir}.")
            return []
        if len(crossphase_table) == 0:
            print(
                f"No crosshand phase solution is present in calibration directory : {caldir}. Applying only bandpass solutions."
            )
            use_only_bandpass = True

        ################################
        # Scale bandpass for attenuators
        ################################
        att_caltables = []
        for bpass_table in bandpass_table:
            att_caltable = scale_bandpass(bpass_table, cal_attn, target_attn)
            att_caltables.append(att_caltable)

        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print(f"Issue in : {ms}")
                os.system("rm -rf {ms}")
        mslist = filtered_mslist
        if len(mslist) == 0:
            print("No valid measurement set.")
            return 1

        ####################################
        # Applycal jobs
        ####################################
        print(f"Total ms list: {len(mslist)}")
        tasks = []
        njobs = min(total_cpu, len(mslist))
        n_threads = max(1, int(total_cpu / njobs))
        mem_limit = total_mem / njobs

        print("#################################")
        print(f"Total dask worker: {njobs}")
        print(f"CPU per worker: {n_threads}")
        print(f"Memory per worker: {round(mem_limit,2)} GB")
        print("#################################")

        for ms in mslist:
            msmd = msmetdata()
            msmd.open(ms)
            ms_freq = msmd.meanfreq(0, unit="MHz")
            msmd.close()
            final_bpasstable = get_nearest_bandpass_table(att_caltables, ms_freq)
            final_crossphasetable = get_nearest_bandpass_table(
                crossphase_table, ms_freq
            )
            final_gaintable = [final_bpasstable, final_crossphasetable]
            interp = ["nearestflag", "nearestflag"]
            tasks.append(
                delayed(applysol)(
                    ms,
                    gaintable=final_gaintable,
                    overwrite_datacolumn=overwrite_datacolumn,
                    applymode=applymode,
                    interp=interp,
                    n_threads=n_threads,
                    parang=parang,
                    memory_limit=mem_limit,
                    force_apply=force_apply,
                )
            )
        print(f"Applying solutions")
        results = list(dask_client.gather(dask_client.compute(tasks)))
        if np.nansum(results) == 0:
            print("##################")
            print(
                "Applying basic calibration solutions for target are done successfully."
            )
            print("##################")
            return 0
        else:
            print("##################")
            print(
                "Applying basic calibration solutions for target scans are not done successfully."
            )
            print("##################")
            return 1
    except Exception as e:
        traceback.print_exc()
        os.system("rm -rf casa*log")
        print("##################")
        print(
            "Applying basic calibration solutions for target scans are not done successfully."
        )
        print("##################")
        return 1


def main(
    mslist,
    calibrator_metafits,
    target_metafits,
    workdir,
    caldir,
    use_only_bandpass=False,
    applymode="calflag",
    overwrite_datacolumn=False,
    force_apply=False,
    do_post_flag=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    dask_client=None,
):
    """
    Apply calibration solutions to a list of measurement sets with optional post-flagging.

    Parameters
    ----------
    mslist : str
        Comma-separated list of measurement set paths to which calibration will be applied.
    calibrator_metafits : str
        Calibrator metafits
    target_metafits : str
        Target metadfits
    workdir : str
        Directory for logs, PID files, and temporary data products.
    caldir : str
        Path to directory containing calibration tables (e.g., bandpass, gain, polarization).
    use_only_bandpass : bool, optional
        If True, applies only the bandpass calibration (no gain or polarization cal). Default is False.
    applymode : str, optional
        CASA calibration application mode (e.g., "calonly", "calflag", "flagonly"). Default is "calflag".
    overwrite_datacolumn : bool, optional
        If True, overwrites the CORRECTED column during calibration. Default is False.
    force_apply : bool, optional
        If True, forces re-application of calibration even if it appears already applied. Default is False.
    start_remote_log : bool, optional
        Whether to enable remote logging using job credentials in `workdir`. Default is False.
    cpu_frac : float, optional
        Fraction of CPU resources to allocate per worker. Default is 0.8.
    mem_frac : float, optional
        Fraction of system memory to allocate per worker. Default is 0.8.
    logfile : str or None, optional
        Path to the logfile. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Identifier for tracking the job and saving PID. Default is 0.
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

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(mslist.split(",")[0])) + "/workdir"
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
                "apply_basiccal", logfile, jobname=jobname, password=password
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
        print("###################################")
        print("Starting applying solutions...")
        print("###################################")

        if caldir == "" or not os.path.exists(caldir):
            print("Provide existing caltable directory.")
            msg = 1
        else:
            mslist = mslist.split(",")
            msg = run_all_applysol(
                mslist,
                calibrator_metafits,
                target_metafits,
                dask_client,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                overwrite_datacolumn=overwrite_datacolumn,
                applymode=applymode,
                force_apply=force_apply,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)
        clean_shutdown(observer)
        if dask_cluster is not None:
            dask_client.close()
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Apply basic calibration solutions to target scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required)",
    )
    basic_args.add_argument(
        "calibrator_metafits",
        type=str,
        help="Calibrator metafits (required)",
    )
    basic_args.add_argument(
        "target_metafits",
        type=str,
        help="Target metafits (required)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory for intermediate files",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Directory containing calibration tables",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--use_only_bandpass",
        action="store_true",
        help="Use only bandpass calibration solutions",
    )
    adv_args.add_argument(
        "--applymode",
        type=str,
        default="calflag",
        help="Applycal mode (e.g. 'calonly', 'calflag')",
    )
    adv_args.add_argument(
        "--overwrite_datacolumn",
        action="store_true",
        help="Overwrite corrected data column in MS",
    )
    adv_args.add_argument(
        "--force_apply",
        action="store_true",
        help="Force apply calibration even if already applied",
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
        "--jobid", type=str, default="0", help="Job ID for logging and process tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        args.mslist,
        args.calibrator_metafits,
        args.target_metafits,
        args.workdir,
        args.caldir,
        use_only_bandpass=args.use_only_bandpass,
        applymode=args.applymode,
        overwrite_datacolumn=args.overwrite_datacolumn,
        force_apply=args.force_apply,
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
        "\n###################\nApplying calibration solutions are done.\n###################\n"
    )
    os._exit(result)
