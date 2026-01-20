import logging
import resource
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import sys
import os
import copy
from casatools import msmetadata, table
from casatasks import flagmanager
from dask import delayed
from functools import partial
from astropy.io import fits
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


def do_polselfcal(
    msname="",
    workdir="",
    selfcaldir="",
    metafits="",
    max_iter=100,
    max_DR=100000,
    min_iter=2,
    threshold=3.0,
    DR_convegerence_frac=0.1,
    uvrange="",
    minuv=0,
    solint="10s",
    weight="briggs",
    robust=0.0,
    solar_selfcal=True,
    ncpu=-1,
    mem=-1,
    logfile="selfcal.log",
):
    """
    Do selfcal iterations and use convergence rules to stop

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory
    selfcaldir : str
        Working directory
    metafits : str
        Metafits file
    max_iter : int, optional
        Maximum numbers of selfcal iterations
    max_DR : float, optional
        Maximum dynamic range
    min_iter : int, optional
        Minimum numbers of seflcal iterations at different stages
    threshold: float, optional
        Threshold of CLEANing
    DR_convegerence_frac : float, optional
        Dynamic range fractional change to consider as converged
    uvrange : str, optional
        UV-range for calibration
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    solint : str, optional
        Solutions interval
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    ncpu : int, optional
        Number of CPU threads to use
    mem : float, optional
        Memory in GB to use
    logfile : str, optional
        Log file name

    Returns
    -------
    int
        Success message
    str
        Final caltable
    """
    limit_threads(n_threads=ncpu)
    from casatasks import split, flagdata, flagmanager

    sub_observer = None
    logger, logfile = create_logger(
        os.path.basename(logfile).split(".log")[0], logfile, verbose=False
    )
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            sub_observer = init_logger(
                "remotelogger_selfcal_{os.path.basename(msname).split('.ms')[0]}",
                logfile,
                jobname=jobname,
                password=password,
            )
    try:
        msname = os.path.abspath(msname.rstrip("/"))
        selfcaldir = selfcaldir.rstrip("/")
        os.makedirs(selfcaldir, exist_ok=True)

        os.chdir(selfcaldir)
        selfcalms = selfcaldir + "/polselfcal_" + os.path.basename(msname)
        if os.path.exists(selfcalms):
            os.system("rm -rf " + selfcalms)
        if os.path.exists(selfcalms + ".flagversions"):
            os.system("rm -rf " + selfcalms + ".flagversions")

        ##############################
        # Spliting corrected data
        ##############################
        hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        msmd = msmetadata()
        msmd.open(msname)
        scan = int(msmd.scannumbers()[0])
        field = int(msmd.fieldsforscan(scan)[0])
        freqMHz = msmd.meanfreq(0, unit="MHz")
        msmd.close()
        if hascor:
            logger.info(f"Spliting corrected data to ms : {selfcalms}")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="corrected",
                )
        else:
            logger.warning("Corrected data column is not present.")
            logger.info(f"Spliting data to ms : {selfcalms}")
            with suppress_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="data",
                )
        msname = selfcalms

        ################################################################
        # Initial flagging -- zeros, extreme bad data
        ################################################################
        logger.info("Initial flagging -- zeros and extreme bad data.")
        with suppress_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="data",
                flagbackup=False,
            )
            result = uvbin_flag(
                msname, uvbin_size=10, mode="rflag", threshold=10.0, flagbackup=True
            )
            if result != 0:
                logger.info(f"UV-bin flagging is not successful.")

        ############################################
        # Imaging and calibration parameters
        ############################################
        logger.info(f"Estimating imaging Parameters ...")
        cellsize = calc_cellsize(msname, 3)
        instrument_fov = calc_field_of_view(msname, FWHM=False)
        sun_size = calc_sun_dia(freqMHz)
        fov = min(
            instrument_fov, 2 * sun_size * 60
        )  # 2 times sun size at that frequency
        imsize = int(fov / cellsize)
        pow2 = np.ceil(np.log2(imsize)).astype("int")
        possible_sizes = []
        for p in range(pow2):
            for k in [3, 5]:
                possible_sizes.append(k * 2**p)
        possible_sizes = np.sort(np.array(possible_sizes))
        possible_sizes = possible_sizes[possible_sizes >= imsize]
        imsize = max(512, int(possible_sizes[0]))
        unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
        refant_name = unflagged_antenna_names[0]
        msmd = msmetadata()
        msmd.open(msname)
        refant = msmd.antennaids(refant)[0]
        msmd.close()

        ############################################
        # Initiating selfcal Parameters
        ############################################
        logger.info(f"Estimating self-calibration parameters...")
        DR1 = 0.0
        DR2 = 0.0
        DR3 = 0.0
        RMS1 = -1.0
        RMS2 = -1.0
        RMS3 = -1.0
        QL1 = QL2 = QL3 = 1.0
        UL1 = UL2 = UL3 = 1.0
        VL1 = VL2 = VL3 = 1.0
        num_iter = 0
        last_round_gaintable = []
        do_uvsub_flag = False
        os.system("rm -rf *_selfcal_present*")

        ##########################################
        # Starting selfcal loops
        ##########################################
        while True:
            ##################################
            # Selfcal round parameters
            ##################################
            logger.info("######################################")
            logger.info(f"Selfcal iteration : " + str(num_iter))
            (
                msg,
                gaintable,
                dyn,
                rms,
                final_image,
                final_model,
                final_residual,
                leakage_info,
            ) = selfcal_round(
                msname,
                metafits,
                logger,
                selfcaldir,
                cellsize,
                imsize,
                round_number=num_iter,
                uvrange=uvrange,
                minuv=minuv,
                solint=solint,
                refant=str(refant),
                threshold=threshold,
                weight=weight,
                robust=robust,
                use_solar_mask=solar_selfcal,
                do_polcal=True,
                do_intensity_cal=False,
                pbcor=True,
                leakagecor=True,
                pbuncor=True,
                ncpu=ncpu,
                mem=round(mem, 2),
            )
            if msg == 1:
                logger.info(f"No model flux is picked up.\n")
                os.system("rm -rf *_selfcal_present*")
                return msg, []
            if msg > 1:
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return msg, []
            leakage_info = np.array(leakage_info)
            avg_leakage = np.nanmedian(leakage_info, axis=0)
            q_leakage, u_leakage, v_leakage, _, _, _ = avg_leakage
            if num_iter == 0:
                DR1 = DR3 = DR2 = dyn
                RMS1 = RMS2 = RMS3 = rms
                QL1 = QL2 = QL3 = q_leakage
                UL1 = UL2 = UL3 = u_leakage
                VL1 = VL2 = VL3 = v_leakage
            elif num_iter == 1:
                DR3 = dyn
                RMS2 = RMS1
                RMS1 = rms
                QL3 = q_leakage
                UL3 = u_leakage
                VL3 = v_leakage
            else:
                DR1 = DR2
                DR2 = DR3
                DR3 = dyn
                RMS3 = RMS2
                RMS2 = RMS1
                RMS1 = rms
                QL1 = QL2
                UL1 = UL2
                VL1 = VL2
                QL2 = QL3
                UL2 = UL3
                VL2 = VL3
                QL3 = q_leakage
                UL3 = u_leakage
                VL3 = v_leakage
            logger.info(
                f"RMS based dynamic ranges: "
                + str(DR1)
                + ","
                + str(DR2)
                + ","
                + str(DR3)
            )
            logger.info(
                f"RMS of the images: " + str(RMS1) + "," + str(RMS2) + "," + str(RMS3)
            )
            logger.info(
                f"Stokes I to Q leakage: {round(QL1*100.0,3)}, {round(QL2*100.0,3)}, {round(QL3*100.0,3)}%."
            )
            logger.info(
                f"Stokes I to U leakage: {round(UL1*100.0,3)}, {round(UL2*100.0,3)}, {round(UL3*100.0,3)}%."
            )
            logger.info(
                f"Stokes I to V leakage: {round(VL1*100.0,3)}, {round(VL2*100.0,3)}, {round(VL3*100.0,3)}%.\n"
            )
            leakage_coverged = (QL3 == 0.0 and UL3 == 0.0 and VL3 == 0.0) or (
                (QL2 - QL3) <= 0.01 and (UL2 - UL3) <= 0.01 and (VL2 - VL3) <= 0.01
            )

            ##############################################################
            # If DR is decreasing (DR decrease in pol selfcal)
            ##############################################################
            if (
                (DR3 < 0.9 * DR2 and DR2 > 1.5 * DR1)
                and num_iter > min_iter
                and leakage_coverged
            ):
                logger.info(
                    f"Dynamic range is decreasing after minimum numbers of rounds.\n"
                )
                if do_uvsub_flag is False:
                    logger.info("Starting uvsub flagging.")
                    do_uvsub_flag = True
                else:
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, last_round_gaintable

            ###########################
            # If maximum DR has reached
            ###########################
            if DR3 >= max_DR and num_iter > min_iter and leakage_coverged:
                logger.info(f"Maximum dynamic range is reached.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, gaintable

            ##########################
            # If DR suddenly decreased
            ##########################
            if DR3 < 0.7 * DR2 and num_iter > min_iter and leakage_coverged:
                logger.info(
                    f"Dynamic range dropped suddenly. Using last round caltable as final.\n"
                )
                if do_uvsub_flag is False:
                    logger.info("Starting uvsub flagging.")
                    do_uvsub_flag = True
                else:
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, last_round_gaintable

            ###########################
            # Checking DR convergence
            ###########################
            ########################################
            # Condition 1
            # If DR does not increase a certain percentage
            # Leakage becomes zero or did not reduce
            ########################################
            if (
                abs(DR1 - DR2) / DR2 < DR_convegerence_frac
                and num_iter > min_iter
                and leakage_coverged
            ):
                if do_uvsub_flag is False:
                    logger.info("Starting uvsub flagging.")
                    do_uvsub_flag = True
                else:
                    logger.info(f"Self-calibration has converged.\n")
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, gaintable
            #########################################
            # In apcal and maximum iteration has reached
            #########################################
            elif num_iter > min_iter and num_iter == max_iter:
                logger.info(
                    f"Self-calibration is finished. Maximum iteration is reached.\n"
                )
                if leakage_coverged is False:
                    logger.warning("Leakage did not converge.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, gaintable
            num_iter += 1
            last_round_gaintable = gaintable
    except Exception as e:
        traceback.print_exc()
        os.system("rm -rf *_selfcal_present*")
        time.sleep(5)
        clean_shutdown(sub_observer)
        return 1, []


def main(
    mslist,
    metafits,
    workdir,
    caldir,
    threshold=3,
    max_iter=100,
    max_DR=100000,
    min_iter=2,
    conv_frac=0.1,
    solint="10s",
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    solar_selfcal=True,
    keep_backup=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_client=None,
):
    """
    Perform iterative self-calibration on a list of measurement sets.

    Parameters
    ----------
    mslist : str
        Comma-separated list of target measurement sets to be self-calibrated.
    metafits : str
        Metafits file
    workdir : str
        Path to the working directory for outputs, intermediate files, and logs.
    caldir : str
        Directory containing calibration tables (e.g., from flux or phase calibrators).
    threshold : float, optional
        Threshold for CLEANing. Default is 3.
    max_iter : int, optional
        Maximum number of self-calibration iterations. Default is 100.
    max_DR : float, optional
        Maximum dynamic range allowed before halting iterations. Default is 100000.
    min_iter : int, optional
        Minimum number of iterations before checking for convergence. Default is 2.
    conv_frac : float, optional
        Convergence criterion: fractional change in dynamic range below which iteration stops. Default is 0.1.
    solint : str, optional
        Solution interval for gain calibration (e.g., "inf", "10s", "int"). Default is "60s".
    uvrange : str, optional
        UV range to be used for imaging and calibration, in CASA format. Default is "" (all baselines).
    minuv : float, optional
        Minimum baseline length (in wavelengths) to include. Default is 0.
    weight : str, optional
        Weighting scheme for imaging (e.g., "natural", "uniform", "briggs"). Default is "briggs".
    robust : float, optional
        Robustness parameter for Briggs weighting (ignored if not using "briggs"). Default is 0.0.
    solar_selfcal : bool, optional
        If True, uses solar-specific masking and flux normalization. Default is True.
    keep_backup : bool, optional
        If True, keeps backup MS before applying selfcal solutions. Default is False.
    cpu_frac : float, optional
        Fraction of available CPUs to use per job. Default is 0.8.
    mem_frac : float, optional
        Fraction of available system memory to use per job. Default is 0.8.
    logfile : str, optional
        Log file name
    jobid : int, optional
        Identifier for job tracking and logging. Default is 0.
    start_remote_log : bool, optional
        Whether to initiate remote logging via job credentials. Default is False.
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

    if caldir == "" or not os.path.exists(caldir):
        caldir = f"{workdir}/caltables"
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
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "all_selfcal", logfile, jobname=jobname, password=password
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
        nworker = min(len(mslist), int(psutil.cpu_count() * cpu_frac) - 1)
        scale_worker_and_wait(dask_cluster, nworker + 1)

    ###########################
    # WSClean container
    ###########################
    container_name = "solarwsclean"
    container_present = check_udocker_container(container_name)
    if not container_present:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name is None:
            print(
                f"Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    org_mslist = copy.deepcopy(mslist)
    try:
        if len(mslist) == 0:
            print("Please provide at-least one measurement set.")
            msg = 1
        else:
            partial_do_selfcal = partial(
                do_polselfcal,
                metafits=str(metafits),
                max_iter=int(max_iter),
                max_DR=float(max_DR),
                min_iter=int(min_iter),
                DR_convegerence_frac=float(conv_frac),
                uvrange=str(uvrange),
                minuv=float(minuv),
                solint=str(solint),
                weight=str(weight),
                robust=float(robust),
                threshold=float(threshold),
                solar_selfcal=solar_selfcal,
            )

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
                    os.system(f"rm -rf {ms}")
            mslist = filtered_mslist

            ######################################
            # Resetting maximum file limit
            ######################################
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            new_soft_limit = max(soft_limit, int(0.8 * hard_limit))
            if soft_limit < new_soft_limit:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))

            num_fd_list = []
            if len(mslist) == 0:
                print("No filtered ms to continue.")
                return 1
            else:
                for ms in mslist:
                    msmd = msmetadata()
                    msmd.open(ms)
                    times = msmd.timesforspws(0)
                    timeres = np.diff(times)
                    pos = np.where(timeres > 3 * np.nanmedian(timeres))[0]
                    max_intervals = min(1, len(pos))
                    msmd.close()
                    per_job_fd = (
                        max_intervals * 4 * 2
                    )  # 4 types of images, 2 is fudge factor
                    if per_job_fd == 0:
                        per_job_fd = 1
                    num_fd_list.append(per_job_fd)
                total_fd = max(num_fd_list) * len(mslist)

                if cpu_frac > 0.8:
                    cpu_frac = 0.8
                total_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
                if mem_frac > 0.8:
                    mem_frac = 0.8
                total_mem = (psutil.virtual_memory().available * mem_frac) / (
                    1024**3
                )  # In GB
                njobs = min(len(mslist), int(new_soft_limit / total_fd))
                njobs = max(1, min(total_cpu, njobs))

                #####################################
                # Determining per jobs resource
                #####################################
                n_threads = max(1, int(total_cpu / njobs))
                mem_limit = total_mem / njobs
                print("#################################")
                print(f"Total dask worker: {njobs}")
                print(f"CPU per worker: {n_threads}")
                print(f"Memory per worker: {round(mem_limit,2)} GB")
                print("#################################")

                #####################################s
                os.makedirs(f"{workdir}/logs", exist_ok=True)
                tasks = []
                for ms in mslist:
                    logfile = (
                        workdir
                        + "/logs/"
                        + os.path.basename(ms).split(".ms")[0]
                        + "_selfcal.log"
                    )
                    print(f"MS name: {ms}, Log file: {logfile}")
                    tasks.append(
                        delayed(partial_do_selfcal)(
                            ms,
                            workdir,
                            workdir
                            + "/"
                            + os.path.basename(ms).split(".ms")[0]
                            + "_selfcal",
                            ncpu=n_threads,
                            mem=mem_limit,
                            logfile=logfile,
                        )
                    )
                print("Starting all self-calibration...")
                results = list(dask_client.gather(dask_client.compute(tasks)))

                dcal_list = []
                for i in range(len(results)):
                    r = results[i]
                    msg = r[0]
                    if msg != 0:
                        print(
                            f"Self-calibration was not successful for ms: {mslist[i]}."
                        )
                    else:
                        gaintables = r[1]
                        dcal = gaintables[0]
                        cal_metadata = get_quartical_table_metadata(dcal)
                        freq_start = cal_metadata["Channel 0 frequency (MHz)"]
                        bw = cal_metadata["Bandwidth (MHz)"]
                        freq_end = freq_start + bw
                        ch_start = freq_to_MWA_coarse(freq_start)
                        ch_end = freq_to_MWA_coarse(freq_end)
                        if freq_end > freq_start and ch_end == ch_start:
                            ch_end = ch_start + 1
                        final_leakage_caltable = (
                            caldir + f"/selfcal_coarsechan_{ch_start}_{ch_end}.dcal"
                        )
                        os.system(f"cp -r {dcal} {final_leakage_caltable}")
                        dcal_list.append(final_leakage_caltable)

                if not keep_backup:
                    for ms in mslist:
                        selfcaldir = (
                            workdir
                            + "/"
                            + os.path.basename(ms).split(".ms")[0]
                            + "_selfcal"
                        )
                        os.system("rm -rf " + selfcaldir)
                if len(dcal_list) > 0:
                    print(f"Final leakage selfcal caltables: {dcal_list}")
                    msg = 0
                else:
                    print("No self-calibration is successful.")
                    msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in org_mslist:
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
        description="Self-calibration", formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Caltable directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration and imaging parameters\n###################"
    )
    adv_args.add_argument(
        "--metafits",
        type=str,
        default="",
        help="Metafits file",
        metavar="String",
    )
    adv_args.add_argument(
        "--threshold",
        type=float,
        default=3,
        help="CLEANing threshold",
        metavar="Float",
    )
    adv_args.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of selfcal iterations",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--max_DR",
        type=float,
        default=100000,
        help="Maximum dynamic range",
        metavar="Float",
    )
    adv_args.add_argument(
        "--min_iter",
        type=int,
        default=2,
        help="Minimum number of selfcal iterations",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--conv_frac",
        type=float,
        default=0.1,
        help="Fractional change in DR to determine convergence",
        metavar="Float",
    )
    adv_args.add_argument("--solint", type=str, default="10s", help="Solution interval")
    adv_args.add_argument(
        "--uvrange",
        type=str,
        default="",
        help="Calibration UV-range (CASA format)",
    )
    adv_args.add_argument(
        "--minuv",
        type=float,
        default=0,
        help="Minimum UV-lambda used for imaging",
        metavar="Float",
    )
    adv_args.add_argument("--weight", type=str, default="briggs", help="Imaging weight")
    adv_args.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust parameter for briggs weight",
        metavar="Float",
    )
    adv_args.add_argument(
        "--no_solar_selfcal",
        action="store_false",
        dest="solar_selfcal",
        help="Do not perform solar self-calibration",
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of self-calibration rounds",
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
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        mslist=args.mslist,
        metafits=args.metafits,
        workdir=args.workdir,
        caldir=args.caldir,
        threshold=args.threshold,
        max_iter=args.max_iter,
        max_DR=args.max_DR,
        min_iter=args.min_iter,
        conv_frac=args.conv_frac,
        solint=args.solint,
        uvrange=args.uvrange,
        minuv=args.minuv,
        weight=args.weight,
        robust=args.robust,
        solar_selfcal=args.solar_selfcal,
        keep_backup=args.keep_backup,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    if result > 0:
        result = 1
    print("\n###################\nSelf-calibration is done.\n###################\n")
    os._exit(result)
