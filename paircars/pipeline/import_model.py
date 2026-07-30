import os
import glob
import numpy as np
import time
import sys
import traceback
import logging
import argparse
from dask import delayed
from casatasks import setjy
from casatools import table as casatable, msmetadata
from paircars.utils.basic_utils import (
    suppress_output,
    get_datadir,
    print_banner,
    capture_all_output,
)
from paircars.utils.logger_utils import (
    clean_shutdown,
    init_logger,
    get_logger_safe,
)
from paircars.utils.mwa_utils import get_ncoarse
from paircars.utils.proc_manage_utils import (
    scale_worker_and_wait,
    get_local_dask_cluster,
    get_scheduler_name,
)
from paircars.utils.resource_utils import drop_cache
from paircars.utils.udocker_utils import (
    run_hyperdrive,
    initialize_hyperdrive_container,
    check_udocker_container,
)
from paircars.utils.imaging import calc_field_of_view

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


def import_hyperdrive_model_wrapper(*args, **kwargs):
    with capture_all_output() as (out, err):
        result = import_hyperdrive_model(*args, **kwargs)
        return args[0], result, out.getvalue(), err.getvalue()


def import_hyperdrive_model(
    msname, metafits, beamfile="", sourcelist="", ncpu=1, verbose=False
):
    """
    Simulate visibilities and import in the measurement set

    Parameters
    ----------
    msname : str
        Name of the measurement set
    metafits : str
        Name of the metafits file
    beamfile : str, optional
        Beam file name
    sourcelist : str, optional
        Source file name
    ncpu : int, optional
        Number of cpu threads to use
    verbose : bool, optional
        Verbose output or not
    """
    ncpu = max(1, ncpu)
    starttime = time.time()
    msname = msname.rstrip("/")
    os.system(f"rm -rf {msname}/.modeling_*")
    msname = os.path.abspath(msname)
    print(f"Importing model for ms: {msname}")
    if beamfile == "" or os.path.exists(beamfile) is False:
        with suppress_output():
            msmd = msmetadata()
            msmd.open(msname)
            freqres = msmd.chanres(0, unit="kHz")[0]
            msmd.close()
        beam_files = glob.glob(f"{datadir}/mwa_full_embedded_element_pattern*.h5")
        beam_files_freqs = []
        for beamfile in beam_files:
            if os.path.basename(beamfile) == "mwa_full_embedded_element_pattern.h5":
                beam_file_freq = 1280.0
            else:
                beam_file_freq = float(
                    os.path.basename(beamfile)
                    .split(".h5")[0]
                    .split("mwa_full_embedded_element_pattern_")[-1]
                )
            beam_files_freqs.append(beam_file_freq)
        beam_files_freqs = np.array(beam_files_freqs)
        pos = np.argmin(np.abs(beam_files_freqs - freqres))
        beamfile = beam_files[pos]
    if sourcelist == "" or os.path.exists(sourcelist) is not True:
        sourcelist = f"{datadir}/GGSM.txt"
    model_msname = msname.split(".ms")[0] + "_model.ms"
    try:
        with suppress_output():
            msmd = msmetadata()
            msmd.open(msname)
            nchan = msmd.nchan(0)
            mid_freq = msmd.meanfreq(0, unit="MHz")
            freqres = msmd.chanres(0, unit="kHz")[0]
            msmd.nantennas()
            times = msmd.timesforfield(0)
            ntime = len(times)
            timeres = msmd.exposuretime(scan=1)["value"]
            msmd.nrows()
            msmd.close()
        print(f"Beam file: {beamfile}")
        print(f"Source model file: {sourcelist}")
        instrument_fov = round(calc_field_of_view(msname, FWHM=True)/3600.0,2)
        hyperdrive_cmd_args = [
            "hyperdrive",
            "vis-simulate",
            "-m",
            metafits,
            "--beam-file",
            beamfile,
            "--middle-freq",
            str(mid_freq),
            "--freq-res",
            str(freqres),
            "--time-res",
            str(timeres),
            "--source-dist-cutoff",
            f"{instrument_fov}",
            "-s",
            sourcelist,
            "-n",
            "1000",
            "--output-model-files",
            f"{model_msname}",
            "--output-model-freq-average",
            f"{freqres}kHz",
            "--num-fine-channels",
            str(nchan),
            "--num-timesteps",
            str(ntime),
            "--output-model-time-average",
            f"{timeres}s",
            "--output-autos",
        ]
        hyperdrive_cmd = " ".join(hyperdrive_cmd_args)
        print(hyperdrive_cmd)
        result = run_hyperdrive(hyperdrive_cmd, ncpu=ncpu, verbose=verbose)
        if result != 0:
            print("Error occured in hyperdrive.")
            os.system(f"touch {msname}/.modeling_failed")
            return 1

        ########################
        # Importing model
        ########################
        print("Copy model data to ms.")
        with suppress_output():
            data_table = casatable()
            data_table.open(msname, nomodify=False)
            column_names = data_table.colnames()
            if "MODEL_DATA" not in column_names:
                data_table.close()
                setjy(
                    vis=msname,
                    standard="manual",
                    fluxdensity=[1, 0, 0, 0],
                    usescratch=True,
                )
                data_table.open(msname, nomodify=False)
            model_table = casatable()
            model_table.open(model_msname, nomodify=False)
            m_array = model_table.getcol("DATA")
            model_table.close()
            data_table.putcol("MODEL_DATA", m_array)
            data_table.close()
        del m_array
        print(f"Model import done in: {round(time.time()-starttime,2)}s")
        os.system(f"touch {msname}/.modeling_succeed")
        return 0
    except Exception:
        print(f"Model simulation and import failed for: {msname}.")
        traceback.print_exc()
        os.system(f"touch {msname}/.modeling_failed")
        return 1
    finally:
        os.system(f"rm -rf {model_msname}")


def run_all_modeling(
    mslist,
    dask_client,
    metafits,
    beamfile,
    sourcelist,
    ncpu,
    verbose,
    logger=None,
):
    """
    Run all modeling

    Parameters
    ----------
    mslist : list
        Measurement set list
    dask_client : dask. client
        Dask client
    metafits : str
        Metafits file
    beamfile : str
        MWA primary beam file
    sourcelist : str
        Source list file
    ncpu : int
        Number of CPU threads
    verbose : bool
        Verbose output

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

    mslist = list(set(mslist))
    ncpu = max(1, ncpu)
    if len(mslist) == 0:
        logger.critical("Please provide a valid measurement set list.")
        return 1, 0, 0
    else:
        succeed = 0
        failed = len(mslist)
    try:
        tasks = []
        for msname in mslist:
            tasks.append(
                delayed(import_hyperdrive_model_wrapper)(
                    msname,
                    metafits,
                    beamfile=beamfile,
                    sourcelist=sourcelist,
                    ncpu=ncpu,
                    verbose=verbose,
                )
            )
        logger.info("Start import modeling.")
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
        failed = sum(results)
        succeed = len(mslist) - failed
        logger.info(f"Total measurement setds: {len(mslist)}")
        logger.info(f"Total failure: {failed}")
        logger.info(f"Total success: {succeed}")
        if len(mslist) == failed:
            msg = 1
        else:
            msg = 0
    except Exception:
        logger.exception("Exception occured in importing model.", exc_info=True)
        msg = 1
    return msg, succeed, failed


def main(
    mslist,
    metafits,
    workdir,
    beamfile="",
    sourcelist="",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid="0",
    verbose=False,
    start_remote_log=False,
    dask_client=None,
):
    """
    Make dynamic spectra

    Parameters
    ----------
    mslist : str
        Measurement set list (comma separated)
    metafits : str
        Metafits file
    workdir : str
        Work directory
    beamfile : str, optional
        MWA beam file
    sourcelist : str, optional
        MWA global sky model (fits or ascii in wsclean format)
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
    verbose : bool, optional
        Verbose logs
    dask_client: dask.client, optional
        Dask client

    Returns
    -------
    int
        Success messsage
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

    ###########################
    # Hyperdrive container
    ###########################
    container_name = "paircarshyperdrive"
    container_present = check_udocker_container(container_name)
    if not container_present:
        logger.debug(f"Initializing {container_name}.")
        container_name = initialize_hyperdrive_container(
            name=container_name, verbose=True
        )
        if container_name is None:
            logger.critical(
                f"Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1

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
                "ds_plot", logfile, jobname=jobname, password=password
            )

    if dask_client is None:
        pass
    else:
        get_scheduler_name()

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
    logger.debug(f"Total coarse channels: {total_ncoarse}.")

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
            return 1
        scale_worker_and_wait(dask_cluster, dask_client, nworker)

    try:
        for banner in print_banner(
            "Starting visibility modeling.", no_print=True
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

        logger.info("#################################")
        logger.info(f"Total dask worker: {njobs}")
        logger.info(f"CPU per worker: {n_threads}")
        logger.info(f"Memory per worker: {mem_limit} GB")
        logger.info("#################################")
        msg, succeed, failed = run_all_modeling(
            mslist,
            dask_client,
            metafits,
            beamfile,
            sourcelist,
            n_threads,
            False,
            logger=logger,
        )
    except Exception:
        logger.exception("Exception occured in importing model.", exc_info=True)
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
            os.system(f"rm -rf {dask_dir}")
    return msg, succeed, failed


################################
# CLI interface
################################
def cli():
    parser = argparse.ArgumentParser(description="Simulate and import MWA visibilities")

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Name of the measurement sets (comma separated)",
    )
    basic_args.add_argument(
        "metafits",
        type=str,
        help="Name of the metafits file",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Work directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--beamfile",
        type=str,
        default="",
        help="Name of the MWA PB file",
    )
    adv_args.add_argument(
        "--sourcelist",
        type=str,
        default="",
        help="Source model file",
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
        help="CPU fraction",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Memory fraction",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg, _, _ = main(
        args.mslist,
        args.metafits,
        args.workdir,
        beamfile=args.beamfile,
        sourcelist=args.sourcelist,
        verbose=args.verbose,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        jobid=args.jobid,
    )
    return msg
