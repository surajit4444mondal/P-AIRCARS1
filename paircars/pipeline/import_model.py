import os
import numpy as np
import time
import sys
import dask
import traceback
import logging
import argparse
import subprocess
from dask import delayed
from casatasks import setjy
from casatools import table as casatable, msmetadata
from paircars.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


def import_model(msname, metafits, beamfile="", sourcelist="", ncpu=-1):
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
    """
    if datadir is None:
        print("Please setup P-AIRCARS first.")
        return
    if beamfile == "" or os.path.exists(beamfile) is not True:
        beamfile = f"{datadir}/mwa_full_embedded_element_pattern.h5"
    if sourcelist == "" or os.path.exists(sourcelist) is not True:
        sourcelist = f"{datadir}/GGSM.txt"
    if ncpu > 0:
        os.environ["RAYON_NUM_THREADS"] = str(ncpu)
    try:
        starttime = time.time()
        print(
            "#######################\nImporting model for ms:"
            + msname
            + "\n###################\n"
        )
        with suppress_output():
            msmd = msmetadata()
            msmd.open(msname)
            nchan = msmd.nchan(0)
            mid_freq = msmd.meanfreq(0, unit="MHz")
            freqres = msmd.chanres(0, unit="kHz")[0]
            npol = msmd.ncorrforpol()[0]
            nant = msmd.nantennas()
            times = msmd.timesforfield(0)
            ntime = len(times)
            timeres = msmd.exposuretime(scan=1)["value"]
            nrow = msmd.nrows()
            msmd.close()

        hyperdrive_cmd = [
            f"{datadir}/hyperdrive",
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
            "180",
            "-s",
            sourcelist,
            "-n",
            "2000",
            "--output-model-files",
            f"{msname.split('.ms')[0]}_model.ms",
            "--output-model-freq-average",
            f"{freqres}kHz",
            "--num-fine-channels",
            str(nchan),
            "--num-timesteps",
            str(ntime),
            "--output-model-time-average",
            f"{timeres}s",
        ]
        subprocess.run(
            hyperdrive_cmd,
            check=True,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
        )
        model_msname = msname.split(".ms")[0] + "_model.ms"
        ########################
        # Importing model
        ########################
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
            baselines = [
                *zip(data_table.getcol("ANTENNA1"), data_table.getcol("ANTENNA2"))
            ]
            m_array = model_table.getcol("DATA")
            pos = np.array([i[0] != i[1] for i in baselines])
            model_array = np.empty((npol, nchan, len(baselines)), dtype="complex")
            model_array[..., pos] = m_array
            model_array[..., ~pos] = 0.0
            data_table.putcol("MODEL_DATA", model_array)
            data_table.close()
            model_table.close()
        del m_array, model_array
        print(f"Model import done in: {round(time.time()-starttime,2)}s")
        return 0
    except Exception as e:
        print(f"Model simulation and import failed for: {msname}.")
        traceback.print_exc()
        return 1
    finally:
        os.system(f"rm -rf {msname.split('.ms')[0]}_model.ms")


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

    try:
        ms_sizes = [get_ms_size(ms) for ms in mslist]
        per_job_mem = 2 * max(ms_sizes)
        mem_limit = (psutil.virtual_memory().available * mem_frac) / (1024**3)
        max_njobs = int(mem_limit / per_job_mem)
        njobs = max(1, min(max_njobs, len(mslist)))
        ncpu = max(1, int(psutil.cpu_count() * cpu_frac / njobs))
        if len(mslist) > 0:
            tasks = []
            for msname in mslist:
                tasks.append(
                    delayed(import_model)(
                        msname,
                        metafits,
                        beamfile=beamfile,
                        sourcelist=sourcelist,
                        ncpu=ncpu,
                    )
                )
            print("Start import modeling...")
            results = []
            for i in range(0, len(tasks), njobs):
                batch = tasks[i : i + njobs]
                futures = dask_client.compute(batch)
                results.extend(dask_client.gather(futures))
            msg = 0
            for i in range(len(results)):
                if results[i] != 0:
                    print(f"Error in model import for ms: {mslist[i]}.")
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
        help="Name of the measurement sets (comma seperated)",
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
        help="CPU fraction",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Memory fraction",
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
        args.workdir,
        beamfile=args.beamfile,
        sourcelist=args.sourcelist,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\Visibility simulation is finished.\n###################\n"
    )
    os._exit(result)
