import os
import numpy as np
import time
import traceback
import logging
import argparse
import subprocess
from contextlib import contextmanager
from casacore.tables import table as casacore_table, makecoldesc

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)


@contextmanager
def suppress_output():
    """
    Supress CASA terminal output
    """
    with open(os.devnull, "w") as fnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


def get_cachedir():
    """
    Get cache directory
    """
    homedir = os.environ.get("HOME")
    if homedir is None:
        homedir = os.path.expanduser("~")
    username = os.getlogin()
    cachedir = f"{homedir}/.solarpipe"
    os.makedirs(cachedir, exist_ok=True)
    os.makedirs(f"{cachedir}/pids", exist_ok=True)
    return cachedir


def get_datadir():
    """
    Get package data directory

    Returns
    -------
    str
        Data directory
    """
    cachedir = get_cachedir()
    if os.path.exists(f"{cachedir}/solarpipe_data_dir.txt") == False:
        return None
    with open(f"{cachedir}/solarpipe_data_dir.txt", "r") as f:
        datadir = f.read().strip()
    os.makedirs(datadir, exist_ok=True)
    return datadir


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
    datadir = get_datadir()
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
            data_table = casacore_table(msname + "/SPECTRAL_WINDOW", readonly=True)
            nchan = data_table.getcol("NUM_CHAN")[0]
            freqres = data_table.getcol("RESOLUTION")[0][0] / 10**3
            mid_freq = data_table.getcol("REF_FREQUENCY")[0] / 10**6
            data_table.close()
            data_table = casacore_table(msname + "/POLARIZATION", readonly=True)
            npol = data_table.getcol("NUM_CORR")[0]
            data_table.close()
            data_table = casacore_table(msname + "/ANTENNA", readonly=True)
            nant = data_table.nrows()
            data_table.close()
            data_table = casacore_table(msname, readonly=False)
            times = np.unique(data_table.getcol("TIME"))
            ntime = times.size
            timeres = data_table.getcol("EXPOSURE")[0]
            data_table.close()
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
        subprocess.run(hyperdrive_cmd, check=True)
        model_msname = msname.split(".ms")[0] + "_model.ms"
        ########################
        # Importing model
        ########################
        data_table = casacore_table(msname, readonly=False)
        model_table = casacore_table(model_msname, readonly=False)
        baselines = [*zip(data_table.getcol("ANTENNA1"), data_table.getcol("ANTENNA2"))]
        m_array = model_table.getcol("DATA")
        pos = np.array([i[0] != i[1] for i in baselines])
        model_array = np.empty((len(baselines), nchan, npol), dtype="complex")
        model_array[pos, ...] = m_array
        model_array[~pos, ...] = 0.0
        column_names = data_table.colnames()
        if "MODEL_DATA" in column_names:
            data_table.putcol("MODEL_DATA", model_array)
        else:
            coldesc = makecoldesc("MODEL_DATA", model_table.getcoldesc("DATA"))
            data_table.addcols(coldesc)
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


################################
# CLI interface
################################
def cli():
    parser = argparse.ArgumentParser(description="Simulate and import MWA visibilities")
    parser.add_argument(
        "--msname",
        required=True,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_argument(
        "--metafits",
        required=True,
        help="Name of the metafits file",
        metavar="String",
    )
    parser.add_argument(
        "--beamfile",
        type=str,
        default="",
        help="Name of the MWA PB file",
        metavar="String",
    )
    parser.add_argument(
        "--sourcelist",
        type=str,
        default="",
        help="Source model file",
        metavar="String",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=-1,
        help="Number of CPU threads to be used (default: all)",
        metavar="Integer",
    )
    args = parser.parse_args()
    msg = import_model(
        args.msname,
        args.metafits,
        args.beamfile,
        args.sourcelist,
        ncpu=args.ncpu,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\Visibility simulation is finished.\n###################\n"
    )
    os._exit(result)
