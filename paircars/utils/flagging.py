import types
import psutil
import numpy as np
import traceback
import glob
import os
import dask
from datetime import datetime as dt, timezone
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from .basic_utils import *
from .resource_utils import *
from .imaging import *


###############################
# Flagging related functions
################################
def flagsummary(msname, summary_file):
    """
    Save flag summary

    Parameters
    ----------
    msname : str
        Measurement set name
    summary_file : str
        Summary file name

    Returns
    -------
    str
        Summary file
    """
    from casatasks import flagdata

    with suppress_output():
        s = flagdata(vis=msname, mode="summary")
    allkeys = s.keys()
    with open(summary_file, "w") as f:
        f.write(f"Flag summary of: {msname}\n")
        for x in allkeys:
            try:
                for y in s[x].keys():
                    try:
                        flagged_percent = 100.0 * (
                            s[x][y]["flagged"] / s[x][y]["total"]
                        )
                        logstring = f"{x} {y} {flagged_percent}\n"
                        f.write(logstring)
                    except:
                        pass
            except:
                pass
    return summary_file


def do_flag_backup(msname, flagtype="flagdata"):
    """
    Take a flag backup

    Parameters
    ----------
    msname : str
        Measurement set name
    flagtype : str, optional
        Flag type
    """
    from casatools import agentflagger

    af = agentflagger()
    af.open(msname)
    versionlist = af.getflagversionlist()
    if len(versionlist) != 0:
        for version_name in versionlist:
            if flagtype in version_name:
                try:
                    version_num = (
                        int(version_name.split(":")[0].split(" ")[0].split("_")[-1]) + 1
                    )
                except BaseException:
                    version_num = 1
            else:
                version_num = 1
    else:
        version_num = 1
    dt_string = dt.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    af.saveflagversion(
        flagtype + "_" + str(version_num), "Flags autosave on " + dt_string
    )
    af.done()


def uvbin_flag(
    msname,
    uvbin_size=10,
    datacolumn="data",
    mode="rflag",
    threshold=10.0,
    flagbackup=True,
):
    """
    Perform uv-bin flag

    Parameters
    ----------
    msname : str
        Measurement set
    uvbin_size : float, optional
        UV-bin size in wavelength
    datacolumn : str, optional
        Data column
    mode : str, optional
        Flag mode (rflag or tfcrop)
    threshold : float, optional
        Flagging threshold
    flagbackup : bool, optional
        Flag backup
    """
    from casatasks import flagdata, flagmanager

    try:
        maxuv_m, maxuv_l = calc_maxuv(msname)
        if flagbackup:
            do_flag_backup(msname, flagtype="uvbin_flagdata")
        maxuv_l = int(maxuv_l)
        uvbin_size = int(uvbin_size)
        for i in range(0, maxuv_l, uvbin_size):
            try:
                with suppress_output():
                    if mode == "rflag":
                        flagdata(
                            vis=msname,
                            mode=mode,
                            datacolumn=datacolumn,
                            uvrange=f"{i}~{i+uvbin_size}lambda",
                            timedevscale=threshold,
                            freqdevscale=threshold,
                            ntime="2s",
                            flagbackup=False,
                        )
                    else:
                        flagdata(
                            vis=msname,
                            mode=mode,
                            datacolumn=datacolumn,
                            uvrange=f"{i}~{i+uvbin_size}lambda",
                            timecutoff=threshold,
                            freqcutoff=threshold,
                            flagbackup=False,
                        )
            except:
                pass
        return 0
    except Exception as e:
        traceback.print_exc()
        if flagbackup:
            with suppress_output():
                flagmanager(vis=msname, mode="restore", versionname="uvbin_flagdata_1")
                flagmanager(vis=msname, mode="delete", versionname="uvbin_flagdata_1")
        return 1


def get_unflagged_antennas(
    msname="",
    scan="",
    n_threads=-1,
):
    """
    Get unflagged antennas of a scan

    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : str
        Scans

    Returns
    -------
    numpy.array
        Unflagged antenna names
    numpy.array
        Flag fraction list
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    with suppress_output():
        flag_summary = flagdata(vis=msname, scan=str(scan), mode="summary")
    antenna_flags = flag_summary["antenna"]
    unflagged_antenna_names = []
    flag_frac_list = []
    for ant in antenna_flags.keys():
        flag_frac = antenna_flags[ant]["flagged"] / antenna_flags[ant]["total"]
        if flag_frac < 1.0:
            unflagged_antenna_names.append(ant)
            flag_frac_list.append(flag_frac)
    unflagged_antenna_names = np.array(unflagged_antenna_names)
    flag_frac_list = np.array(flag_frac_list)
    return unflagged_antenna_names, flag_frac_list


def get_chans_flag(
    msname="",
    field="",
    n_threads=-1,
):
    """
    Get flag/unflag channel list

    Parameters
    ----------
    msname : str
        Measurement set name
    field : str, optional
        Field name or ID

    Returns
    -------
    list
        Unflag channel list
    list
        Flag channel list
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    with suppress_output():
        summary = flagdata(vis=msname, field=field, mode="summary", spwchan=True)
    unflag_chans = []
    flag_chans = []
    for chan in summary["spw:channel"]:
        r = summary["spw:channel"][chan]
        chan_number = int(chan.split("0:")[-1])
        flag_frac = r["flagged"] / r["total"]
        if flag_frac == 1:
            flag_chans.append(chan_number)
        else:
            unflag_chans.append(chan_number)
    return unflag_chans, flag_chans


def calc_flag_fraction(
    msname="",
    field="",
    scan="",
    n_threads=-1,
):
    """
    Function to calculate the fraction of total data flagged.

    Parameters
    ----------
    msname : str
        Name of the measurement set
    field : str, optional
        Field names
    scan : str, optional
        Scan names

    Returns
    -------
    float
        Fraction of the total data flagged
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    with suppress_output():
        summary = flagdata(vis=msname, field=field, scan=scan, mode="summary")
    flagged_fraction = summary["flagged"] / summary["total"]
    return flagged_fraction


def flag_outside_uvrange(vis, uvrange, n_threads=-1, flagbackup=True):
    """
    Flag outside the given uv range

    Parameters
    ----------
    vis : str
        Measurement set name
    uvrange : str
        UV-range
    flagbackup : bool, optional
        Flag backup
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    try:
        if "lambda" in uvrange:
            islambda = True
            uvrange = uvrange.replace("lambda", "")
        else:
            islambda = False
        if "~" in uvrange:
            low, high = uvrange.split("~")
            if islambda:
                low = f"{low}lambda"
                high = f"{high}lambda"
            cmds = [
                {"mode": "manual", "uvrange": f"<{low}", "flagbackup": flagbackup},
                {"mode": "manual", "uvrange": f">{high}", "flagbackup": flagbackup},
            ]
        elif ">" in uvrange:
            low = uvrange.split(">")[-1]
            if islambda:
                low = f"{low}lambda"
            cmds = [
                {"mode": "manual", "uvrange": f"<{low}", "flagbackup": flagbackup},
            ]
        elif "<" in uvrange:
            high = uvrange.split("<")[-1]
            if islambda:
                high = f"{high}lambda"
            cmds = [
                {"mode": "manual", "uvrange": f">{high}", "flagbackup": flagbackup},
            ]
        else:
            cmds = []
        if len(cmds) > 0:
            for cmd in cmds:
                print(f"Flagging command: {cmd}")
                flagdata(vis=vis, **cmd)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def flag_quartical_table(caltable, threshold=10.0):
    """
    Flag quartical caltable

    Parameters
    ----------
    caltable : str
        Caltable name
    threshold : float
        Flagging threshold

    Returns
    -------
    str
        Flagged caltable name
    """
    caltable = caltable.rstrip("/")
    caltable_dirs = os.listdir(caltable)
    soltype = caltable_dirs[0]
    gains = xds_from_zarr(f"{caltable}::{soltype}")
    gain_data = gains[0].gains.to_numpy()  # Shape: ntime, nchan, nant, ndir, npol
    gain_flag = gains[0].gain_flags.to_numpy()
    pre_flags = np.nansum(gain_flag)
    gain_flag = gain_flag.astype("bool")
    gain_data[gain_flag] = np.nan
    d1 = gain_data[..., 1]
    d2 = gain_data[..., 2]
    d1_real = np.real(d1) - np.nanmean(np.real(d1))
    d1_imag = np.imag(d1) - np.nanmean(np.imag(d1))
    d2_real = np.real(d2) - np.nanmean(np.real(d2))
    d2_imag = np.imag(d2) - np.nanmean(np.imag(d2))
    d1_real_std = np.nanstd(d1_real)
    d1_imag_std = np.nanstd(d1_imag)
    d2_real_std = np.nanstd(d2_real)
    d2_imag_std = np.nanstd(d2_imag)
    pos = np.where(
        (d1_real > threshold * d1_real_std)
        | (d1_imag > threshold * d1_imag_std)
        | (d2_real > threshold * d2_real_std)
        | (d2_imag > threshold * d2_imag_std)
    )
    gain_data[np.isnan(gain_data)] = 1.0
    gain_flag[pos] = True
    gain_data[gain_flag] = 1.0
    gain_flag = gain_flag.astype("int")
    new_flags = np.nansum(gain_flag)
    gains[0].update(
        {
            "gain_flags": (
                ["gain_time", "gain_freq", "antenna", "direction"],
                gain_flag,
            )
        }
    )
    gains[0].update(
        {
            "gains": (
                ["gain_time", "gain_freq", "antenna", "direction", "correlation"],
                gain_data,
            )
        }
    )
    output_path = f"{caltable}::{soltype}"
    os.system(f"rm -rf {caltable}")
    write_xds_list = xds_to_zarr(gains, output_path)
    dask.compute(write_xds_list)
    return caltable


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
