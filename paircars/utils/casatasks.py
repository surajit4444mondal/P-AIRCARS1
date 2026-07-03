import numpy as np
import os
import traceback
import time
from casatools import table, msmetadata
from .basic_utils import suppress_output
from .resource_utils import limit_threads


#############################
# General CASA tasks
#############################
def check_scan_in_caltable(caltable, scan):
    """
    Check scan number available in caltable or not

    Parameters
    ----------
    caltable : str
        Name of the caltable
    scan : int
        Scan number

    Returns
    -------
    bool
        Whether scan is present in the caltable or not
    """
    tb = table()
    tb.open(caltable)
    scans = tb.getcol("SCAN_NUMBER")
    tb.close()
    if int(scan) in scans:
        return True
    else:
        return False


def reset_weights_and_flags(
    msname="",
    restore_flag=True,
    force_reset=False,
    n_threads=-1,
):
    """
    Reset weights and flags for the ms

    Parameters
    ----------
    msname : str
        Measurement set
    restore_flag : bool, optional
        Restore flags or not
    force_reset : bool, optional
        Force reset
    """
    n_threads = max(1, n_threads)
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    msname = msname.rstrip("/")
    if not os.path.exists(f"{msname}/.reset") or force_reset:
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        if restore_flag:
            print(f"Restoring flags of measurement set : {msname}")
            if os.path.exists(msname + ".flagversions"):
                os.system("rm -rf " + msname + ".flagversions")
            flagdata(vis=msname, mode="unflag", flagbackup=False)
        print(f"Resetting previous weights of the measurement set: {msname}")
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        tb = table()
        tb.open(msname, nomodify=False)
        colnames = tb.colnames()
        nrows = tb.nrows()
        if "WEIGHT" in colnames:
            print(f"Resetting weight column to ones of measurement set : {msname}.")
            weight = np.ones((npol, nrows))
            tb.putcol("WEIGHT", weight)
        if "SIGMA" in colnames:
            print(f"Resetting sigma column to ones of measurement set: {msname}.")
            sigma = np.ones((npol, nrows))
            tb.putcol("SIGMA", sigma)
        if "WEIGHT_SPECTRUM" in colnames:
            print(f"Removing weight spectrum of measurement set: {msname}.")
            tb.removecols("WEIGHT_SPECTRUM")
        if "SIGMA_SPECTRUM" in colnames:
            print(f"Removing sigma spectrum of measurement set: {msname}.")
            tb.removecols("SIGMA_SPECTRUM")
        tb.flush()
        tb.close()
        os.system(f"touch {msname}/.reset")
    return


def single_mstransform(
    msname="",
    outputms="",
    width=1,
    timebin="",
    datacolumn="DATA",
    spw="",
    corr="",
    timerange="",
    numsubms="auto",
    n_threads=-1,
):
    """
    Perform mstransform

    Parameters
    ----------
    msname : str
        Name of the measurement set
    outputms : str
        Output ms name
    width : int, optional
        Number of channels to average
    timebin : str, optional
        Time to average
    datacolumn : str, optional
        Data column to split
    spw : str, optional
        Spectral window
    corr : str, optional
        Correlation to split
    timerange : str, optional
        Time range
    n_threads : int, optional
        Number of CPU threads

    Returns
    -------
    str
        Output measurement set name
    """
    n_threads = max(1, n_threads)

    limit_threads(n_threads=n_threads)
    from casatasks import mstransform, initweights, flagdata

    if timebin == "" or timebin is None:
        timeaverage = False
    else:
        timeaverage = True
    if width > 1:
        chanaverage = True
    else:
        chanaverage = False
    outputms = outputms.rstrip("/")
    if os.path.exists(outputms):
        os.system("rm -rf " + outputms)
    if os.path.exists(outputms + ".flagversions"):
        os.system("rm -rf " + outputms + ".flagversions")
    try:
        print(f"Spliting ms: {msname}, Outputvis: {outputms}.")
        with suppress_output():
            mstransform(
                vis=msname,
                outputvis=outputms,
                spw=spw,
                timerange=timerange,
                datacolumn=datacolumn,
                correlation=corr,
                timeaverage=timeaverage,
                timebin=timebin,
                chanaverage=chanaverage,
                chanbin=int(width),
                nthreads=n_threads,
            )
        time.sleep(5)
        if os.path.exists(outputms):
            print(f"Initiating weights for ms: {outputms}")
            with suppress_output():
                initweights(vis=outputms, wtmode="ones", dowtsp=True)
                flagdata(
                    vis=outputms,
                    mode="clip",
                    clipzeros=True,
                    datacolumn="data",
                    flagbackup=False,
                )
        os.system(f"touch {outputms}/.splited")
        return 0, outputms
    except Exception:
        traceback.print_exc()
        if os.path.exists(outputms):
            os.system("rm -rf " + outputms)
        return 1, outputms


def calc_normzlized_crosscorr(data, flag, ant1, ant2, time):
    """
    Calculate normalized cross correlation

    Parameters
    ----------
    data : numpy.array
        Data array
    flag : numpy.array
        Flag array
    ant1 : numpy.array
        Antenna 1 array
    ant2 : numpy.array
        Antenna 2 array
    time : numpy.array
        Time array

    Returns
    -------
    numpy.array
        Normalized data (only if writeto_file is False)
    numpy.array
        New flag (only if writeto_file is False)
    """
    nrow = data.shape[-1]
    # Identify autocorrelation rows
    auto_mask = ant1 == ant2
    auto_rows = np.where(auto_mask)[0]
    auto_ant = ant1[auto_mask]
    auto_time = time[auto_mask]
    # Build lookup (vectorized via sorting)
    # Combine (time, antenna) into structured array
    auto_keys = np.core.records.fromarrays([auto_time, auto_ant], names="time,ant")
    cross_keys_1 = np.core.records.fromarrays([time, ant1], names="time,ant")
    cross_keys_2 = np.core.records.fromarrays([time, ant2], names="time,ant")
    sort_idx = np.argsort(auto_keys)
    auto_keys_sorted = auto_keys[sort_idx]
    auto_rows_sorted = auto_rows[sort_idx]

    # Find matching autos using searchsorted
    def match_keys(query_keys):
        idx = np.searchsorted(auto_keys_sorted, query_keys)
        idx = np.clip(idx, 0, len(auto_keys_sorted) - 1)
        matched = auto_keys_sorted[idx] == query_keys
        out = np.full(nrow, -1, dtype=int)
        out[matched] = auto_rows_sorted[idx][matched]
        return out

    idx1 = match_keys(cross_keys_1)
    idx2 = match_keys(cross_keys_2)
    # valid rows where both autos exist and not autocorr
    valid = (idx1 >= 0) & (idx2 >= 0) & (ant1 != ant2)
    # Vectorized normalization
    norm = np.zeros_like(data, dtype=np.complex64)
    auto1_xx = np.abs(data[0, :, idx1[valid]])
    auto2_xx = np.abs(data[0, :, idx2[valid]])
    auto1_yy = np.abs(data[-1, :, idx1[valid]])
    auto2_yy = np.abs(data[-1, :, idx2[valid]])
    npol = data.shape[0]
    with suppress_output():
        if npol == 2:
            for p in range(npol):
                if p == 0:
                    denom = np.sqrt(auto1_xx * auto2_xx)
                else:
                    denom = np.sqrt(auto1_yy * auto2_yy)
                norm[p, :, valid] = data[p, :, valid] / denom
        else:
            for p in range(npol):
                if p == 0:
                    denom = np.sqrt(auto1_xx * auto2_xx)
                elif p == 1:
                    denom = np.sqrt(auto1_xx * auto2_yy)
                elif p == 2:
                    denom = np.sqrt(auto1_yy * auto2_xx)
                else:
                    denom = np.sqrt(auto1_yy * auto2_yy)
                norm[p, :, valid] = data[p, :, valid] / denom
    # Clean up
    flag[np.isnan(norm)] = True
    norm[np.isnan(norm)] = 0.0 + 0.0j
    return norm, flag


def normalized_crosscorr_ms(msname, datacolumn="DATA"):
    """
    Perform normalized cross-correlation of the measurement set

    Parameters
    ----------
    msname : str
        Measurement set
    datacolumn : str, optional
        Data column to normalize

    Returns
    -------
    str
        Normalized measurement set
    """
    try:
        msname = msname.rstrip("/")
        outfile = f"{msname}.norm"
        if os.path.exists(outfile):
            os.system(f"rm -rf {outfile}")
        os.system(f"cp -r {msname} {outfile}")
        with suppress_output():
            tb = table()
            tb.open(outfile, nomodify=False)
            datacolumn = datacolumn.upper()
            if datacolumn == "CORRECTED":
                datacolumn = "CORRECTED_DATA"
            if datacolumn == "MODEL":
                datacolumn = "MODEL_DATA"
            if datacolumn not in tb.colnames():
                datacolumn = "DATA"
            data = tb.getcol(datacolumn)  # (npol, nchan, nrow)
            flag = tb.getcol("FLAG")
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            time = tb.getcol("TIME")
            norm, flag = calc_normzlized_crosscorr(data, flag, ant1, ant2, time)
            tb.putcol(datacolumn, norm)
            tb.putcol("FLAG", flag)
            tb.flush()
            tb.close()
        return outfile
    except Exception:
        traceback.print_exc()
        return msname
