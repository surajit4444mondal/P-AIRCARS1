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
    with limit_threads(n_threads=n_threads):
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

    with limit_threads(n_threads=n_threads):
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
