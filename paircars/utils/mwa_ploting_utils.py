import astropy.units as u
import logging
import numpy as np
import warnings
import glob
import requests
import os
import traceback
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from sunpy.timeseries import TimeSeries
from aiapy.calibrate import update_pointing, register, correct_degradation
from astropy.visualization import ImageNormalize, PowerStretch, LogStretch
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, solar_system_ephemeris
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from casatools import msmetadata
from datetime import datetime as dt
from PIL import Image
from sunpy.coordinates import SphericalScreen, frames, sun
from matplotlib.colors import ListedColormap
from matplotlib import cm
from sunpy.map import make_fitswcs_header
from collections import namedtuple
from daskms.experimental.zarr import xds_from_zarr
from .basic_utils import (
    mjdsec_to_timestamp,
    timestamp_to_mjdsec,
    get_datadir,
    interpolate_nans,
)
from .image_utils import calc_solar_image_stat, cutout_image
from .ms_metadata import (
    get_ms_scan_size,
    check_datacolumn_valid,
    get_ms_scans,
)
from .resource_utils import drop_cache
from .udocker_utils import (
    run_shadems,
    check_udocker_container,
    initialize_wsclean_container,
)
from .calibration import get_quartical_soltype

warnings.simplefilter("ignore", VerifyWarning)
warnings.simplefilter("ignore", category=FITSFixedWarning)
logging.getLogger("sunpy").setLevel(logging.ERROR)
logging.getLogger("reproject.common").setLevel(logging.WARNING)

#####################################
# Sun position related
#####################################
datadir = get_datadir()
try:
    solar_system_ephemeris.set(f"{datadir}/de440s")
except Exception:
    solar_system_ephemeris.set("builtin")


#################################
# Plotting related functions
#################################
def plot_ms_diagnostics(
    msname,
    outdir,
    ncpu=1,
    total_mem=1,
    verbose=False,
):
    """
    Plot diagonistics plots for measurement set

    Parameters
    ----------
    msname : str
        Measurement set
    outdir : str, optional
        Output directory
    ncpu : int, optional
        Number of CPU threads
    total_mem : float, optional
        Total memory in GB
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    list
        Output plot file list
    """
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))

    ncpu = max(1, ncpu)
    total_mem = max(1, total_mem)

    from casatools import ms as casamstool

    output_pdf = f"{os.path.basename(msname).split('.ms')[0]}_plots"
    suffix = os.path.basename(msname).split(".ms")[0]
    os.makedirs(outdir, exist_ok=True)

    msname = msname.rstrip("/")
    mstool = casamstool()
    mstool.open(msname)
    nrow = mstool.nrow()
    mstool.close()
    msmd = msmetadata()
    msmd.open(msname)
    npol = msmd.ncorrforpol()[0]
    scan_list = msmd.scannumbers()
    msmd.close()
    scan_sizes = [get_ms_scan_size(msname, scan) for scan in scan_list]

    max_scan_size = max(scan_sizes)
    frac_chunk = min(1, total_mem / max_scan_size)
    nchunk = int(nrow * frac_chunk)
    output_pdf_list = []

    container_name = "paircarsshadems"
    container_present = check_udocker_container(container_name)
    if not container_present:
        print(f"Initializing {container_name}...")
        container_name = initialize_wsclean_container(name=container_name, verbose=True)
        if container_name is None:
            print(
                f"Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1, []
    try:
        #######################
        # Commands to run
        ######################
        cmds = []
        # Define correlation groups
        corr_sets = [
            ("XX,YY", True),  # parallel hands, always plotted
            ("XY,YX", npol == 4),  # cross hands, only if 4 pols
        ]

        # Define y-axis modes and labels
        plot_types = {
            "amp": "Amplitude",
            "phase": "Phase(deg)",
            "real": "Real",
            "imag": "Imaginary",
        }

        # Define x-axis settings
        xaxes = {"uv": ("UV(m)",), "FREQ": ("Frequency(GHz)",), "TIME": ("Time",)}

        # Determine ploting coloumn
        cols = []
        if check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA"):
            cols.append("CORRECTED_DATA")
            if check_datacolumn_valid(msname, datacolumn="MODEL_DATA"):
                cols.append("CORRECTED_DATA-MODEL_DATA")
        else:
            cols.append("DATA")
            if check_datacolumn_valid(msname, datacolumn="MODEL_DATA"):
                cols.append("DATA-MODEL_DATA")

        for corr, do_plot in corr_sets:
            if not do_plot:
                continue
            for yaxis, ylabel in plot_types.items():
                for xaxis, (xlabel,) in xaxes.items():
                    for col in cols:
                        cmds.append(
                            f"shadems --no-lim-save --xaxis {xaxis} --yaxis {yaxis} "
                            f"--col {col} -j {ncpu} -z {nchunk} "
                            f"--xlabel '{xlabel}' --ylabel '{ylabel}' "
                            f"--corr {corr} --colour-by CORR --iter-scan --iter-field "
                            f"--dmap tab10 -s {suffix} {msname}"
                        )

        print(f"Making plots of: {msname}")
        for cmd in cmds:
            run_shadems(cmd, verbose=verbose)

        for yaxis, ylabel in plot_types.items():
            #########################
            # Making plots
            #########################
            pngs = glob.glob(f"{mspath}/*{yaxis}*{suffix}*.png")
            outfile = f"{outdir}/{output_pdf}_{yaxis}.pdf"
            if len(pngs) > 0:
                images = []
                for image in pngs:
                    images.append(Image.open(image).convert("RGB"))
                images[0].save(outfile, save_all=True, append_images=images[1:])
                output_pdf_list.append(outfile)
                for png in pngs:
                    os.system(f"rm -rf {png}")
            else:
                print(f"No plot for {ylabel} is made.")

        if len(output_pdf_list) > 0:
            return 0, output_pdf_list
        else:
            print("No plot is made.")
            return 1, []
    except Exception:
        traceback.print_exc()
    finally:
        drop_cache(msname)
        os.system("rm -rf log-shadems.txt")


def plot_quartical_tables(caltables, output_prefix, ncols=3, nrows=3):
    """
    Plot quartical gaintables

    Parameters
    ----------
    caltables : list
        Quartical caltable list
    output_prefix : str
        Output files names prefix
    ncols : int, optional
        Number of columns in the plot
    nrows : int, optional
        Number of rows in the plot

    Returns
    -------
    int
        Success message
    list
        Plot file names
    """
    try:
        all_freqs = []
        all_gains = []
        for caltable in caltables:
            caltable = caltable.rstrip("/")
            soltypes = get_quartical_soltype(caltable)
            if len(soltypes) == 0:
                print("No solution is present. Not performing interpolation.")
                pass
            else:
                soltype = soltypes[0]
                gains = xds_from_zarr(f"{caltable}::{soltype}")
                freqs = gains[0].gain_freq.to_numpy()
                gain_data = gains[
                    0
                ].gains.to_numpy()  # Shape: ntime, nchan, nant, ndir, npol
                gain_flag = gains[0].gain_flags.to_numpy()
                gain_flag = gains[0].gain_flags.values.astype(bool)
                gain_data[gain_flag, :] = np.nan
                all_freqs.append(freqs)
                all_gains.append(gain_data)

        all_freqs = np.concatenate(all_freqs, axis=0)
        all_gains = np.concatenate(all_gains, axis=1)
        all_freqs = all_freqs.flatten()
        pos = np.argsort(all_freqs)
        all_freqs_sorted = all_freqs[pos]
        all_gains_sorted = all_gains[:, pos, ...]
        plots_per_fig = ncols * nrows
        max_ant = all_gains.shape[2]
        all_ants = np.arange(0, max_ant)
        plots_per_fig = min(max_ant, ncols * nrows)
        if plots_per_fig < ncols * nrows:
            ncols = nrows = int(np.sqrt(plots_per_fig))
        all_gains_sorted = np.nanmean(all_gains_sorted, axis=0)[..., 0, :]
        output_pdfs = []
        copolar_gains = np.take(all_gains_sorted, [0, 3], axis=-1)
        crosspolar_gains = np.take(all_gains_sorted, [1, 2], axis=-1)

        for p in range(2):
            if p == 0:
                all_gains_sub = copolar_gains
                polar = "Copolar"
                pols = [r"$G_\mathrm{X}$", r"$G_\mathrm{Y}$"]
            else:
                all_gains_sub = crosspolar_gains
                polar = "Crosspolar"
                pols = [r"$D_\mathrm{X}$", r"$D_\mathrm{Y}$"]
            for quantity in ["amp", "phase"]:
                out_files = []
                for idx in range(0, max_ant, plots_per_fig):
                    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
                    if quantity == "amp":
                        fig.suptitle(
                            f"Frequency vs Gain Amplitude, {polar}", fontsize=14
                        )
                        x = np.abs(np.array(all_gains_sub))
                        miny = np.nanmin(x)
                        maxy = np.nanmax(x)
                        pad = 0.1 * (maxy - miny)
                    else:
                        fig.suptitle(f"Frequency vs Gain Phase, {polar}", fontsize=14)
                        x = np.angle(np.array(all_gains_sub), deg=True)
                        miny = np.nanmin(x)
                        maxy = np.nanmax(x)
                        pad = 0.1 * (maxy - miny)
                    if axes is None:
                        return 1, []

                    if not isinstance(axes, np.ndarray):
                        axes = np.array([axes])
                    axes = axes.flatten()
                    for i, ant in enumerate(all_ants[idx : idx + plots_per_fig]):
                        ax = axes[i]
                        for j in range(2):  # loop over polarizations
                            if j == 0:
                                c = "r"
                            else:
                                c = "k"
                            label = f"{pols[j]}"
                            if quantity == "amp":
                                ax.scatter(
                                    all_freqs_sorted,
                                    np.abs(all_gains_sub[:, ant, j]),
                                    label=label,
                                    color=c,
                                    s=14,
                                )
                                ax.set_ylabel("Gain Amplitude", fontsize=14)
                            else:
                                ax.scatter(
                                    all_freqs_sorted,
                                    np.angle(all_gains_sub[:, ant, j], deg=True),
                                    label=label,
                                    color=c,
                                    s=14,
                                )
                                ax.set_ylabel("Gain Phase (degree)", fontsize=14)
                        ax.set_title(f"Antenna {ant+1}", fontsize=14)
                        ax.set_xlabel("Frequency (MHz)", fontsize=14)
                        ax.legend(fontsize=10, ncol=2, loc="upper right")
                        ax.set_ylim(miny - pad, maxy + pad)
                    for j in range(i + 1, plots_per_fig):
                        fig.delaxes(axes[j])
                    plt.tight_layout(rect=[0, 0, 1, 0.99])
                    savefile = f"{output_prefix}_gain_{quantity}_batch_{idx // plots_per_fig + 1}.png"
                    plt.savefig(savefile)
                    plt.close(fig)
                    out_files.append(savefile)
                images = []
                output_pdf = f"{output_prefix}_freqs_vs_{quantity}_{polar.lower()}.pdf"
                if os.path.exists(output_pdf):
                    os.system(f"rm -rf {output_pdf}")
                for image in out_files:
                    images.append(Image.open(image).convert("RGB"))
                images[0].save(output_pdf, save_all=True, append_images=images[1:])
                for outpng in out_files:
                    os.system(f"rm -rf {outpng}")
                output_pdfs.append(output_pdf)
        return 0, output_pdfs
    except Exception:
        traceback.print_exc()
        return 1, []


def plot_G_jones_time_vs_gain(
    all_times,
    all_gains,
    all_ants,
    all_ant_names,
    ncols,
    nrows,
    pols,
    prefix,
    output_prefix,
    quantities=["amp", "phase"],
):
    """
    Plot time vs. gain
    """
    plots_per_fig = ncols * nrows
    max_ant = np.nanmax(np.array(all_ants))
    plots_per_fig = min(max_ant, ncols * nrows)
    if plots_per_fig < ncols * nrows:
        ncols = nrows = int(np.sqrt(plots_per_fig))
    min_time = np.nanmin(np.array(all_times))
    start_timestamp = mjdsec_to_timestamp(min_time)
    output_pdfs = []
    for quantity in quantities:
        out_files = []
        for idx in range(0, max_ant, plots_per_fig):
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
            if quantity == "amp":
                fig.suptitle(
                    f"Time vs Gain Amplitude, Start time: {start_timestamp}",
                    fontsize=14,
                )
                x = np.abs(np.array(all_gains))
                miny = np.nanmin(x)
                maxy = np.nanmax(x)
                pad = 0.1 * (maxy - miny)
            else:
                fig.suptitle(
                    f"Time vs Gain Phase, Start time: {start_timestamp}", fontsize=14
                )
                x = np.angle(np.array(all_gains), deg=True)
                miny = np.nanmin(x)
                maxy = np.nanmax(x)
                pad = 0.1 * (maxy - miny)
            axes = axes.flatten()
            for n in range(len(all_ants)):
                ants = all_ants[n]
                ant_names = all_ant_names[n]
                times = all_times[n]
                gains = all_gains[n]
                for i, ant in enumerate(ants[idx : idx + plots_per_fig]):
                    ax = axes[i]
                    for j in range(2):  # loop over polarizations
                        if j == 0:
                            c = "r"
                        else:
                            c = "k"
                        if n == 0:
                            label = f"Pol: {pols[j]}"
                        else:
                            label = None
                        if quantity == "amp":
                            ax.scatter(
                                times - min_time,
                                np.abs(gains[j, 0, :, ant]),
                                label=label,
                                color=c,
                                s=14,
                            )
                            if n == 0:
                                ax.set_ylabel("Gain Amplitude", fontsize=14)
                        else:
                            ax.scatter(
                                times - min_time,
                                np.angle(gains[j, 0, :, ant], deg=True),
                                label=label,
                                color=c,
                                s=14,
                            )
                            if n == 0:
                                ax.set_ylabel("Gain Phase (degree)", fontsize=14)
                    if n == 0:
                        ax.set_title(f"Antenna {ant+1}, {ant_names[ant]}", fontsize=14)
                        ax.set_xlabel("Time (s)", fontsize=14)
                        ax.legend(fontsize=10, ncol=2, loc="upper right")
                    ax.set_ylim(miny - pad, maxy + pad)
            for j in range(i + 1, plots_per_fig):
                fig.delaxes(axes[j])
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            savefile = f"{prefix}_gain_{quantity}_batch_{idx // plots_per_fig + 1}.png"
            plt.savefig(savefile)
            plt.close(fig)
            out_files.append(savefile)
        images = []
        output_pdf = f"{output_prefix}_times_vs_{quantity}.pdf"
        if os.path.exists(output_pdf):
            os.system(f"rm -rf {output_pdf}")
        for image in out_files:
            images.append(Image.open(image).convert("RGB"))
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        for outpng in out_files:
            os.system(f"rm -rf {outpng}")
        output_pdfs.append(output_pdf)
    return output_pdfs


def plot_B_jones_freq_vs_gain(
    all_freqs,
    all_gains,
    all_ants,
    all_ant_names,
    ncols,
    nrows,
    pols,
    prefix,
    output_prefix,
    quantities=["amp", "phase"],
    plot_all_ants=True,
):
    """
    Plot freq vs. gain
    """
    plots_per_fig = ncols * nrows
    if plot_all_ants:
        max_ant = np.nanmax(np.array(all_ants))
    else:
        max_ant = 1
    plots_per_fig = min(max_ant, ncols * nrows)
    if plots_per_fig < ncols * nrows:
        ncols = nrows = int(np.sqrt(plots_per_fig))
    output_pdfs = []
    for quantity in quantities:
        out_files = []
        for idx in range(0, max_ant, plots_per_fig):
            if plot_all_ants:
                fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            if quantity == "amp":
                fig.suptitle("Frequency vs Gain Amplitude", fontsize=14)
                x = np.abs(np.array(all_gains))
                miny = np.nanmin(x)
                maxy = np.nanmax(x)
                pad = 0.1 * (maxy - miny)
            else:
                fig.suptitle("Frequency vs Gain Phase", fontsize=14)
                x = np.angle(np.array(all_gains), deg=True)
                miny = np.nanmin(x)
                maxy = np.nanmax(x)
                pad = 0.1 * (maxy - miny)
            if plot_all_ants:
                axes = axes.flatten()
            else:
                pass
            for n in range(len(all_ants)):
                if plot_all_ants:
                    ants = all_ants[n]
                    ant_names = all_ant_names[n]
                else:
                    ants = [all_ants[n][0]]
                    ant_names = [all_ant_names[n][0]]
                freqs = all_freqs[n]
                gains = all_gains[n]
                for i, ant in enumerate(ants[idx : idx + plots_per_fig]):
                    if plot_all_ants:
                        ax = axes[i]
                    else:
                        ax = axes
                    for j in range(2):  # loop over polarizations
                        if j == 0:
                            c = "r"
                        else:
                            c = "k"
                        if n == 0:
                            label = f"Pol: {pols[j]}"
                        else:
                            label = None
                        if quantity == "amp":
                            ax.scatter(
                                freqs,
                                np.abs(np.nanmean(gains[j, :, :, ant], axis=1)),
                                label=label,
                                color=c,
                                s=14,
                            )
                            if n == 0:
                                ax.set_ylabel("Gain Amplitude", fontsize=14)
                        else:
                            ax.scatter(
                                freqs,
                                np.angle(
                                    np.nanmean(gains[j, :, :, ant], axis=1), deg=True
                                ),
                                label=label,
                                color=c,
                                s=14,
                            )
                            if n == 0:
                                ax.set_ylabel("Gain Phase (degree)", fontsize=14)
                    if n == 0:
                        if plot_all_ants:
                            ax.set_title(
                                f"Antenna {ant+1}, {ant_names[ant]}", fontsize=14
                            )
                        ax.set_xlabel("Frequency (MHz)", fontsize=14)
                        ax.legend(fontsize=10, ncol=2, loc="upper right")
                    ax.set_ylim(miny - pad, maxy + pad)
            for j in range(i + 1, plots_per_fig):
                fig.delaxes(axes[j])
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            savefile = f"{prefix}_gain_{quantity}_batch_{idx // plots_per_fig + 1}.png"
            plt.savefig(savefile)
            plt.close(fig)
            out_files.append(savefile)
        images = []
        output_pdf = f"{output_prefix}_freqs_vs_{quantity}.pdf"
        if os.path.exists(output_pdf):
            os.system(f"rm -rf {output_pdf}")
        for image in out_files:
            images.append(Image.open(image).convert("RGB"))
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        for outpng in out_files:
            os.system(f"rm -rf {outpng}")
        output_pdfs.append(output_pdf)
    return output_pdfs


def plot_caltable_diagnostics(
    caltables, outfile_prefix, quantities=["amp", "phase"], plot_all_ants=True
):
    """
    Plot diagonistic plot of casa caltables

    Parameters
    ----------
    caltables : list
        Caltable names
    outfile_prefix : str
        Output plot file name prefix
    quantities : list
        Quantities to plot (amp, phase)
    plot_all_ants : bool, optional
        Plot all antennas or only the single one

    Returns
    -------
    int
        Success messsage
    str
        Output file
    """
    from casatools import table

    pols = ["X", "Y"]
    ncols = 3
    nrows = 3
    tb = table()
    outdir = os.path.dirname(outfile_prefix)
    os.makedirs(outdir, exist_ok=True)
    try:
        all_freqs = []
        all_times = []
        all_ants = []
        all_gains = []
        all_ant_names = []
        final_caltables = []
        last_caltype = ""
        for caltable in caltables:
            tb.open(caltable)
            cal_type = tb.getkeywords()["VisCal"]
            tb.close()
            if cal_type == last_caltype or last_caltype == "":
                tb.open(f"{caltable}/SPECTRAL_WINDOW")
                freqs = tb.getcol("CHAN_FREQ") / 10**6  # In MHz
                tb.close()
                tb.open(f"{caltable}/ANTENNA")
                ant_names = tb.getcol("NAME")
                tb.close()
                tb.open(caltable)
                gains = tb.getcol("CPARAM")
                flags = tb.getcol("FLAG")
                gains[flags] = np.nan + 1j * np.nan
                ants = np.unique(tb.getcol("ANTENNA1"))
                times = np.unique(tb.getcol("TIME"))
                np.nanmax(ants) + 1
                tb.close()
                ntime = len(times)
                shape = gains.shape
                gains = gains.reshape(shape[0], shape[1], ntime, shape[2] // ntime)
                all_freqs.append(freqs)
                all_times.append(times)
                all_gains.append(gains)
                all_ants.append(ants)
                all_ant_names.append(ant_names)
                last_caltype = cal_type
                final_caltables.append(caltable)
        final_cal_type = last_caltype
        if final_cal_type == "G Jones":
            output_pdfs = plot_G_jones_time_vs_gain(
                all_times,
                all_gains,
                all_ants,
                all_ant_names,
                ncols,
                nrows,
                pols,
                "GJones",
                outfile_prefix,
                quantities=quantities,
            )
        elif final_cal_type == "B Jones":
            output_pdfs = plot_B_jones_freq_vs_gain(
                all_freqs,
                all_gains,
                all_ants,
                all_ant_names,
                ncols,
                nrows,
                pols,
                "BJones",
                outfile_prefix,
                quantities=quantities,
                plot_all_ants=plot_all_ants,
            )
        else:
            print(f"{final_cal_type} is not implemented.")
            output_pdfs = []
        return 0, output_pdfs
    except Exception:
        traceback.print_exc()
        return 1, []
    finally:
        if len(caltables) > 0:
            for caltable in caltables:
                drop_cache(caltable)


def get_mwamap(fits_image, pol="I", do_sharpen=False):
    """
    Make MWA sunpy map

    Parameters
    ----------
    fits_image : str
        MWA fits image
    pol : str, optional
        Stokes plane
    do_sharpen : bool, optional
        Sharpen the image

    Returns
    -------
    sunpy.map
        Sunpy map
    """
    from scipy.ndimage import gaussian_filter
    from sunpy.map import make_fitswcs_header
    from astropy.coordinates import EarthLocation

    logging.getLogger("sunpy").setLevel(logging.ERROR)

    MWALAT = -26.703319  # degrees
    MWALON = 116.670815  # degrees
    MWAALT = 377.0  # meters
    mwa_hdu = fits.open(fits_image)  # Opening MWA fits file
    mwa_header = mwa_hdu[0].header  # mwa header
    mwa_data = mwa_hdu[0].data
    if mwa_data.ndim == 4:
        if pol == "I":
            mwa_data = mwa_data[0, 0, ...]  # mwa data
        elif pol == "Q":
            if mwa_data.shape[0] > 1:
                mwa_data = mwa_data[1, 0, ...]
            else:
                mwa_data = mwa_data[0, 1, ...]
        elif pol == "U":
            if mwa_data.shape[0] > 1:
                mwa_data = mwa_data[2, 0, ...]
            else:
                mwa_data = mwa_data[0, 2, ...]
        elif pol == "V":
            if mwa_data.shape[0] > 1:
                mwa_data = mwa_data[3, 0, ...]
            else:
                mwa_data = mwa_data[0, 3, ...]
    elif mwa_data.ndim == 3:
        if pol == "I":
            mwa_data = mwa_data[0, ...]  # mwa data
        elif pol == "Q":
            mwa_data = mwa_data[1, ...]
        elif pol == "U":
            mwa_data = mwa_data[2, ...]
        elif pol == "V":
            mwa_data = mwa_data[3, ...]

    if mwa_header["CTYPE3"] == "FREQ":
        frequency = mwa_header["CRVAL3"] * u.Hz
    elif mwa_header["CTYPE4"] == "FREQ":
        frequency = mwa_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    try:
        mwa_header["BUNIT"]
    except BaseException:
        pass
    obstime = Time(mwa_header["date-obs"])
    mwapos = EarthLocation(lat=MWALAT * u.deg, lon=MWALON * u.deg, height=MWAALT * u.m)
    # Converting into GCRS coordinate
    mwa_gcrs = SkyCoord(mwapos.get_gcrs(obstime))
    reference_coord = SkyCoord(
        mwa_header["crval1"] * u.Unit(mwa_header["cunit1"]),
        mwa_header["crval2"] * u.Unit(mwa_header["cunit2"]),
        frame="gcrs",
        obstime=obstime,
        obsgeoloc=mwa_gcrs.cartesian,
        obsgeovel=mwa_gcrs.velocity.to_cartesian(),
        distance=mwa_gcrs.hcrs.distance,
    )
    reference_coord_arcsec = reference_coord.transform_to(
        frames.Helioprojective(observer=mwa_gcrs)
    )
    cdelt1 = (np.abs(mwa_header["cdelt1"]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(mwa_header["cdelt2"]) * u.deg).to(u.arcsec)
    P1 = sun.P(obstime)  # Relative rotation angle
    new_mwa_header = make_fitswcs_header(
        mwa_data,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [mwa_header["crpix1"] - 1, mwa_header["crpix2"] - 1] * u.pixel
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-P1,
        wavelength=frequency.to(u.MHz).round(2),
        observatory="MWA",
    )
    if do_sharpen:
        blurred = gaussian_filter(mwa_data, sigma=10)
        mwa_data = mwa_data + (mwa_data - blurred)
    mwa_map = Map(mwa_data, new_mwa_header)
    mwa_map_rotate = mwa_map.rotate()
    return mwa_map_rotate


def save_in_hpc(fits_image, outdir="", xlim=[], ylim=[]):
    """
    Save solar image in helioprojective coordinates

    Parameters
    ----------
    fits_image : str
        FITS image name
    outdir : str, optional
        Output directory
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond

    Returns
    -------
    str
        FITS image in helioprojective coordinate
    """
    logging.getLogger("sunpy").setLevel(logging.ERROR)
    fits_header = fits.getheader(fits_image)
    org_data = fits.getdata(fits_image)
    data_shape = org_data.shape
    del org_data
    if fits_header["NAXIS4"] == 4 or fits_header["NAXIS3"] == 4:
        stokes = "IQUV"
    else:
        stokes = "I"
    pol_list = list(stokes)
    if outdir == "":
        outdir = os.path.dirname(os.path.abspath(fits_image))
    outfile = f"{outdir}/{os.path.basename(fits_image).split('.fits')[0]}_HPC.fits"
    if os.path.exists(outfile):
        os.system(f"rm -rf {outfile}")
    for p in range(len(pol_list)):
        pol = pol_list[p]
        mwamap = get_mwamap(fits_image, pol=pol)
        if len(xlim) == 2 and len(ylim) == 2:
            top_right = SkyCoord(
                xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=mwamap.coordinate_frame
            )
            bottom_left = SkyCoord(
                xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=mwamap.coordinate_frame
            )
            mwamap = mwamap.submap(bottom_left, top_right=top_right)
        if p == 0:
            mwamap.save(outfile, filetype="fits")
            data = fits.getdata(outfile)
            data = data[np.newaxis, np.newaxis, ...]
        if data.ndim == 4:
            if data_shape[0] == 4:
                if p == 0:
                    data = np.repeat(data, 4, axis=0)
                data[p, 0, ...] = mwamap.data
            else:
                if p == 0:
                    data = np.repeat(data, 4, axis=1)
                data[0, p, ...] = mwamap.data
        elif data.ndim == 3:
            if p == 0:
                data = np.repeat(data, 4, axis=0)
            data[p, ...] = mwamap.data
        else:
            data = mwamap.data
    hpc_header = fits.getheader(outfile)
    for key in [
        "NAXIS",
        "NAXIS3",
        "NAXIS4",
        "BUNIT",
        "CTYPE3",
        "CRPIX3",
        "CRVAL3",
        "CDELT3",
        "CUNIT3",
        "CTYPE4",
        "CRPIX4",
        "CRVAL4",
        "CDELT4",
        "CUNIT4",
        "DATE-OBS",
        "AUTHOR",
        "PIPELINE",
        "BAND",
        "BMAJ",
        "BMIN",
        "BPA",
        "MAX",
        "MIN",
        "RMS",
        "SUM",
        "MEAN",
        "MEDIAN",
        "RMSDYN",
        "MIMADYN",
        "CALAPP",
        "POLSELF",
        "LEAKUNIT",
        "QLEAK",
        "ULEAK",
        "VLEAK",
    ]:
        if key in fits_header:
            hpc_header[key] = fits_header[key]
    fits.writeto(outfile, data=data, header=hpc_header, overwrite=True)
    return outfile


def plot_in_hpc(
    fits_image,
    draw_limb=False,
    extensions=["png"],
    outdirs=[],
    plot_range=[],
    power=0.5,
    xlim=[-3200, 3200],
    ylim=[-3200, 3200],
    contour_levels=[],
    showgui=False,
):
    """
    Function to convert MWA image into Helioprojective co-ordinate

    Parameters
    ----------
    fits_image : str
        Name of the fits image
    draw_limb : bool, optional
        Draw solar limb or not
    extensions : list, optional
        Output file extensions
    outdirs : list, optional
        Output directories for each extensions
    plot_range : list, optional
        Plot range
    power : float, optional
        Power stretch
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond
    contour_levels : list, optional
        Contour levels in fraction of peak, both positive and negative values allowed
    showgui : bool, optional
        Show GUI

    Returns
    -------
    outfiles
        Saved plot file names
    sunpy.Map
        MWA image in helioprojective co-ordinate
    """
    from matplotlib.patches import Ellipse, Rectangle

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    if showgui:
        matplotlib.use("TkAgg")
    matplotlib.rcParams.update({"font.size": 12})
    fits_image = fits_image.rstrip("/")
    mwa_header = fits.getheader(fits_image)  # Opening MWA fits file
    if mwa_header["CTYPE3"] == "FREQ":
        mwa_header["CRVAL3"] * u.Hz
    elif mwa_header["CTYPE4"] == "FREQ":
        mwa_header["CRVAL4"] * u.Hz
    else:
        pass
    try:
        pixel_unit = mwa_header["BUNIT"]
    except BaseException:
        pass
    pixel_scale = abs(mwa_header["CDELT1"]) * 3600.0  # In arcsec
    obstime = Time(mwa_header["date-obs"])
    mwa_map_rotate = get_mwamap(fits_image, pol="I")
    top_right = SkyCoord(
        xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=mwa_map_rotate.coordinate_frame
    )
    bottom_left = SkyCoord(
        xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=mwa_map_rotate.coordinate_frame
    )
    cropped_map = mwa_map_rotate.submap(bottom_left, top_right=top_right)
    mwa_data = cropped_map.data
    if len(plot_range) < 2:
        norm = ImageNormalize(
            mwa_data,
            vmin=0.03 * np.nanmax(mwa_data),
            vmax=0.99 * np.nanmax(mwa_data),
            stretch=PowerStretch(power),
        )
    else:
        norm = ImageNormalize(
            mwa_data,
            vmin=np.nanmin(plot_range),
            vmax=np.nanmax(plot_range),
            stretch=PowerStretch(power),
        )
    cmap = "inferno"
    pos_color = "white"
    neg_color = "cyan"
    try:
        fig = plt.figure()
        ax = plt.subplot(projection=cropped_map)
        cropped_map.plot(cmap=cmap, axes=ax)
        if len(contour_levels) > 0:
            contour_levels = np.array(contour_levels)
            pos_cont = contour_levels[contour_levels >= 0]
            neg_cont = contour_levels[contour_levels < 0]
            if len(pos_cont) > 0:
                cropped_map.draw_contours(
                    np.sort(pos_cont) * np.nanmax(mwa_data), colors=pos_color
                )
            if len(neg_cont) > 0:
                cropped_map.draw_contours(
                    np.sort(neg_cont) * np.nanmax(mwa_data), colors=neg_color
                )
        ax.coords.grid(False)
        rgba_vmin = plt.get_cmap(cmap)(norm(norm.vmin))
        ax.set_facecolor(rgba_vmin)
        # Read synthesized beam from header
        try:
            bmaj = mwa_header["BMAJ"] * u.deg.to(u.arcsec)  # in arcsec
            bmin = mwa_header["BMIN"] * u.deg.to(u.arcsec)
            bpa = mwa_header["BPA"] - sun.P(obstime).deg  # in degrees
        except KeyError:
            bmaj = bmin = bpa = None
        # Plot PSF ellipse in bottom-left if all values are present
        if bmaj and bmin and bpa is not None:
            # Coordinates where to place the beam (e.g., 5% above bottom-left
            # corner)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            beam_center = SkyCoord(
                x0 + 0.08 * (x1 - x0),
                y0 + 0.08 * (y1 - y0),
                unit=u.arcsec,
                frame=cropped_map.coordinate_frame,
            )

            # Add ellipse patch
            beam_ellipse = Ellipse(
                (beam_center.Tx.value, beam_center.Ty.value),  # center in arcsec
                width=bmin / pixel_scale,
                height=bmaj / pixel_scale,
                angle=bpa,
                edgecolor="white",
                facecolor="white",
                lw=1,
            )
            ax.add_patch(beam_ellipse)
            # Draw square box around the ellipse
            box_size = (
                max(0.2 * (x1 - x0), 1.5 * max(bmin, bmaj)) / pixel_scale
            )  # slightly bigger than beam
            rect = Rectangle(
                (
                    beam_center.Tx.value - box_size / 2,
                    beam_center.Ty.value - box_size / 2,
                ),
                width=box_size,
                height=box_size,
                edgecolor="white",
                facecolor="none",
                lw=1.2,
                linestyle="solid",
            )
            ax.add_patch(rect)
        if draw_limb:
            cropped_map.draw_limb()
        formatter = ticker.FuncFormatter(lambda x, _: f"{int(x):.0e}")
        cbar = plt.colorbar(format=formatter)
        # Optional: set max 5 ticks to prevent clutter
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        if pixel_unit == "K":
            cbar.set_label("Brightness temperature (K)")
        elif pixel_unit == "JY/BEAM":
            cbar.set_label("Flux density (Jy/beam)")
        fig.tight_layout()
        output_image_list = []
        for i in range(len(extensions)):
            ext = extensions[i]
            try:
                outdir = outdirs[i]
            except BaseException:
                outdir = os.path.dirname(os.path.abspath(fits_image))
            if len(contour_levels) > 0:
                output_image = (
                    outdir
                    + "/"
                    + os.path.basename(fits_image).split(".fits")[0].split("_IQUV")[0]
                    + f"_I_contour.{ext}"
                )
            else:
                output_image = (
                    outdir
                    + "/"
                    + os.path.basename(fits_image).split(".fits")[0].split("_IQUV")[0]
                    + f"_I.{ext}"
                )
            output_image_list.append(output_image)
        for output_image in output_image_list:
            fig.savefig(output_image)
        if showgui:
            plt.show()
        plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
    return output_image_list, cropped_map


def plot_hpc_collage(
    fits_images,
    draw_limb=True,
    power=0.5,
    xlim=[-3200, 3200],
    ylim=[-3200, 3200],
    outfile="collage.png",
    showgui=False,
):
    """
    Plot a collage for spectral fits files

    Parameters
    ----------
    fits_images : list
        Fits images list
    draw_limb : bool, optionnal
        Plot solar limb
    power : float, optional
        Power stretch
    xlim : list
        X-axis limit in arcseconds
    ylim : list
        Y-axis limit in arcseconds
    outfile : str, optional
        Output file name
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Output file name
    """
    from matplotlib.patches import Ellipse

    if showgui:
        matplotlib.use("TkAgg")
    maps, datas = [], []
    # ---- Load & crop ----
    fits_images = sorted(fits_images)
    ncols = int(np.sqrt(len(fits_images)))
    for fits_image in fits_images:
        hdr = fits.getheader(fits_image)
        obstime = Time(hdr["date-obs"])
        mwa_map = get_mwamap(fits_image)
        tr = SkyCoord(
            xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=mwa_map.coordinate_frame
        )
        bl = SkyCoord(
            xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=mwa_map.coordinate_frame
        )
        cropped = mwa_map.submap(bl, top_right=tr)
        maps.append((cropped, hdr, obstime))
        datas.append(cropped.data)
    # ---- Global normalization ----
    all_data = np.concatenate([d.flatten() for d in datas])
    vmin = 0.03 * np.nanmax(all_data)
    vmax = 0.99 * np.nanmax(all_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=PowerStretch(power))
    try:
        maps = sorted(maps, key=lambda x: x[1].get("CRVAL3", 0))
    except Exception:
        try:
            maps = sorted(maps, key=lambda x: x[1].get("CRVAL3", 0))
        except Exception:
            pass
    # ---- Layout ----
    nrows = 4
    ncols = 6
    total = nrows * ncols
    fig = plt.figure(figsize=(24, 16))
    if len(maps) == 0:
        return
    else:
        ref_map = maps[0][0]
    for i in range(total):
        ax = plt.subplot(nrows, ncols, i + 1)
        if i < len(maps):
            m, hdr, obstime = maps[i]
            ax.remove()
            ax = plt.subplot(nrows, ncols, i + 1, projection=m)
            m.plot(axes=ax, cmap="inferno", norm=norm)
            if draw_limb:
                m.draw_limb(axes=ax)
            ax.coords.grid(False)
            ax.coords[0].set_ticks_visible(False)
            ax.coords[1].set_ticks_visible(False)
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[1].set_ticklabel_visible(False)
            try:
                bmaj = hdr["BMAJ"] * u.deg.to(u.arcsec)
                bmin = hdr["BMIN"] * u.deg.to(u.arcsec)
                bpa = hdr["BPA"] - sun.P(obstime).deg
                pixel_scale = abs(hdr["CDELT1"]) * 3600.0
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                x = x0 + 0.1 * (x1 - x0)
                y = y0 + 0.1 * (y1 - y0)
                beam = Ellipse(
                    (x, y),
                    width=bmin / pixel_scale,
                    height=bmaj / pixel_scale,
                    angle=bpa,
                    edgecolor="white",
                    facecolor="white",
                    lw=0.7,
                )
                ax.add_patch(beam)
            except Exception:
                pass
            try:
                if hdr.get("CTYPE3") == "FREQ":
                    freq = hdr["CRVAL3"] / 1e6
                elif hdr.get("CTYPE4") == "FREQ":
                    freq = hdr["CRVAL4"] / 1e6
                else:
                    freq = None
                if freq is not None:
                    ax.set_title(f"{freq:.0f} MHz", fontsize=20)
                else:
                    ax.set_title("", fontsize=20)
            except Exception:
                pass
        else:
            zero_data = np.zeros_like(ref_map.data)
            efmap = Map(zero_data, ref_map.meta)
            ax.remove()
            ax = plt.subplot(nrows, ncols, i + 1, projection=efmap)
            efmap.plot(axes=ax, cmap="inferno", norm=norm)
            try:
                ax.coords.grid(False)
                ax.coords[0].set_ticks_visible(False)
                ax.coords[1].set_ticks_visible(False)
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[1].set_ticklabel_visible(False)
            except Exception:
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_facecolor("black")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("white")
                spine.set_linewidth(0.6)
            ax.text(
                0.5,
                0.5,
                "No Data",
                color="gray",
                fontsize=20,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("", fontsize=20)
    # ---- Layout (NO gaps, space for labels) ----
    plt.subplots_adjust(
        left=0.04, right=0.87, bottom=0.03, top=0.95, wspace=0.1, hspace=0.0
    )
    # ---- Global labels ----
    fig.text(0.5, 0.02, "Solar-X", ha="center", fontsize=24)
    fig.text(0.02, 0.5, "Solar-Y", va="center", rotation="vertical", fontsize=24)
    # ---- Save ----
    fig.savefig(outfile, dpi=120, bbox_inches="tight")
    if showgui:
        plt.show()
    plt.close(fig)
    return outfile


def get_aia_map(
    obs_date,
    obs_time,
    workdir,
    obs_end_date="",
    obs_end_time="",
    aia_wavelength=193,
    ncpu=1,
):
    """
    Get SDO AIA map

    Parameters
    ----------
    obs_date : str
        Observation date in yyyy-mm-dd format
    obs_time : str
        Observation time in hh:mm format
    workdir : str
        Work directory
    obs_end_date : str, optional
        Observation end date in yyyy-mm-dd format
    obs_end_time : str, optional
        Observation end time in hh:mm format
    aia_wavelength : float, optional
        Wavelength, options: 94, 131, 171, 193, 211, 304, 335 Å
    ncpu : int, optional
        Number of CPU to use for parallel download

    Returns
    -------
    list
        AIA fits files
    """
    logging.getLogger("sunpy").setLevel(logging.ERROR)
    logging.getLogger("drms").setLevel(logging.ERROR)
    logging.getLogger("drms.client").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="This download has been started in a thread which is not the main thread",
    )
    aia_wavelengths = np.array([94, 131, 171, 193, 211, 304, 335])
    if aia_wavelength not in aia_wavelengths:
        pos = np.argmin(np.abs(aia_wavelength - aia_wavelengths))
        aia_wavelength = aia_wavelengths[pos]
    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    aiadir = f"{workdir}/aiamaps"
    os.makedirs(aiadir, exist_ok=True)
    try:
        final_time_range = []
        if obs_end_date == "" or obs_end_time == "":
            start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
            t_start = start_time.strftime("%Y-%m-%dT%H:%M")
            time = a.Time(t_start, t_start)
            final_time_range.append(t_start)
        else:
            start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
            t_start = start_time.strftime("%Y-%m-%dT%H:%M")
            end_time = dt.fromisoformat(f"{obs_end_date}T{obs_end_time}")
            t_end = end_time.strftime("%Y-%m-%dT%H:%M")
            start_mjdsec = timestamp_to_mjdsec(f"{start_time}", date_format=2)
            end_mjdsec = timestamp_to_mjdsec(f"{end_time}", date_format=2)
            if end_mjdsec > start_mjdsec:
                time = a.Time(t_start, t_end)
                final_time_range.append(t_start)
                final_time_range.append(t_end)
            else:
                time = a.Time(t_start, t_start)
                final_time_range.append(t_start)

        print("Downloading AIA images....")
        a.Instrument("aia")
        jsoc_wavelength = a.Wavelength(aia_wavelength * u.angstrom)
        results = Fido.search(
            time,
            a.jsoc.Series("aia.lev1_euv_12s"),
            a.jsoc.Notify("paircarsnotification@gmail.com"),
            jsoc_wavelength,
        )
        num_files = results.file_num
        if num_files == 0:
            return []
        else:
            downloaded_files = Fido.fetch(
                results,
                path=workdir,
                progress=True,
                overwrite=False,
                max_conn=ncpu,
            )
            final_maps = []
            if len(downloaded_files) > 0:
                if len(final_time_range) == 1:
                    downloaded_files = downloaded_files[:1]
                for final_image in downloaded_files:
                    aia_map = Map(final_image)
                    # Step 1: Pointing correction
                    try:
                        pointing_corrected_map = update_pointing(aia_map)
                    except Exception:
                        pointing_corrected_map = aia_map
                    # Step 2: register (we are skipping PSF deconvolution)
                    try:
                        registered_map = register(pointing_corrected_map)
                    except Exception:
                        registered_map = pointing_corrected_map
                    # Step 3: instrument degradation correction
                    try:
                        corrected_map = correct_degradation(registered_map)
                    except Exception:
                        corrected_map = registered_map
                    # Step 4: Normalize by exposure time
                    try:
                        normalized_data = (
                            corrected_map.data
                            / corrected_map.exposure_time.to(u.s).value
                        )
                        normalized_map = Map(normalized_data, corrected_map.meta)
                    except Exception:
                        normalized_map = corrected_map
                    for image in downloaded_files:
                        basename = image.split(".image")[0]
                        os.system(f"rm -rf {basename}*")
                    final_fits = f"{aiadir}/{os.path.basename(basename)}.fits"
                    normalized_map.save(final_fits, overwrite=True)
                    if os.path.exists(final_fits):
                        final_maps.append(final_fits)
                return final_maps
            else:
                return []
    except Exception:
        os.system(f"rm -rf {workdir}/*aia*.fits")
        traceback.print_exc()
        return []
    finally:
        os.chdir(cwd)


def get_suvi_map(
    obs_date,
    obs_time,
    workdir,
    obs_end_date="",
    obs_end_time="",
    suvi_wavelength=195,
    ncpu=1,
    keep_suvi_fits=False,
):
    """
    Get GOES SUVI map

    Parameters
    ----------
    obs_date : str
        Observation date in yyyy-mm-dd format
    obs_time : str
        Observation time in hh:mm format
    workdir : str
        Work directory
    obs_end_date : str
        Observation end date in yyyy-mm-dd format
    obs_end_time : str
        Observation end time in hh:mm format
    suvi_wavelength : float, optional
        Wavelength, options: 94, 131, 171, 195, 284, 304 Å
    ncpu : int, optional
        Number of CPU threads to use for parallel download
    keep_suvi_fits : bool, optional
        Keep SUVI fits file or not

    Returns
    -------
    list
        List of Sunpy SUVIMap
    """
    from parfive import Downloader
    from bs4 import BeautifulSoup

    def list_url_directory(url, ext=""):
        page = requests.get(url).text
        soup = BeautifulSoup(page, "html.parser")
        return [
            url + node.get("href")
            for node in soup.find_all("a")
            if node.get("href").endswith(ext)
        ]

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="This download has been started in a thread which is not the main thread",
    )
    suvi_wavelengths = np.array([94, 131, 171, 195, 284, 304])
    if suvi_wavelength not in suvi_wavelengths:
        pos = np.argmin(np.abs(suvi_wavelength - suvi_wavelengths))
        suvi_wavelength = suvi_wavelengths[pos]
    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        print("Downloading SUVI images...")
        baseurl1 = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes"
        baseurl2 = "l2/data"
        ext = ".fits"

        spacecraft_numbers = [16, 18]
        wvln_path = dict(
            {
                94: "suvi-l2-ci094",
                131: "suvi-l2-ci131",
                171: "suvi-l2-ci171",
                195: "suvi-l2-ci195",
                284: "suvi-l2-ci284",
                304: "suvi-l2-ci304",
            }
        )
        date_str = "/".join(obs_date.split("-"))
        all_files = []
        start_times = []
        out_files = []
        final_maps = []
        for spacecraft in spacecraft_numbers:
            url = f"{baseurl1}{spacecraft}/{baseurl2}/{wvln_path[suvi_wavelength]}/{date_str}/"
            request = requests.get(url)
            if not request.status_code == 200:
                pass
            else:
                for file_name in list_url_directory(url, ext):
                    all_files.append(file_name)
                    file_base = os.path.basename(file_name)
                    start_times.append(file_base.split("_")[-3])
                    out_files.append(f"{workdir}/{file_base}")
                times_dt = [dt.strptime(t, "s%Y%m%dT%H%M%Sz") for t in start_times]
                if obs_end_date == "" or obs_end_time == "":
                    start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
                    closest_time = min(times_dt, key=lambda t: abs(t - start_time))
                    pos = [times_dt.index(closest_time)]
                else:
                    start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
                    end_time = dt.fromisoformat(f"{obs_end_date}T{obs_end_time}")
                    start_mjdsec = timestamp_to_mjdsec(
                        f"{obs_date}T{obs_time}:00.0", date_format=1
                    )
                    end_mjdsec = timestamp_to_mjdsec(
                        f"{obs_end_date}T{obs_end_time}:00.0", date_format=1
                    )
                    if end_mjdsec > start_mjdsec:
                        pos = [
                            i
                            for i, t in enumerate(times_dt)
                            if start_time <= t <= end_time
                        ]
                    else:
                        closest_time = min(times_dt, key=lambda t: abs(t - start_time))
                        pos = [times_dt.index(closest_time)]
                if len(pos) > 0:
                    download_urls = [all_files[p] for p in pos]
                    out_files = [out_files[p] for p in pos]
                    dl = Downloader(max_conn=ncpu, progress=True, overwrite=False)
                    for i in range(len(out_files)):
                        out_file = out_files[i]
                        download_url = download_urls[i]
                        if os.path.exists(out_file) is False:
                            dl.enqueue_file(download_url, path=workdir)
                    dl.download()
                    filtered_outfiles = []
                    for outfile in out_files:
                        if os.path.exists(outfile):
                            filtered_outfiles.append(outfile)
                    if len(filtered_outfiles) > 0:
                        for image in filtered_outfiles:
                            suvi_map = Map(image)
                            final_maps.append(suvi_map)
                            if keep_suvi_fits is False:
                                os.system(f"rm -rf {image}")
                        break
        return final_maps
    except Exception:
        os.system("rm -rf *suvi*fits")
        traceback.print_exc()
        return []
    finally:
        os.chdir(cwd)


def enhance_offlimb(sunpy_map, do_sharpen=True):
    """
    Enhance off-disk emission

    Parameters
    ----------
    sunpy_map : sunpy.map
        Sunpy map
    do_sharpen : bool, optional
        Sharpen images

    Returns
    -------
    sunpy.map
        Off-disk enhanced emission
    """
    from scipy.ndimage import gaussian_filter
    from sunpy.map.maputils import all_coordinates_from_map

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    hpc_coords = all_coordinates_from_map(sunpy_map)
    r = np.sqrt(hpc_coords.Tx**2 + hpc_coords.Ty**2) / sunpy_map.rsun_obs
    rsun_step_size = 0.01
    rsun_array = np.arange(1, r.max(), rsun_step_size)
    y = np.array(
        [
            sunpy_map.data[(r > this_r) * (r < this_r + rsun_step_size)].mean()
            for this_r in rsun_array
        ]
    )
    pos = np.where(y < 10e-3)[0][0]
    r_lim = round(rsun_array[pos], 2)
    params = np.polyfit(
        rsun_array[rsun_array < r_lim], np.log(y[rsun_array < r_lim]), 1
    )
    scale_factor = np.exp((r - 1) * -params[0])
    scale_factor[r < 1] = 1
    if do_sharpen:
        blurred = gaussian_filter(sunpy_map.data, sigma=3)
        data = sunpy_map.data + (sunpy_map.data - blurred)
    else:
        data = sunpy_map.data
    scaled_map = Map(data * scale_factor, sunpy_map.meta)
    scaled_map.plot_settings["norm"] = ImageNormalize(stretch=LogStretch(10))
    return scaled_map


_map_cache = {}


def get_map_cached(mapfile):
    from sunpy.map import Map

    if mapfile not in _map_cache:
        _map_cache[mapfile] = Map(mapfile)
    return _map_cache[mapfile]


def get_all_euv_maps(mwa_fits_images, workdir, wavelength=195, ncpu=1):
    """
    Get all EUV maps for all MWA fits images

    Parameters
    ----------
    mwa_fits_images : list, str
        MWA FITS images list
    workdir : str
        Work directory
    wavelength : float, optional
        GOES SUVI/ SDO AIA wavelength, options: 94, 131, 171, 195(193), 284, 304 Å
    ncpu : int, optional
        Number of CPU threads to use

    Returns
    -------
    list
        List of sunpy fits image names in same order of input images
    """
    cwd = os.getcwd()
    os.chdir(workdir)
    if isinstance(mwa_fits_images, str):
        mwa_fits_images = [mwa_fits_images]
    try:
        obstimes = []
        all_obstimes_mjdsecs = []
        for mwa_image in mwa_fits_images:
            obs_datetime = fits.getheader(mwa_image)["DATE-OBS"]
            if obs_datetime not in obstimes:
                obstimes.append(obs_datetime)
            all_obstimes_mjdsecs.append(
                timestamp_to_mjdsec(obs_datetime, date_format=1)
            )
        mjdsecs = [timestamp_to_mjdsec(t, date_format=1) for t in obstimes]
        mid_mjdsec = mjdsecs[int(len(mjdsecs) / 2)]
        start_time = mjdsec_to_timestamp(mid_mjdsec, str_format=0)[:-5]
        start_obs_date = start_time.split("T")[0]
        start_obs_time = ":".join(start_time.split("T")[-1].split(":")[:2])
        start_year = int(start_obs_date.split("-")[0])
        if start_year >= 2019:
            euv_images = get_suvi_map(
                start_obs_date,
                start_obs_time,
                workdir,
                suvi_wavelength=wavelength,
                ncpu=ncpu,
            )
            if len(euv_images) == 0:
                euv_images = get_aia_map(
                    start_obs_date,
                    start_obs_time,
                    workdir,
                    aia_wavelength=wavelength,
                    ncpu=ncpu,
                )
        else:
            euv_images = get_aia_map(
                start_obs_date,
                start_obs_time,
                workdir,
                aia_wavelength=wavelength,
                ncpu=ncpu,
            )
        map_obstimes = []
        if len(euv_images) == 0:
            print("No EUV images are found.")
            return []
        else:
            for euv_fits in euv_images:
                m = Map(euv_fits)
                map_obstimes.append(m.date.value.split(".")[0])
            map_mjdsecs = [timestamp_to_mjdsec(t, date_format=1) for t in map_obstimes]
            final_maps = []
            map_mjdsecs = np.array(map_mjdsecs)
            all_obstimes_mjdsecs = np.array(all_obstimes_mjdsecs)
            for fits_time in all_obstimes_mjdsecs:
                pos = np.argmin(np.abs(map_mjdsecs - fits_time))
                final_maps.append(euv_images[pos])
            return final_maps
    except Exception:
        traceback.print_exc()
        return []
    finally:
        os.chdir(cwd)


def make_mwa_overlay(
    mwa_image,
    euv_fits,
    workdir,
    plot_file_prefix,
    plot_mwa_colormap=True,
    enhance_offdisk=False,
    pol="I",
    contour_levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    euv_image_scaling=0.5,
    do_sharpen_euv=True,
    xlim=[-2500, 2500],
    ylim=[-2500, 2500],
    extensions=["png"],
    outdirs=[],
    showgui=False,
    verbose=False,
):
    """
    Make overlay of MWA image on GOES SUVI/ SDO AIA image

    Parameters
    ----------
    mwa_image : str
        MWA image
    euv_fits : EUV image fits
        GOES SUVI/ SDO AIA EUV image fits
    workdir : str
        Work directory
    plot_file_prefix : str
        Plot file prefix name
    plot_mwa_colormap : bool, optional
        Plot MWA map colormap
    enhance_offdisk : bool, optional
        Enhance off-disk emission
    pol : str, optional
        Stokes plane of MWA image
    contour_levels : list, optional
        Contour levels in fraction of peak
    euv_image_scaling : float, optional
       EUV image pixel scaling (should be smaller than 1.0)
    do_sharpen_euv : bool, optional
        Do sharpen EUV images
    xlim : list, optional
        X-axis limit in arcsec
    tlim : list, optional
        Y-axis limit in arcsec
    extensions : list, optional
        Image file extensions
    outdirs : list, optional
        Output directories for each extensions
    verbose: bool, optinal
        Verbose output

    Returns
    -------
    list
        Plot file names
    """
    matplotlib.use("Agg")
    mwa_image = mwa_image.rstrip("/")
    if verbose:
        print(f"Making overlay for image: {os.path.basename(mwa_image)}")
    euv_map = get_map_cached(euv_fits)

    mwamap = get_mwamap(mwa_image, pol=pol)
    if enhance_offdisk:
        euv_map = enhance_offlimb(euv_map, do_sharpen=do_sharpen_euv)

    projected_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        obstime=mwamap.observer_coordinate.obstime,
        frame="helioprojective",
        observer=mwamap.observer_coordinate,
        rsun=mwamap.coordinate_frame.rsun,
    )
    mwa_header = mwamap.meta
    euv_header = euv_map.meta

    euv_pix = max(1024, int(euv_header["naxis1"] * euv_image_scaling))
    euv_header["naxis1"] * euv_header["cdelt1"]
    mwa_image_fov = mwa_header["naxis1"] * mwa_header["cdelt1"]

    new_scale = float(mwa_image_fov / euv_pix) * u.arcsec / u.pix
    SpatialPair = namedtuple("SpatialPair", "axis1 axis2")
    new_scale = SpatialPair(axis1=new_scale, axis2=new_scale)
    new_shape = (euv_pix, euv_pix)

    projected_header = make_fitswcs_header(
        new_shape,
        projected_coord,
        scale=u.Quantity(new_scale),
        instrument=euv_map.instrument,
        wavelength=euv_map.wavelength,
    )

    with SphericalScreen(mwamap.observer_coordinate):
        mwa_tmp = mwamap.reproject_to(projected_header)
    mwa_reprojected = Map(mwa_tmp.data.astype(np.float32), mwa_tmp.meta)

    with SphericalScreen(euv_map.observer_coordinate):
        euv_reprojected = euv_map.reproject_to(projected_header)

    mwatime = mwamap.meta["date-obs"].split(".")[0]
    euvtime = euv_map.meta["date-obs"].split(".")[0]
    try:
        if plot_mwa_colormap and len(contour_levels) > 0:
            matplotlib.rcParams.update({"font.size": 18})
            fig = plt.figure(figsize=(16, 8))
            ax_colormap = fig.add_subplot(1, 2, 1, projection=euv_reprojected)
            ax_contour = fig.add_subplot(1, 2, 2, projection=euv_reprojected)
        elif plot_mwa_colormap:
            matplotlib.rcParams.update({"font.size": 14})
            fig = plt.figure(figsize=(10, 8))
            ax_colormap = fig.add_subplot(projection=euv_reprojected)
        elif len(contour_levels) > 0:
            matplotlib.rcParams.update({"font.size": 14})
            fig = plt.figure(figsize=(10, 8))
            ax_contour = fig.add_subplot(projection=euv_reprojected)
        else:
            print("No overlay is plotting.")
            return

        title = f"EUV time: {euvtime}\n MWA time: {mwatime}\n Stokes {pol}"
        if "transparent_inferno" not in plt.colormaps():
            cmap = cm.get_cmap("inferno", 256)
            colors = cmap(np.linspace(0, 1, 256))
            x = np.linspace(0, 1, 256)
            alpha = 0.8 * (1 - np.exp(-3 * x))
            colors[:, -1] = alpha  # Update the alpha channel
            transparent_inferno = ListedColormap(colors)
            plt.colormaps.register(name="transparent_inferno", cmap=transparent_inferno)
        if plot_mwa_colormap and len(contour_levels) > 0:
            suptitle = title.replace("\n", ",")
            title = ""
            fig.suptitle(suptitle)
        if plot_mwa_colormap:
            z = 0
            euv_reprojected.plot(
                axes=ax_colormap,
                title=title,
                autoalign=True,
                clip_interval=(3, 99.9) * u.percent,
                zorder=z,
            )
            z += 1
            mwa_reprojected.plot(
                axes=ax_colormap,
                title=title,
                clip_interval=(3, 99.9) * u.percent,
                cmap="transparent_inferno",
                zorder=z,
            )
            ax_colormap.set_facecolor("black")

        if len(contour_levels) > 0:
            z = 0
            euv_reprojected.plot(
                axes=ax_contour,
                title=title,
                autoalign=True,
                clip_interval=(3, 99.9) * u.percent,
                zorder=z,
            )
            z += 1
            contour_levels = np.array(contour_levels) * np.nanmax(mwa_reprojected.data)
            mwa_reprojected.draw_contours(
                contour_levels, axes=ax_contour, cmap="YlGnBu", zorder=z
            )
            ax_contour.set_facecolor("black")

        if len(xlim) > 0:
            x_pix_limits = []
            for x in xlim:
                sky = SkyCoord(
                    x * u.arcsec, 0 * u.arcsec, frame=euv_reprojected.coordinate_frame
                )
                x_pix = euv_reprojected.world_to_pixel(sky)[0].value
                x_pix_limits.append(x_pix)
            if plot_mwa_colormap and len(contour_levels) > 0:
                ax_colormap.set_xlim(x_pix_limits)
                ax_contour.set_xlim(x_pix_limits)
            elif plot_mwa_colormap:
                ax_colormap.set_xlim(x_pix_limits)
            elif len(contour_levels) > 0:
                ax_contour.set_xlim(x_pix_limits)
        if len(ylim) > 0:
            y_pix_limits = []
            for y in ylim:
                sky = SkyCoord(
                    0 * u.arcsec, y * u.arcsec, frame=euv_reprojected.coordinate_frame
                )
                y_pix = euv_reprojected.world_to_pixel(sky)[1].value
                y_pix_limits.append(y_pix)
            if plot_mwa_colormap and len(contour_levels) > 0:
                ax_colormap.set_ylim(y_pix_limits)
                ax_contour.set_ylim(y_pix_limits)
            elif plot_mwa_colormap:
                ax_colormap.set_ylim(y_pix_limits)
            elif len(contour_levels) > 0:
                ax_contour.set_ylim(y_pix_limits)
        if plot_mwa_colormap and len(contour_levels) > 0:
            ax_colormap.coords.grid(False)
            ax_contour.coords.grid(False)
        elif plot_mwa_colormap:
            ax_colormap.coords.grid(False)
        elif len(contour_levels) > 0:
            ax_contour.coords.grid(False)
        fig.subplots_adjust(
            left=0.1,  # space from left edge
            right=0.98,  # space from right edge
            bottom=0.08,  # space from bottom
            top=0.9,  # space from top
            wspace=0.27,  # horizontal space between panels
            hspace=0.05,  # vertical space between panels
        )
        plot_file_list = []
        for i in range(len(extensions)):
            ext = extensions[i]
            try:
                savedir = outdirs[i]
            except BaseException:
                savedir = workdir
            plot_file = f"{savedir}/{plot_file_prefix}.{ext}"
            plt.savefig(plot_file, bbox_inches="tight")
            if verbose:
                print(f"Plot saved: {plot_file}")
            plot_file_list.append(plot_file)
    except Exception:
        traceback.print_exc()
    finally:
        del (
            mwamap,
            euv_map,
            euv_reprojected,
            mwa_reprojected,
            projected_header,
            projected_coord,
        )
        plt.close("all")
    return plot_file_list


def plot_goes_full_timeseries(
    msname, workdir, plot_file_prefix=None, extension="png", showgui=False
):
    """
    Plot GOES full time series on the day of observation

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    plot_file_prefix : str, optional
        Plot file name prefix
    extension : str, optional
        Save file extension
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Plot file name
    """
    os.makedirs(workdir, exist_ok=True)
    if showgui:
        matplotlib.use("TkAgg")
    matplotlib.rcParams.update({"font.size": 14})
    scans = get_ms_scans(msname)
    msmd = msmetadata()
    msmd.open(msname)
    tstart_mjd = min(msmd.timesforscan(int(min(scans))))
    tend_mjd = max(msmd.timesforscan(int(max(scans))))
    msmd.close()
    tstart = mjdsec_to_timestamp(tstart_mjd, str_format=2)
    tend = mjdsec_to_timestamp(tend_mjd, str_format=2)
    print(f"Time range: {tstart}~{tend}")
    results = Fido.search(
        a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
    )
    files = Fido.fetch(results, path=workdir, overwrite=False)
    goes_tseries = TimeSeries(files, concatenate=True)
    for f in files:
        os.system(f"rm -rf {f}")
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    goes_tseries.plot(axes=ax)
    times = goes_tseries.time
    times_dt = times.to_datetime()
    ax.axvspan(tstart, tend, alpha=0.2)
    ax.set_xlim(times_dt[0], times_dt[-1])
    # Save or show
    if plot_file_prefix:
        plot_file = f"{workdir}/{plot_file_prefix}.{extension}"
        plt.savefig(plot_file, bbox_inches="tight")
        print(f"Plot saved: {plot_file}")
    else:
        plot_file = None
    if showgui:
        plt.show()
        plt.close(fig)
        plt.close("all")
    else:
        plt.close(fig)
    return plot_file


def rename_mwasolar_image(
    imagename,
    imagetype="image",
    imagedir="",
    pol="",
    cutout_rsun=10.0,
    make_plots=True,
    pol_selfcal=True,
    cal_sol=True,
    paircars_input="",
):
    """
    Rename and move image to image directory

    Parameters
    ----------
    imagename : str
        Image name
    imagetype : str, optional
        Image type (image, model, residual)
    imagedir : str, optional
        Image directory (default given image directory)
    pol : str, optional
        Stokes parameters
    cutout_rsun : float, optional
        Cutout in solar radii from center (default: 10.0 solar radii)
    make_plots : bool, optional
        Make radio map plot in helioprojective coordinates
    pol_selfcal : bool, optional
        Whether polarisation self-calibration solutions are applied
    cal_sol : bool, optional
        Whether calibration solutions are applied or not
    paircars_input : str, optional
        P-AIRCARS input command line 
        Note: If provided, it will be written in header

    Returns
    -------
    str
        New imagename with full path
    """
    imagename = imagename.rstrip("/")
    if imagetype == "image":
        maxval, minval, rms, total_val, mean_val, median_val, rms_dyn, minmax_dyn = (
            calc_solar_image_stat(imagename, disc_size=50)
        )
    imagename = cutout_image(
        imagename, imagename, x_deg=(cutout_rsun * 2 * 16.0) / 60.0
    )
    if imagetype == "image" and (rms == 0 or np.isnan(rms_dyn)):
        os.system(f"rm -rf {imagename}")
        return

    header = fits.getheader(imagename)
    time = header["DATE-OBS"]
    with fits.open(imagename, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["AUTHOR"] = "DevojyotiKansabanik"
        hdr["PIPELINE"] = "P-AIRCARS"
        if imagetype == "image":
            hdr["MAX"] = maxval
            hdr["MIN"] = minval
            hdr["RMS"] = rms
            hdr["SUM"] = total_val
            hdr["MEAN"] = mean_val
            hdr["MEDIAN"] = median_val
            hdr["RMSDYN"] = rms_dyn
            hdr["MIMADYN"] = minmax_dyn
        if cal_sol:
            hdr["CALAPP"] = "TRUE"
        else:
            hdr["CALAPP"] = "FALSE"
        if pol_selfcal:
            hdr["POLSELF"] = "TRUE"
        else:
            hdr["POLSELF"] = "FALSE"
        if paircars_input!="":
            hdr["RUNCMD"]=paircars_input
    freq = round(header["CRVAL3"] / 10**6, 2)
    t_str = "".join(time.split("T")[0].split("-")) + (
        "".join(time.split("T")[-1].split(":"))
    )
    new_name = "time_" + t_str + "_freq_" + str(freq)
    if pol != "":
        new_name += "_pol_" + str(pol)
    if "MFS" in imagename:
        new_name += "_MFS"
    new_name = new_name + ".fits"
    if imagedir == "":
        imagedir = os.path.dirname(os.path.abspath(imagename))
    new_name = imagedir + "/" + new_name
    os.system("mv " + imagename + " " + new_name)
    if imagetype == "image":
        hpcdir = f"{os.path.dirname(imagedir)}/images/hpcs"
        os.makedirs(hpcdir, exist_ok=True)
        save_in_hpc(new_name, outdir=hpcdir)
        if make_plots:
            try:
                pngdir = f"{os.path.dirname(imagedir)}/images/pngs"
                os.makedirs(pngdir, exist_ok=True)
                outimages, cropped_map = plot_in_hpc(
                    new_name,
                    draw_limb=True,
                    extensions=["png"],
                    outdirs=[pngdir],
                )
            except Exception:
                pass
    return new_name


def make_ds_plot(dsfiles, plot_file=None, plot_quantity="TB", showgui=False):
    """
    Make dynamic spectrum plot

    Parameters
    ----------
    dsfile : list
        DS files list
    plot_file : str, optional
        Plot file name to save the plot
    plot_quantity : str, optional
        Plot quantity (TB or flux)
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Plot name
    """
    if showgui:
        matplotlib.use("TkAgg")
    matplotlib.rcParams.update({"font.size": 18})
    if isinstance(dsfiles, str):
        dsfiles = [dsfiles]
    dsfiles = np.array(dsfiles)
    start_freqs = []
    for dsfile in dsfiles:
        freqs, _, _, _, _, _ = np.load(dsfile, allow_pickle=True)
        start_freqs.append(freqs[0])
    pos = np.argsort(start_freqs)
    dsfiles = dsfiles[pos].tolist()
    for i, dsfile in enumerate(dsfiles):
        freqs_i, times_i, timestamps_i, T_data_i, S_data_i, flags = np.load(
            dsfile, allow_pickle=True
        )
        data_i = T_data_i / 1e6 if plot_quantity == "TB" else S_data_i
        data_i[flags] = np.nan
        # interpolate along freq axis
        for t in range(data_i.shape[1]):
            t_data = data_i[:, t]
            t_data_interp = interpolate_nans(t_data)
            t_data_interp[t_data_interp == 0] = np.nan
            data_i[:, t] = t_data_interp
        if i == 0:
            freqs = freqs_i
            timestamps = timestamps_i
            data = data_i
        else:
            df = np.nanmedian(np.diff(freqs))
            gapsize = int(np.round((np.nanmin(freqs_i) - np.nanmax(freqs)) / df))
            gapsize = 1

            if 0 < gapsize < 5:
                last_med = np.nanmedian(data[-1, :])
                new_med = np.nanmedian(data_i[0, :])
                if np.isfinite(new_med) and new_med != 0:
                    data_i = (data_i / new_med) * last_med
            if gapsize > 0:
                gap = np.full((gapsize, data.shape[1]), np.nan)
                data = np.concatenate([data, gap, data_i], axis=0)
                freqs = np.append(freqs, np.full(gapsize, np.nan))
            else:
                data = np.concatenate([data, data_i], axis=0)
            freqs = np.append(freqs, freqs_i)
    # --------------------------------------------------
    # Trim invalid freq rows
    # --------------------------------------------------
    median_bandshape = np.nanmedian(data, axis=-1)
    pos = np.where(~np.isnan(median_bandshape))[0]
    if len(pos) > 0:
        data = data[min(pos) : max(pos), :]
        freqs = freqs[min(pos) : max(pos)]
    # --------------------------------------------------
    # Convert timestamps → datetime (MASTER AXIS)
    # --------------------------------------------------
    times_dt = np.array(
        [
            (
                datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
                if "." in t
                else datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
            )
            for t in timestamps
        ]
    )
    # --------------------------------------------------
    # Fetch GOES (for overlay only)
    # --------------------------------------------------
    tstart = times_dt[0].strftime("%Y-%m-%d %H:%M")
    tend = times_dt[-1].strftime("%Y-%m-%d %H:%M")
    try:
        results = Fido.search(
            a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
        )
        files = Fido.fetch(results, path=os.path.dirname(dsfiles[0]), overwrite=False)
        goes_tseries = TimeSeries(files, concatenate=True).truncate(tstart, tend)
    except Exception:
        goes_tseries = None
    # --------------------------------------------------
    # MWA time series
    # --------------------------------------------------
    timeseries = np.nanmean(data, axis=0)
    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------
    norm = ImageNormalize(
        data,
        stretch=LogStretch(1),
        vmin=0.99 * np.nanmin(data),
        vmax=0.99 * np.nanmax(data),
    )
    try:
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 2, width_ratios=[1, 0.03], height_ratios=[4, 1.5, 2])
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_ts = fig.add_subplot(gs[1, 0])
        ax_goes = fig.add_subplot(gs[2, 0])
        cax = fig.add_subplot(gs[:, 1])
        # --------------------------------------------------
        # Dynamic spectrum (FIXED AXIS)
        # --------------------------------------------------
        im = ax_spec.imshow(
            data,
            aspect="auto",
            origin="lower",
            norm=norm,
            cmap="magma",
            extent=[
                mdates.date2num(times_dt[0]),
                mdates.date2num(times_dt[-1]),
                0,
                data.shape[0],
            ],
        )
        ax_spec.set_ylabel("Frequency (MHz)")
        ax_spec.set_xticklabels([])
        # Frequency ticks
        freqs_arr = np.array(freqs)
        valid = ~np.isnan(freqs_arr)
        idx = np.where(valid)[0]
        idx = idx[:: max(1, len(idx) // 12)]
        ax_spec.set_yticks(idx)
        ax_spec.set_yticklabels([f"{freqs_arr[i]:.1f}" for i in idx])
        # --------------------------------------------------
        # MWA time series
        # --------------------------------------------------
        ax_ts.plot(times_dt, timeseries)
        ax_ts.set_xlim(times_dt[0], times_dt[-1])
        ax_ts.set_ylabel("TB (MK)" if plot_quantity == "TB" else "S (SFU)")
        ax_ts.set_xticklabels([])
        # --------------------------------------------------
        # GOES plot (aligned, not controlling axis)
        # --------------------------------------------------
        if goes_tseries is not None:
            goes_tseries.plot(axes=ax_goes)
        ax_goes.set_xlim(times_dt[0], times_dt[-1])
        ax_goes.set_ylabel(r"Flux ($W/m^2$)")
        ax_goes.set_title("GOES light curve", fontsize=14)
        # Titles
        ax_spec.set_title("MWA dynamic spectrum", fontsize=14)
        ax_ts.set_title("MWA light curve", fontsize=14)
        ax_goes.set_xlabel(f"Time (UTC), {times_dt[0].strftime('%Y-%m-%d')}")
        # --------------------------------------------------
        # Shared time formatting
        # --------------------------------------------------
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter("%H:%M:%S")
        for ax in [ax_spec, ax_ts, ax_goes]:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        # --------------------------------------------------
        # Colorbar
        # --------------------------------------------------
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(
            "Brightness temperature (MK)"
            if plot_quantity == "TB"
            else "Flux density (SFU)"
        )
        plt.tight_layout()
        if plot_file:
            plt.savefig(plot_file, bbox_inches="tight")
            print(f"Plot saved: {plot_file}")
        if showgui:
            plt.show()
        plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
    return plot_file
