import types
import astropy.units as u
import logging
import psutil
import numpy as np
import warnings
import glob
import dask
import requests
import os
import traceback
import matplotlib
import matplotlib.pyplot as plt
from parfive import Downloader
from bs4 import BeautifulSoup
from dask import delayed, compute
from multiprocessing.pool import ThreadPool
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from sunpy.timeseries import TimeSeries
from aiapy.calibrate import *
from astropy.visualization import ImageNormalize, PowerStretch, LogStretch
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.wcs import FITSFixedWarning
from astroquery.jplhorizons import Horizons
from casatools import msmetadata, ms as casamstool, table
from datetime import datetime as dt, timedelta
from dask import delayed
from PIL import Image
from collections import namedtuple
from .basic_utils import *
from .image_utils import *
from .proc_manage_utils import *
from .ms_metadata import *
from .mwa_utils import *

warnings.simplefilter("ignore", category=FITSFixedWarning)


#################################
# Plotting related functions
#################################
def plot_ms_diagnostics(
    msname, outdir="", dask_client=None, cpu_frac=0.8, mem_frac=0.8
):
    """
    Plot diagonistics plots for measurement set

    Parameters
    ----------
    msname : str
        Measurement set
    outdir : str, optional
        Output directory
    dask_client : dask.client
        Dask client
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction

    Returns
    -------
    int
        Success message
    list
        Output plot file list
    """
    if outdir == "":
        outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    output_pdf = f"{outdir}/{os.path.basename(msname).split('.ms')[0]}_plots"
    output_pdf_list = glob.glob(f"{output_pdf}*.pdf")
    if len(output_pdf_list) > 0:
        return 0, output_pdf_list

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

    if cpu_frac > 0.8:
        cpu_frac = 0.8
    ncpu = max(1, int(psutil.cpu_count() * cpu_frac))
    if mem_frac > 0.8:
        mem_frac = 0.8
    total_mem = (psutil.virtual_memory().available * mem_frac) / (1024**3)  # In GB
    max_scan_size = max(scan_sizes)
    frac_chunk = min(1, total_mem / max_scan_size)
    nchunk = int(nrow * frac_chunk)
    output_pdf_list = []
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
            "phase": "Phase (deg)",
            "real": "Real",
            "imag": "Imaginary",
        }

        # Define x-axis settings
        xaxes = {"uv": ("UV(m)",), "FREQ": ("Frequency (GHz)",), "TIME": ("Time",)}

        for corr, do_plot in corr_sets:
            if not do_plot:
                continue
            for yaxis, ylabel in plot_types.items():
                for xaxis, (xlabel,) in xaxes.items():
                    for col in ["CORRECTED_DATA", "CORRECTED_DATA-MODEL_DATA"]:
                        cmds.append(
                            f"shadems --no-lim-save --xaxis {xaxis} --yaxis {yaxis} "
                            f"--col {col} -j {ncpu} -z {nchunk} "
                            f"--xlabel '{xlabel}' --ylabel '{ylabel}' "
                            f"--corr {corr} --colour-by CORR --iter-scan --iter-field "
                            f"--dmap tab10 {msname}"
                        )

        print(f"Making plots of: {msname}")
        for cmd in cmds:
            run_shadems(cmd, verbose=False)

        for yaxis, ylabel in plot_types.items():
            #########################
            # Making plots
            #########################
            pngs = glob.glob(f"*{yaxis}*.png")
            outfile = f"{output_pdf}_{yaxis}.pdf"
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
        os.system(f"rm -rf log-shadems.txt")


def plot_caltable_diagnostics(caltable, outdir=""):
    """
    Plot diagonistic plot of a caltable

    Parameters
    ----------
    caltable : str
        Caltable name
    outdir : str, optional
        Output directory

    Returns
    -------
    int
        Success messsage
    str
        Output file
    """
    caltable = caltable.rstrip("/")
    if outdir == "":
        outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    output_pdf = f"{outdir}/{os.path.basename(caltable)}_plots.pdf"
    if os.path.exists(output_pdf):
        return 0, output_pdf
    pols = ["X", "Y"]
    ncols = 3
    nrows = 3
    plots_per_fig = ncols * nrows
    out_files = []
    try:
        tb = table()
        tb.open(f"{caltable}/SPECTRAL_WINDOW")
        freqs = tb.getcol("CHAN_FREQ") / 10**9  # In GHz
        tb.close()
        tb.open(caltable)
        cal_type = tb.getkeywords()["VisCal"]
        if cal_type == "K Jones":
            gain = tb.getcol("FPARAM")
            flag = tb.getcol("FLAG")
        else:
            gain = tb.getcol("CPARAM")
            flag = tb.getcol("FLAG")
        gain[flag] = np.nan
        ants = np.unique(tb.getcol("ANTENNA1"))
        times = np.unique(tb.getcol("TIME"))
        nant = np.nanmax(ants) + 1
        tb.close()
        print(f"Ploting {cal_type}")
        if cal_type == "K Jones":
            plt.figure(figsize=(15, 10))
            gain = np.nanmean(gain, axis=1)
            for i in range(2):
                plt.scatter(
                    range(gain.shape[-1]), gain[i, ...], label=f"Pol: {pols[i]}"
                )
            plt.xlabel("Antenna index", fontsize=14)
            plt.ylabel("Delay (ns)", fontsize=14)
            plt.title("Antenna vs Delay", fontsize=14)
            plt.legend()
            plt.tight_layout()
            savefile = f"{caltable}.png"
            plt.savefig(savefile)
            plt.clf()
            out_files.append(savefile)
        else:
            if cal_type == "G Jones":
                ntime = int(gain.shape[-1] / nant)
                gain = gain.reshape(gain.shape[0], gain.shape[1], nant, ntime)
                gain = gain[:, 0, ...]
            elif cal_type == "T Jones":
                ntime = int(gain.shape[-1] / nant)
                gain = gain.reshape(gain.shape[0], gain.shape[1], nant, ntime)
                gain = gain[0, 0, ...]
            elif cal_type == "B Jones" or cal_type == "Df Jones":
                ntime = int(gain.shape[-1] / nant)
                gain = gain.reshape(gain.shape[0], gain.shape[1], nant, ntime)
                gain = np.nanmean(gain, axis=-1)
            else:
                print(f"{cal_type} is not implemented yet.")
                return
            for quantity in ["amp", "phase"]:
                if cal_type == "G Jones":
                    for idx in range(0, len(ants), plots_per_fig):
                        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
                        if quantity == "amp":
                            fig.suptitle("Time vs Gain Amplitude", fontsize=14)
                        else:
                            fig.suptitle("Time vs Gain Phase", fontsize=14)
                        axes = axes.flatten()
                        for i, ant in enumerate(ants[idx : idx + plots_per_fig]):
                            ax = axes[i]
                            for j in range(2):  # loop over polarizations
                                if quantity == "amp":
                                    ax.scatter(
                                        times - np.nanmin(times),
                                        np.abs(gain[j, ant, :]),
                                        label=f"Pol: {pols[j]}",
                                        s=14,
                                    )
                                    ax.set_ylabel("Gain Amplitude", fontsize=14)
                                else:
                                    ax.scatter(
                                        times - np.nanmin(times),
                                        np.angle(gain[j, ant, :], deg=True),
                                        label=f"Pol: {pols[j]}",
                                        s=14,
                                    )
                                    ax.set_ylabel("Gain Phase (degree)", fontsize=14)
                            ax.set_title(f"Antenna {ant+1}", fontsize=14)
                            ax.set_xlabel("Time (s)", fontsize=14)
                            ax.legend(fontsize=10)
                        for j in range(i + 1, plots_per_fig):
                            fig.delaxes(axes[j])
                        plt.tight_layout(rect=[0, 0, 1, 0.99])
                        savefile = f"{caltable}_gain_{quantity}_batch_{idx // plots_per_fig + 1}.png"
                        plt.savefig(savefile)
                        plt.close()
                        out_files.append(savefile)
                elif cal_type == "T Jones":
                    for idx in range(0, len(ants), plots_per_fig):
                        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
                        if quantity == "amp":
                            fig.suptitle("Time vs Gain Amplitude", fontsize=14)
                        else:
                            fig.suptitle("Time vs Gain Phase", fontsize=14)
                        axes = axes.flatten()
                        for i, ant in enumerate(ants[idx : idx + plots_per_fig]):
                            ax = axes[i]
                            if quantity == "amp":
                                ax.scatter(
                                    times - np.nanmin(times), np.abs(gain[ant, :])
                                )
                                ax.set_ylabel("Gain Amplitude", fontsize=14)
                            else:
                                ax.scatter(
                                    times - np.nanmin(times),
                                    np.angle(gain[ant, :], deg=True),
                                )
                                ax.set_ylabel("Gain Phase", fontsize=14)
                            ax.set_title(f"Antenna {ant+1}", fontsize=14)
                            ax.set_xlabel("Time (s)", fontsize=14)
                        for j in range(i + 1, plots_per_fig):
                            fig.delaxes(axes[j])
                        plt.tight_layout(rect=[0, 0, 1, 0.99])
                        savefile = f"{caltable}_gain_{quantity}_batch_{idx // plots_per_fig + 1}.png"
                        plt.savefig(savefile)
                        plt.close()
                        out_files.append(savefile)
                elif cal_type == "B Jones" or cal_type == "Df Jones":
                    for idx in range(0, len(ants), plots_per_fig):
                        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
                        if quantity == "amp":
                            fig.suptitle("Frequency vs Gain Amplitude", fontsize=14)
                        else:
                            fig.suptitle("Frequency vs Gain Phase", fontsize=14)
                        axes = axes.flatten()
                        for i, ant in enumerate(ants[idx : idx + plots_per_fig]):
                            ax = axes[i]
                            for j in range(2):
                                if quantity == "amp":
                                    ax.scatter(
                                        freqs,
                                        np.abs(gain[j, :, ant]),
                                        label=f"Pol: {pols[j]}",
                                        s=14,
                                    )
                                    ax.set_ylabel("Gain Amplitude", fontsize=14)
                                else:
                                    ax.scatter(
                                        freqs,
                                        np.angle(gain[j, :, ant], deg=True),
                                        label=f"Pol: {pols[j]}",
                                        s=14,
                                    )
                                    ax.set_ylabel("Gain Phase (degree)", fontsize=14)
                            ax.set_title(f"Antenna {ant+1}", fontsize=14)
                            ax.set_xlabel("Frequency (GHz)", fontsize=14)
                            ax.legend(fontsize=10)
                        for j in range(i + 1, plots_per_fig):
                            fig.delaxes(axes[j])
                        plt.tight_layout(rect=[0, 0, 1, 0.99])
                        savefile = f"{caltable}_gain_{quantity}_batch_{idx // plots_per_fig + 1}.png"
                        plt.savefig(savefile)
                        plt.close()
                        out_files.append(savefile)
        images = []
        for image in out_files:
            images.append(Image.open(image).convert("RGB"))
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        return 0, output_pdf
    except Exception:
        traceback.print_exc()
        return 1, ""
    finally:
        drop_cache(caltable)
        for png in out_files:
            os.system(f"rm -rf {png}")


def get_mwamap(fits_image, do_sharpen=False):
    """
    Make MWA sunpy map

    Parameters
    ----------
    fits_image : str
        MWA fits image
    do_sharpen : bool, optional
        Sharpen the image

    Returns
    -------
    sunpy.map
        Sunpy map
    """
    from scipy.ndimage import gaussian_filter
    from sunpy.map import make_fitswcs_header
    from sunpy.coordinates import frames, sun

    logging.getLogger("sunpy").setLevel(logging.ERROR)

    MWALAT = -26.703319  # degrees
    MWALON = 116.670815  # degrees
    MWAALT = 377.0  # meters
    mwa_hdu = fits.open(fits_image)  # Opening MWA fits file
    mwa_header = mwa_hdu[0].header  # mwa header
    mwa_data = mwa_hdu[0].data
    if len(mwa_data.shape) > 2:
        mwa_data = mwa_data[0, 0, :, :]  # mwa data
    if mwa_header["CTYPE3"] == "FREQ":
        frequency = mwa_header["CRVAL3"] * u.Hz
    elif mwa_header["CTYPE4"] == "FREQ":
        frequency = mwa_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    try:
        pixel_unit = mwa_header["BUNIT"]
    except BaseException:
        pixel_nuit = ""
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
        observatory="mwaKAT",
    )
    if do_sharpen:
        blurred = gaussian_filter(mwa_data, sigma=10)
        mwa_data = mwa_data + (mwa_data - blurred)
    mwa_map = Map(mwa_data, new_mwa_header)
    mwa_map_rotate = mwa_map.rotate()
    return mwa_map_rotate


def save_in_hpc(fits_image, outdir="", xlim=[-1600, 1600], ylim=[-1600, 1600]):
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
    mwamap = get_mwamap(fits_image)
    if len(xlim) == 2 and len(ylim) == 2:
        top_right = SkyCoord(
            xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=mwamap.coordinate_frame
        )
        bottom_left = SkyCoord(
            xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=mwamap.coordinate_frame
        )
        mwamap = mwamap.submap(bottom_left, top_right=top_right)
    if outdir == "":
        outdir = os.path.dirname(os.path.abspath(fits_image))
    outfile = f"{outdir}/{os.path.basename(fits_image).split('.fits')[0]}_HPC.fits"
    if os.path.exists(outfile):
        os.system(f"rm -rf {outfile}")
    mwamap.save(outfile, filetype="fits")
    data = fits.getdata(outfile)
    data = data[np.newaxis, np.newaxis, ...]
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
        "AUTHOR",
        "PIPELINE",
        "BAND",
        "MAX",
        "MIN",
        "RMS",
        "SUM",
        "MEAN",
        "MEDIAN",
        "RMSDYN",
        "MIMADYN",
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
    import matplotlib.ticker as ticker
    from matplotlib.patches import Ellipse, Rectangle
    from matplotlib.colors import ListedColormap
    from matplotlib import cm
    from sunpy.coordinates import sun

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    if not showgui:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    matplotlib.rcParams.update({"font.size": 12})
    fits_image = fits_image.rstrip("/")
    mwa_header = fits.getheader(fits_image)  # Opening mwaKAT fits file
    if mwa_header["CTYPE3"] == "FREQ":
        frequency = mwa_header["CRVAL3"] * u.Hz
    elif mwa_header["CTYPE4"] == "FREQ":
        frequency = mwa_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    try:
        pixel_unit = mwa_header["BUNIT"]
    except BaseException:
        pixel_nuit = ""
    pixel_scale = abs(mwa_header["CDELT1"])*3600.0 # In arcsec
    obstime = Time(mwa_header["date-obs"])
    mwa_map_rotate = get_mwamap(fits_image)
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
                width=bmin/pixel_scale,
                height=bmaj/pixel_scale,
                angle=bpa,
                edgecolor="white",
                facecolor="white",
                lw=1,
            )
            ax.add_patch(beam_ellipse)
            # Draw square box around the ellipse
            box_size = max(0.2*(x1-x0),1.5*max(bmin,bmaj))/pixel_scale  # slightly bigger than beam
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
                    + os.path.basename(fits_image).split(".fits")[0]
                    + f"_contour.{ext}"
                )
            else:
                output_image = (
                    outdir
                    + "/"
                    + os.path.basename(fits_image).split(".fits")[0]
                    + f".{ext}"
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


def get_aia_map(obs_date, obs_time, workdir, aia_wavelength=193, keep_aia_fits=False):
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
    aia_wavelength : float, optional
        Wavelength, options: 94, 131, 171, 193, 211, 304, 335 Å
    keep_aia_fits : bool, optional
        Keep AIA fits file or not

    Returns
    -------
    sunpy.map
        Sunpy AIAMap
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
        pos = np.argmin(np.abs(aia_wavelength-aia_wavelengths))
        aia_wavelength = aia_wavelengths[pos]
    os.makedirs(workdir, exist_ok=True)
    start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
    t_start = start_time.strftime("%Y-%m-%dT%H:%M")
    time = a.Time(t_start, t_start)
    instrument = a.Instrument("aia")
    jsoc_wavelength = a.Wavelength(aia_wavelength * u.angstrom)
    results = Fido.search(time,a.jsoc.Series('aia.lev1_euv_12s'),a.jsoc.Notify("paircarsnotification@gmail.com"),jsoc_wavelength)
    num_files = results.file_num
    if num_files==0:
        return
    else:
        downloaded_files = Fido.fetch(results, path=workdir, progress=False, overwrite=False)
        if len(downloaded_files)>0:
            final_image = downloaded_files[0]
            aia_map = Map(final_image)
            # Step 1: Pointing correction
            try:    
                pointing_corrected_map=update_pointing(aia_map)
            except:
                pointing_corrected_map=aia_map
            # Step 2: register (we are skipping PSF deconvolution)
            registered_map = register(pointing_corrected_map)
            # Step 3: instrument degradation correction
            try:
                corrected_map=correct_degradation(registered_map)
            except:
                corrected_map=registered_map
            # Step 4: Normalize by exposure time
            normalized_data = (
                corrected_map.data / corrected_map.exposure_time.to(u.s).value
            )
            normalized_map = Map(normalized_data, corrected_map.meta)
            if keep_aia_fits is False:
                os.system(f"rm -rf {final_image}")
            return normalized_map
        else:
            return
    
    
def get_suvi_map(obs_date, obs_time, workdir, suvi_wavelength=195, keep_suvi_fits=False):
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
    suvi_wavelength : float, optional
        Wavelength, options: 94, 131, 171, 195, 284, 304 Å
    keep_suvi_fits : bool, optional
        Keep SUVI fits file or not

    Returns
    -------
    sunpy.map
        Sunpy SUVIMap
    """
    def list_url_directory(url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
        
    logging.getLogger("sunpy").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="This download has been started in a thread which is not the main thread",
    )
    suvi_wavelengths=np.array([94,131,171,195,284,304])
    if suvi_wavelength not in suvi_wavelengths:
        pos = np.argmin(np.abs(suvi_wavelength-suvi_wavelengths))
        suvi_wavelength = suvi_wavelengths[pos]
    os.makedirs(workdir, exist_ok=True)

    baseurl1 = 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes'
    baseurl2 = 'l2/data'
    ext = '.fits'

    spacecraft_numbers = [16, 18]
    wvln_path = dict({94:'suvi-l2-ci094', 131:'suvi-l2-ci131', 171:'suvi-l2-ci171', \
                      195:'suvi-l2-ci195', 284:'suvi-l2-ci284', 304:'suvi-l2-ci304'})
    date_str = "/".join(obs_date.split("-"))
    all_files  = []
    start_times = []
    out_files = []
        
    for spacecraft in spacecraft_numbers:
        url = f"{baseurl1}{spacecraft}/{baseurl2}/{wvln_path[suvi_wavelength]}/{date_str}/"
        request = requests.get(url)
        if not request.status_code == 200:
            pass
        else:
            for file_name in list_url_directory(url, ext):
                all_files.append(file_name)
                file_base = os.path.basename(file_name)
                out_files.append(file_base)
                start_times.append(file_base.split("_")[-3])
            times_dt = [dt.strptime(t, "s%Y%m%dT%H%M%Sz") for t in start_times]
            start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
            closest_time = min(times_dt, key=lambda t: abs(t - start_time))
            pos = times_dt.index(closest_time)
            download_url = all_files[pos]
            out_file = out_files[pos]
            if os.path.exists(out_file) is False:
                dl = Downloader()
                dl.enqueue_file(download_url, path=out_file)
                downloaded_files = dl.download()
                if len(downloaded_files)>0:
                    final_image=downloaded_files[0]
            else:
                final_image = out_file
            suvi_map = Map(final_image)
            if keep_suvi_fits is False:
                os.system(f"rm -rf {final_image}")
            return suvi_map
    return 


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


def make_mwa_overlay(
    mwa_image,
    wavelength=195,
    plot_file_prefix=None,
    plot_mwa_colormap=True,
    enhance_offdisk=False,
    contour_levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    euv_image_scaling=0.25,
    do_sharpen_euv=True,
    xlim=[-2500, 2500],
    ylim=[-2500, 2500],
    extensions=["png"],
    outdirs=[],
    ncpu=-1,
    keep_euv_fits=False,
    showgui=False,
    verbose=False,
):
    """
    Make overlay of MWA image on GOES SUVI/ SDO AIA image

    Parameters
    ----------
    mwa_image : str
        MWA image
    wavelength : float, optional
        GOES SUVI/ SDO AIA wavelength, options: 94, 131, 171, 195(193), 284, 304 Å
    plot_file_prefix : str, optional
        Plot file prefix name
    plot_mwa_colormap : bool, optional
        Plot MWA map colormap
    enhance_offdisk : bool, optional
        Enhance off-disk emission
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
    ncpu : int, optional
        Number of CPUs to use
    keep_euv_fits : bool, optional
        Keep EUV fits file
    showgui : bool, optional
        Show GUI
    verbose: bool, optinal
        Verbose output

    Returns
    -------
    list
        Plot file names
    """
    import matplotlib
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    from sunpy.coordinates import SphericalScreen
    from matplotlib.colors import ListedColormap
    from matplotlib import cm
    from sunpy.map import make_fitswcs_header

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    logging.getLogger("reproject.common").setLevel(logging.WARNING)

    @delayed
    def reproject_map(smap, target_header):
        with SphericalScreen(smap.observer_coordinate):
            return smap.reproject_to(target_header)
           
    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    workdir = os.path.dirname(os.path.abspath(mwa_image))
    mwamap = get_mwamap(mwa_image)
    obs_datetime = fits.getheader(mwa_image)["DATE-OBS"]
    obs_date = obs_datetime.split("T")[0]
    year = int(obs_date.split("-")[0])
    obs_time = ":".join(obs_datetime.split("T")[-1].split(":")[:2])
    if year>=2019:
        euv_map = get_suvi_map(obs_date, obs_time, workdir, suvi_wavelength=wavelength, keep_suvi_fits=keep_euv_fits)
    else:
        euv_map=None
    if euv_map is None:
        euv_map = get_aia_map(obs_date, obs_time, workdir, aia_wavelength=wavelength, keep_aia_fits=keep_euv_fits)
        if euv_map is None:
            print ("Could not get either SUVI or AIA images.")
            return
    if enhance_offdisk:
        euv_map = enhance_offlimb(euv_map, do_sharpen=do_sharpen_euv)
    
    projected_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        obstime=mwamap.observer_coordinate.obstime,
        frame="helioprojective",
        observer=mwamap.observer_coordinate,
        rsun=mwamap.coordinate_frame.rsun,)
    mwa_header=mwamap.meta
    euv_header=euv_map.meta
    
    euv_pix = max(1024,int(euv_header["naxis1"]*euv_image_scaling))
    euv_current_fov=euv_header["naxis1"]*euv_header["cdelt1"] 
    mwa_image_fov=mwa_header["naxis1"]*mwa_header["cdelt1"] 
  
    new_scale = float(mwa_image_fov/euv_pix)* u.arcsec/u.pix
    SpatialPair=namedtuple("SpatialPair","axis1 axis2")
    new_scale = SpatialPair(axis1=new_scale, axis2=new_scale)
    new_shape = (euv_pix,euv_pix)
    
    projected_header = make_fitswcs_header(
        new_shape,
        projected_coord,
        scale=u.Quantity(new_scale),
        instrument=euv_map.instrument,
        wavelength=euv_map.wavelength,
    )
    
    reprojected = [
        reproject_map(mwamap, projected_header),
        reproject_map(euv_map, projected_header),
    ]
        
    if ncpu < 1:
        ncpu = 1
    pool = ThreadPool(processes=ncpu)
    with dask.config.set(pool=pool):
        mwa_reprojected, euv_reprojected = compute(*reprojected, scheduler="threads")
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

        title = f"EUV time: {euvtime}\n MWA time: {mwatime}"
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
            left=0.1,    # space from left edge
            right=0.98,   # space from right edge
            bottom=0.08,  # space from bottom
            top=0.9,     # space from top
            wspace=0.27,  # horizontal space between panels
            hspace=0.05   # vertical space between panels
        )
        plot_file_list = []
        if verbose:
            print("#######################")
        if plot_file_prefix:
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
            if verbose:
                print("#######################\n")
        else:
            plot_file = None
        if showgui:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
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
    else:
        matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 14})
    scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    valid_scans = get_valid_scans(msname)
    filtered_scans = []
    for scan in scans:
        if scan in valid_scans:
            filtered_scans.append(scan)
    msmd = msmetadata()
    msmd.open(msname)
    tstart_mjd = min(msmd.timesforscan(int(min(filtered_scans))))
    tend_mjd = max(msmd.timesforscan(int(max(filtered_scans))))
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
    plt.tight_layout()
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
    imagedir="",
    pol="",
    cutout_rsun=4.0,
    make_overlay=True,
    make_plots=True,
    keep_euv_fits=False,
):
    """
    Rename and move image to image directory

    Parameters
    ----------
    imagename : str
        Image name
    imagedir : str, optional
        Image directory (default given image directory)
    pol : str, optional
        Stokes parameters
    cutout_rsun : float, optional
        Cutout in solar radii from center (default: 4.0 solar radii)
    make_overlay : bool, optional
        Make overlay on SUVI/AIA
    make_plots : bool, optional
        Make radio map plot in helioprojective coordinates
    keep_euv_fits : bool, optional
        Keep EUV images or not

    Returns
    -------
    str
        New imagename with full path
    """
    imagename = imagename.rstrip("/")
    imagename = cutout_image(
        imagename, imagename, x_deg=(cutout_rsun * 2 * 16.0) / 60.0
    )
    maxval, minval, rms, total_val, mean_val, median_val, rms_dyn, minmax_dyn = (
        calc_solar_image_stat(imagename, disc_size=35)
    )
    if np.isnan(rms_dyn):
        os.system(f"rm -rf {imagename}")
        return
    
    header = fits.getheader(imagename)
    time = header["DATE-OBS"]
    astro_time = Time(time, scale="utc")
    sun_jpl = Horizons(id="10", location="500", epochs=astro_time.jd)
    eph = sun_jpl.ephemerides()
    sun_coords = SkyCoord(
        ra=eph["RA"][0] * u.deg, dec=eph["DEC"][0] * u.deg, frame="icrs"
    )
    
    with fits.open(imagename, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["AUTHOR"] = "DevojyotiKansabanik"
        hdr["PIPELINE"] = "P-AIRCARS"
        hdr["CRVAL1"] = sun_coords.ra.deg
        hdr["CRVAL2"] = sun_coords.dec.deg
        hdr["MAX"] = maxval
        hdr["MIN"] = minval
        hdr["RMS"] = rms
        hdr["SUM"] = total_val
        hdr["MEAN"] = mean_val
        hdr["MEDIAN"] = median_val
        hdr["RMSDYN"] = rms_dyn
        hdr["MIMADYN"] = minmax_dyn
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
    if make_overlay:
        try:
            overlay_pngdir = f"{os.path.dirname(imagedir)}/overlays_pngs"
            os.makedirs(overlay_pngdir, exist_ok=True)
            outimages = make_mwa_overlay(
                new_name,
                plot_file_prefix=os.path.basename(new_name).split(".fits")[0]
                + "_euv_mwa_overlay",
                extensions=["png"],
                outdirs=[overlay_pngdir],
                keep_euv_fits=keep_euv_fits,
                verbose=False,
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
    from matplotlib.gridspec import GridSpec

    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 18})
    if type(dsfiles) == str:
        dsfiles = [dsfiles]
    for i, dsfile in enumerate(dsfiles):
        freqs_i, times_i, timestamps_i, T_data_i, S_data_i = np.load(
            dsfile, allow_pickle=True
        )
        if plot_quantity == "TB":
            data_i = T_data_i / 10**6
        else:
            data_i = S_data_i
        if i == 0:
            freqs = freqs_i
            times = times_i
            timestamps = timestamps_i
            data = data_i
        else:
            df = np.nanmedian(np.diff(freqs))
            gapsize = int(np.round((np.nanmin(freqs_i) - np.nanmax(freqs)) / df))
            gapsize = 1  # max(gapsize, 0)

            if 0 < gapsize < 5:
                last_freq_median = np.nanmedian(data[-1, :])
                new_freq_median = np.nanmedian(data_i[0, :])
                if np.isfinite(new_freq_median) and new_freq_median != 0:
                    data_i = (data_i / new_freq_median) * last_freq_median

            if gapsize > 0:
                gap = np.full((gapsize, data.shape[1]), np.nan)
                data = np.concatenate([data, gap, data_i], axis=0)
                freqs = np.append(freqs, np.full(gapsize, np.nan))
            else:
                data = np.concatenate([data, data_i], axis=0)

            freqs = np.append(freqs, freqs_i)

    ########################################
    # Time and frequency range
    ########################################
    median_bandshape = np.nanmedian(data, axis=-1)
    pos = np.where(np.isnan(median_bandshape) == False)[0]
    if len(pos) > 0:
        data = data[min(pos) : max(pos), :]
        freqs = freqs[min(pos) : max(pos)]
    temp_times = times[np.isnan(times) == False]
    maxtimepos = np.argmax(temp_times)
    mintimepos = np.argmin(temp_times)
    datestamp = f"{timestamps[mintimepos].split('T')[0]}"
    tstart = f"{timestamps[mintimepos].split('T')[0]} {':'.join(timestamps[mintimepos].split('T')[-1].split(':')[:2])}"
    tend = f"{timestamps[maxtimepos].split('T')[0]} {':'.join(timestamps[maxtimepos].split('T')[-1].split(':')[:2])}"
    results = Fido.search(
        a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
    )
    files = Fido.fetch(results, path=os.path.dirname(dsfiles[0]), overwrite=False)
    goes_tseries = TimeSeries(files, concatenate=True)
    for goes_f in files:
        os.system(f"rm -rf {goes_f}")
    goes_tseries = goes_tseries.truncate(tstart, tend)
    timeseries = np.nanmean(data, axis=0)
    # Normalization
    data_std = np.nanstd(data)
    data_median = np.nanmedian(data)
    norm = ImageNormalize(
        data,
        stretch=LogStretch(1),
        vmin=0.99 * np.nanmin(data),
        vmax=0.99 * np.nanmax(data),
    )
    try:
        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(
            nrows=3, ncols=2, width_ratios=[1, 0.03], height_ratios=[4, 1.5, 2]
        )
        # Axes
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_ts = fig.add_subplot(gs[1, 0])
        ax_goes = fig.add_subplot(gs[2, 0])
        cax = fig.add_subplot(gs[:, 1])  # colorbar spans both rows
        # Plot dynamic spectrum
        im = ax_spec.imshow(
            data, aspect="auto", origin="lower", norm=norm, cmap="magma"
        )
        ax_spec.set_ylabel("Frequency (MHz)")
        ax_spec.set_xticklabels([])  # Remove x-axis labels from top plot
        # Y-ticks
        yticks = ax_spec.get_yticks()
        yticks = yticks[(yticks >= 0) & (yticks < len(freqs))]
        ax_spec.set_yticks(yticks)
        ax_spec.set_yticklabels([f"{freqs[int(i)]:.1f}" for i in yticks])
        # Plot time series
        ax_ts.plot(timeseries)
        ax_ts.set_xlim(0, len(timeseries) - 1)
        if plot_quantity == "TB":
            ax_ts.set_ylabel("TB (MK)")
        else:
            ax_ts.set_ylabel("S (SFU)")
        goes_tseries.plot(axes=ax_goes)
        goes_times = goes_tseries.time
        times_dt = goes_times.to_datetime()
        ax_goes.set_xlim(times_dt[0], times_dt[-1])
        ax_goes.set_ylabel(r"Flux ($\frac{W}{m^2}$)")
        ax_goes.legend(ncol=2, loc="upper right")
        ax_goes.set_title("GOES light curve", fontsize=14)
        ax_ts.set_title("MWA light curve", fontsize=14)
        ax_spec.set_title("MWA dynamic spectrum", fontsize=14)
        ax_goes.set_xlabel("Time (UTC)")
        # Format x-ticks
        ax_ts.set_xticks([])
        ax_ts.set_xticklabels([])
        # Colorbar
        cbar = fig.colorbar(im, cax=cax)
        if plot_quantity == "TB":
            cbar.set_label("Brightness temperature (MK)")
        else:
            cbar.set_label("Flux density (SFU)")
        plt.tight_layout()
        # Save or show
        if plot_file:
            plt.savefig(plot_file, bbox_inches="tight")
            print(f"Plot saved: {plot_file}")
        if showgui:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
    return plot_file


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
