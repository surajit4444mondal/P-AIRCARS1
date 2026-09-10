import numpy as np
import traceback
import warnings
import copy
import os
import io
from collections import defaultdict
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from .basic_utils import average_timestamp, timestamp_to_mjdsec
from .udocker_utils import run_wsclean

warnings.simplefilter("ignore", category=FITSFixedWarning)


##########################
# Image analysis related
##########################
def create_circular_mask(msname, cellsize, imsize, mask_radius=96):
    """
    Create fits solar mask

    Parameters
    ----------
    msname : str
        Name of the measurement set
    cellsize : float
        Cell size in arcsec
    imsize : int
        Imsize in number of pixels
    mask_radius : float
        Mask radius in arcmin (default: 4 solr radii)

    Returns
    -------
    str
        Fits mask file name
    """
    try:
        msname = msname.rstrip("/")
        imagename_prefix = (
            os.path.dirname(os.path.abspath(msname))
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_solar"
        )
        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-nwlayers 1",
            "-niter 0 -name " + imagename_prefix,
        ]
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        msg = run_wsclean(wsclean_cmd, "paircarswsclean", verbose=False)
        if msg == 0:
            radius = mask_radius * 60 / cellsize
            os.system(
                "cp -r "
                + imagename_prefix
                + "-image.fits mask-"
                + os.path.basename(imagename_prefix)
                + ".fits"
            )
            os.system("rm -rf " + imagename_prefix + "*")
            data = fits.getdata("mask-" + os.path.basename(imagename_prefix) + ".fits")
            header = fits.getheader(
                "mask-" + os.path.basename(imagename_prefix) + ".fits"
            )
            maxpos = np.where(data[0, 0, ...] == np.nanmax(data[0, 0, ...]))
            y_cen, x_cen = maxpos[0][0], maxpos[1][0]
            mask = create_circular_mask_array(
                data[0, 0, ...], radius, center_x=x_cen, center_y=y_cen
            )
            data[0, 0, ...][mask] = 1.0
            data[0, 0, ...][~mask] = 0.0
            fits.writeto(
                imagename_prefix + "-mask.fits",
                data=data,
                header=header,
                overwrite=True,
            )
            os.system("rm -rf mask-" + os.path.basename(imagename_prefix) + ".fits")
            if os.path.exists(imagename_prefix + "-mask.fits"):
                return imagename_prefix + "-mask.fits"
            else:
                print("Circular mask could not be created.")
                return
        else:
            print("Circular mask could not be created.")
            return
    except Exception:
        traceback.print_exc()
        return


def create_circular_mask_array(data, radius, center_x=None, center_y=None):
    """
    Creating circular mask of a Numpy array

    Parameters
    ----------
    data : numpy.array
        2D numpy array
    radius : int
        Radius in pixels

    Returns
    -------
    numpy.array
        Mask array
    """
    shape = data.shape
    center = (shape[0] // 2, shape[1] // 2)
    if center_x is None:
        center_x = center[1]
    if center_y is None:
        center_y = center[0]
    Y, X = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = (X - center_x) ** 2 + (Y - center_y) ** 2
    mask = dist_from_center <= radius**2
    return mask


def get_image_npol(imagename):
    """
    Get image cube number of polarisation

    Parameters
    ----------
    imagename : str
        Imagename

    Returns
    -------
    int
        Number of polarisation planes
    """
    header = fits.getheader(imagename)
    if header["CTYPE4"] == "STOKES":
        npol = int(header["NAXIS4"])
    elif header["CTYPE3"] == "STOKES":
        npol = int(header["NAXIS3"])
    else:
        npol = 1
    return npol


def check_valid_image(imagename):
    """
    Check whether the image is valid or not

    Parameters
    ----------
    imagename : str
        Image name

    Returns
    -------
    bool
        Whether valid image or not
    """
    data = fits.getdata(imagename)
    if np.nansum(data) == 0:
        return False
    else:
        return True
        
        
def calc_solar_image_stat(imagename, modelname="", disc_radius=50):
    """
    Calculate solar image dynamic range

    Parameters
    ----------
    imagename : str
        Fits image name
    modelname : str, optional
        Model image name (N.B.: If provided total model flux is also returned)
    disc_radius : float, optional
        Disc radius in arcmin (default : 50)

    Returns
    -------
    float
        Maximum value
    float
        Minimum value
    float
        RMS values
    float
        Total value
    float
        Mean value
    float
        Median value
    float
        RMS dynamic range
    float
        Min-max dynamic range
    float
        Total model flux
    """
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    total_pix = int(header["NAXIS1"])
    pix_size = abs(header["CDELT1"]) * 3600.0  # In arcsec
    radius = int((disc_radius * 60) / pix_size)
    if radius > total_pix:
        radius = total_pix / 4.0
    if data.ndim == 4:
        data = data[0, 0, ...]
    elif data.ndim == 3:
        data = data[0, ...]
    else:
        data = data
    center_y, center_x = np.where(data == np.nanmax(data))
    mask = create_circular_mask_array(
        data, radius, center_x=center_x[0], center_y=center_y[0]
    )
    masked_data = copy.deepcopy(data)
    masked_data[mask] = np.nan
    unmasked_data = copy.deepcopy(data)
    unmasked_data[~mask] = np.nan
    maxval = float(np.nanmax(unmasked_data))
    minval = float(np.nanmin(masked_data))
    rms = float(np.nanstd(masked_data))
    total_val = float(np.nansum(unmasked_data))
    if rms != 0:
        rms_dyn = float(maxval / rms)
    else:
        rms_dyn = np.nan
    if abs(minval) != 0:
        minmax_dyn = float(maxval / abs(minval))
    else:
        minmax_dyn = np.nan
    mean_val = float(np.nanmean(unmasked_data))
    median_val = float(np.nanmedian(unmasked_data))
    del data, mask, unmasked_data, masked_data
    if modelname != "" and os.path.exists(modelname):
        modeldata = fits.getdata(modelname)
        total_model_flux = round(np.nansum(modeldata), 2)
    else:
        total_model_flux = np.nan
    return (
        round(maxval, 2),
        round(minval, 2),
        round(rms, 2),
        round(total_val, 2),
        round(mean_val, 2),
        round(median_val, 2),
        round(rms_dyn, 2),
        round(minmax_dyn, 2),
        total_model_flux,
    )


def generate_tb_map(imagename, outfile=""):
    """
    Function to generate brightness temperature map

    Parameters
    ----------
    imagename : str
        Name of the flux calibrated image
    outfile : str, optional
        Output brightess temperature image name

    Returns
    -------
    str
        Output image name
    """
    if outfile == "":
        outfile = imagename.split(".fits")[0] + "_TB.fits"
    image_header = fits.getheader(imagename)
    image_data = fits.getdata(imagename)
    major = float(image_header["BMAJ"]) * 3600.0  # In arcsec
    minor = float(image_header["BMIN"]) * 3600.0  # In arcsec
    if image_header["CTYPE3"] == "FREQ":
        freq = image_header["CRVAL3"] / 10**9  # In GHz
    elif image_header["CTYPE4"] == "FREQ":
        freq = image_header["CRVAL4"] / 10**9  # In GHz
    else:
        print(f"No frequency information is present in header for {imagename}.")
        return
    TB_conv_factor = (1.222e6) / ((freq**2) * major * minor)
    TB_data = image_data * TB_conv_factor
    image_header["BUNIT"] = "K"
    fits.writeto(outfile, data=TB_data, header=image_header, overwrite=True)
    return outfile


def cutout_image(fits_file, output_file, x_deg=2):
    """
    Cutout central part of the image

    Parameters
    ----------
    fits_file : str
        Input fits file
    output_file : str
        Output fits file name (If same as input, input image will be overwritten)
    x_deg : float, optional
        Size of the output image in degree

    Returns
    -------
    str
        Output image name
    """
    hdu = fits.open(fits_file)[0]
    data = hdu.data  # shape: (nfreq, nstokes, ny, nx)
    header = hdu.header
    _, _, ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    # Get pixel scale (deg/pixel)
    pix_scale_deg = np.abs(header["CDELT1"])
    x_pix = int((x_deg / pix_scale_deg) / 2)
    # Adjust if cutout size exceeds image size
    max_half_x = nx // 2
    max_half_y = ny // 2
    x_pix = min(x_pix, max_half_x)
    y_pix = min(x_pix, max_half_y)  # Assume square pixels
    # Define slice indices
    x0 = center_x - x_pix
    x1 = center_x + x_pix
    y0 = center_y - y_pix
    y1 = center_y + y_pix
    # Slice data
    cutout_data = data[:, :, y0:y1, x0:x1]
    # Update header
    new_header = header.copy()
    new_header["NAXIS1"] = x1 - x0
    new_header["NAXIS2"] = y1 - y0
    new_header["CRPIX1"] -= x0
    new_header["CRPIX2"] -= y0
    # Save
    fits.writeto(output_file, cutout_data, header=new_header, overwrite=True)
    return output_file


def make_timeavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a time averaged image

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    timestamps = []
    data = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        image_data = fits.getdata(image)
        if len(data) == 0:
            data.append(image_data)
        else:
            last_data = data[-1]
            if image_data.shape == last_data.shape:
                data.append(image_data)
            else:
                print(
                    f"Image data shape: {image_data.shape} does not match with last data: {last_data.shape}"
                )
        timestamps.append(fits.getheader(image)["DATE-OBS"])
    data = np.array(data)
    data = np.nanmean(data, axis=0)
    avg_timestamp = average_timestamp(timestamps)
    header = fits.getheader(wsclean_images[0])
    header["DATE-OBS"] = avg_timestamp
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def make_freqavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a frequency averaged image

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    freqs = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        header = fits.getheader(image)
        if header["CTYPE3"] == "FREQ":
            freqs.append(float(header["CRVAL3"]))
            freqaxis = 3
        elif header["CTYPE4"] == "FREQ":
            freqs.append(float(header["CRVAL4"]))
            freqaxis = 4
    data /= len(wsclean_images)
    if len(freqs) > 0:
        mean_freq = np.nanmean(freqs)
        width = max(freqs) - min(freqs)
        header = fits.getheader(wsclean_images[0])
        if freqaxis == 3:
            header["CRAVL3"] = mean_freq
            header["CDELT3"] = width
        elif freqaxis == 4:
            header["CRAVL4"] = mean_freq
            header["CDELT4"] = width
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def make_stokes_wsclean_imagecube(
    wsclean_images, outfile_name, keep_wsclean_images=True
):
    """
    Convert WSClean images into a Stokes cube image.

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    stokes = sorted(
        set(
            (
                os.path.basename(i).split(".fits")[0].split(" - ")[-2]
                if " - " in i
                else "I"
            )
            for i in wsclean_images
        )
    )
    valid_stokes = [
        {"I"},
        {"I", "V"},
        {"I", "Q", "U", "V"},
        {"XX", "YY"},
        {"LL", "RR"},
        {"Q", "U"},
        {"I", "Q"},
    ]
    if set(stokes) not in valid_stokes:
        print("Invalid Stokes combination.")
        return
    imagename_prefix = "temp_" + os.path.basename(wsclean_images[0]).split(" - I")[0]
    imagename_prefix + ".image"
    data, header = fits.getdata(wsclean_images[0]), fits.getheader(wsclean_images[0])
    for img in wsclean_images[1:]:
        data = np.append(data, fits.getdata(img), axis=0)
    header.update(
        {"NAXIS4": len(stokes), "CRVAL4": 1 if "I" in stokes else -5, "CDELT4": 1}
    )
    imagename_prefix + ".fits"
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def filter_images(imagelist, min_time_sep=60.0):
    """
    Select images with maximum bandwidth, then for each frequency
    keep images separated by at least `min_time_sep` seconds.

    Parameters
    ----------
    imagelist : list
        Image list
    min_time_sep : float, optional
        Minimum time seperation in seconds

    Returns
    -------
    list
        Filtered image list
    """
    image_info = []
    try:
        for image in imagelist:
            header = fits.getheader(image)
            bw = -1
            freq = -1
            if header.get("CTYPE3") == "FREQ":
                bw = abs(float(header.get("CDELT3", -1))) / 1e6
                freq = float(header.get("CRVAL3", -1))
            elif header.get("CTYPE4") == "FREQ":
                bw = abs(float(header.get("CDELT4", -1))) / 1e6
                freq = float(header.get("CRVAL4", -1))
            bw = round(bw, 2)
            timeobs = header["DATE-OBS"].split(".")[0]
            mjdsec = timestamp_to_mjdsec(timeobs, date_format=1)
            image_info.append(
                {"image": image, "bw": bw, "freq": freq, "mjdsec": mjdsec}
            )
        bws = np.array([info["bw"] for info in image_info])
        max_bw = np.max(bws)
        image_info = [info for info in image_info if info["bw"] == max_bw]
        freq_groups = defaultdict(list)
        for info in image_info:
            freq_groups[info["freq"]].append(info)
        final_images = []
        for freq, group in freq_groups.items():
            group = sorted(group, key=lambda x: x["mjdsec"])
            last_time = -np.inf
            if min_time_sep > 0:
                for info in group:
                    if info["mjdsec"] - last_time >= min_time_sep:
                        final_images.append(info["image"])
                        last_time = info["mjdsec"]
            else:
                info = group[int(len(group) / 2)]
                final_images.append(info["image"])
        return sorted(final_images)
    except Exception:
        print("Error in filtering out images.")
        traceback.print_exc()
        return []


#############################################
# Image compression and decompression
#############################################


def compress_fits(
    input_file,
    outputdir="",
    output_file="",
    quantize_level=4,
    keep_original=True,
):
    """
    Compress a FITS image using FITS tile compression with compression type RICE_1.
    N.B.: Quantization noise is saved in header per Stokes plane, assuming either it is only Stokes I or Stokes IQUV image

    Parameters
    ----------
    input_file : str
        Original FITS file.
    outputdir : str, optional
        Output directory
    output_file : str, optional
        Output file name
    quantize_level : float or None
        Lossy floating-point quantization.
    keep_original : bool, optional
        Keep original fits or not

    Returns
    -------
    int
        Success message
    str
        Compressed file name
    """
    quantize_level = max(1, quantize_level)
    if output_file == "":
        output_file = f"{input_file.split('.fits')[0]}_compressed.fits"
    if outputdir != "":
        os.makedirs(outputdir, exist_ok=True)
        output_file = f"{outputdir}/{os.path.basename(output_file)}"
    try:
        buffer = io.BytesIO()
        compressed_image = False
        with fits.open(input_file, memmap=True) as hdul:
            for i, hdu in enumerate(hdul):
                if (
                    type(hdu).__name__ == "CompImageHDU"
                    or hdu.name == "COMPRESSED_IMAGE"
                ):
                    compressed_image = True
            if compressed_image:
                print(f"{input_file} is a compressed fits.")
                return 1, input_file
            # Preserve original data and header
            data = hdul[0].data
            header = hdul[0].header.copy()
            # Create compressed image
            kwargs = {
                "data": data,
                "header": header,
                "compression_type": "RICE_1",
                "quantize_level": quantize_level,
            }
            if quantize_level is not None:
                kwargs["quantize_level"] = quantize_level
            compressed_hdu = fits.CompImageHDU(**kwargs)
            compressed_hdu.writeto(buffer)
            buffer.seek(0)
            with fits.open(buffer, memmap=False) as hdul:
                if isinstance(hdul[0], fits.CompImageHDU):
                    restored = hdul[0].data
                else:
                    restored = hdul[1].data
            data_shape = data.shape
            header_keys = compressed_hdu.header.keys()
            if "RUNCMD" in header_keys:
                run_cmd = compressed_hdu.header["RUNCMD"]
                del compressed_hdu.header["RUNCMD"]
            else:
                run_cmd = ""
            compressed_hdu.header["COMPRESS"] = "RICE_1"
            compressed_hdu.header["QUANLEVE"] = quantize_level
            for i in range(data_shape[0]):
                qs_noise = np.nanstd(data[i, 0, ...] - restored[i, 0, ...])
                if i == 1:
                    key = "QNOISE_Q"
                elif i == 2:
                    key = "QNOISE_U"
                elif i == 3:
                    key = "QNOISE_V"
                else:
                    key = "QNOISE_I"
                compressed_hdu.header[key] = qs_noise
            if run_cmd != "":
                compressed_hdu.header["RUNCMD"] = run_cmd
            compressed_hdu.writeto(output_file, overwrite=True)
        if not keep_original and os.path.exists(output_file):
            os.system(f"rm -rf {input_file}")
        if os.path.exists(output_file):
            return 0, output_file
        else:
            print(
                f"Compression did not show error, but compressed file: {output_file} does not exist.\n"
            )
            return 1, input_file
    except Exception:
        print(f"Error occured in compressing: {input_file}.\n")
        traceback.print_exc()
        return 1, input_file
    finally:
        buffer.close()
        del buffer


def decompress_fits(
    compressed_file, outputdir="", output_file="", keep_compressed=True
):
    """
    Restore a compressed FITS image to a normal FITS image.

    Parameters
    ----------
    compressed_file : str
        Compressed FITS file.
    outputdir : str, optional
        Output directory
    output_file : str, optional
        Restored normal FITS file.
    keep_compressed : bool, optinal
        Keep original compressed fits or not

    Returns
    -------
    int
        Success message
    str
        Output decompressed file
    """
    if output_file == "":
        output_file = f"{compressed_file.split('.fits')[0].split('_compressed')[0]}_decompressed.fits"
    if outputdir != "":
        os.makedirs(outputdir, exist_ok=True)
        output_file = f"{outputdir}/{os.path.basename(output_file)}"
    compressed_image = False
    try:
        with fits.open(compressed_file, memmap=True) as hdul:
            for i, hdu in enumerate(hdul):
                if (
                    type(hdu).__name__ == "CompImageHDU"
                    or hdu.name == "COMPRESSED_IMAGE"
                ):
                    compressed_image = True
            if not compressed_image:
                print(f"{compressed_file} is not a compressed fits.")
                return 1, compressed_file
            # CompImageHDU is normally extension 1
            if isinstance(hdul[0], fits.CompImageHDU):
                compressed_hdu = hdul[0]
            else:
                compressed_hdu = hdul[1]
            data = compressed_hdu.data
            header = compressed_hdu.header.copy()
            # Make normal uncompressed FITS
            fits.writeto(output_file, data=data, header=header, overwrite=True)
        if os.path.exists(output_file):
            if not keep_compressed:
                os.system(f"rm -rf {compressed_file}")
            return 0, output_file
        else:
            print(
                f"Decompression did not show any error, but output file: {output_file} does not exist.\n"
            )
            return 1, compressed_file
    except Exception:
        print(f"Error occured in decompressing: {compressed_file}.\n")
        traceback.print_exc()
        return 1, compressed_file
