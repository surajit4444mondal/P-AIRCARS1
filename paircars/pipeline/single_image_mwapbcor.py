import os
import time
import traceback
import warnings
import argparse
import numpy as np
from numpy.linalg import inv
from astropy.io import fits
from astropy.coordinates import EarthLocation
from paircars.utils import *

warnings.filterwarnings("ignore")

datadir = get_datadir()
MWA_PB_file_paircars = f"{datadir}/mwa_full_embedded_element_pattern.h5"
sweet_spot_file_paircars = f"{datadir}/MWA_sweet_spots.npy"
haslam_map_paircars = f"{datadir}/haslam_map.fits"
MWALON = 116.67
MWALAT = -26.7
MWAALT = 377.8
MWAPOS = EarthLocation.from_geodetic(
    lon="116:40:14.93", lat="-26:42:11.95", height=377.8
)


def get_pbcor_image(
    imagename,
    outfile,
    metafits,
    MWA_PB_file="",
    sweet_spot_file="",
    iau_order=False,
    pb_jones_file="",
    save_pb_file="",
    interpolated=True,
    gridpoint=-1,
    nthreads=-1,
    restore=False,
):
    """
    Correct FITS image for MWA primary beam

    Parameters
    ----------
    imagename : str
        Name of the input file
    outfile : str
        Basename of the outputfile
    metafits : str
        Metafits file path
    MWA_PB_file : str, optional
        MWA primary beam file path
    sweet_spot_file : str, optional
        MWA sweet spot file
    iau_order : bool
        PB Jones in IAU order or not
    pb_jones_file : str
        Numpy file with primary beam jones matrices
    save_pb_file : str
        Save primary beam jones matrices for future use in this file
    interpolated : bool
        Calculate spatially interpolated beams or not
    gridpoint : int
        MWA gridpoint number (default : -1, provide if you do not have metafits file)
    nthreads : int
        Number of cpu threads use for parallel computing
    restore : bool
        Whether correct for MWA primary beam or restore the correction

    Returns
    -------
    str
        Output image name
    """
    if MWA_PB_file == "" or os.path.exists(MWA_PB_file) is False:
        MWA_PB_file = MWA_PB_file_paircars
    if sweet_spot_file == "" or os.path.exists(sweet_spot_file) is False:
        sweet_spot_file = sweet_spot_file_paircars
    nthreads = max(1, nthreads)
    outfile = os.path.basename(outfile)

    if restore == False:
        print(
            "Correcting image : {os.path.basename(imagename)} for MWA primary beam response.\n"
        )
    else:
        print(
            "Undo the correction image : {os.path.basename(imagename)} for MWA primary beam response.\n"
        )

    ###################################
    # Determining beamformer delays
    ###################################
    if metafits == "" or os.path.isfile(metafits) is False:
        if gridpoint == -1:
            print("Either provide correct metafits file or grid point number.")
            return 1
        else:
            sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
            delay = sweet_spots[int(gridpoint)][-1]
    else:
        metadata = fits.getheader(metafits)
        gridpoint = metadata["GRIDNUM"]
        sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
        delay = sweet_spots[int(gridpoint)][-1]

    ###############################
    # Reading image data and header
    ###############################
    imagename = imagename.rstrip("/")
    imagedata = fits.getdata(imagename)
    imageheader = fits.getheader(imagename)

    ##############################
    # Determining frequency
    ##############################
    if imageheader["CTYPE3"] == "FREQ":
        freq = imageheader["CRVAL3"]
    elif imageheader["CTYPE4"] == "FREQ":
        freq = imageheader["CRVAL4"]
    else:
        print(f"No frequency axis in image: {imagename}.")
        return

    ###############################
    # Determining stokes
    ##############################
    if imageheader["CTYPE3"] == "STOKES":
        stokesaxis = 3
    elif imageheader["CTYPE4"] == "STOKES":
        stokesaxis = 4
    else:
        stokesaxis = 1

    ####################################
    # Preparing data and data grid
    ####################################
    if stokes == "I":
        imagedata = np.repeat(imagedata, 4, 0)
    alt_az_array = get_azza_from_fits(fitsfile)

    if pb_jones_file == "" or os.path.exists(pb_jones_file) is False:
        jones_array = get_jones_array(
            90 - np.rad2deg(alt_az_array["za_rad"].flatten()),
            np.rad2deg(alt_az_array["astro_az_rad"].flatten()),
            freq,
            gridpoint,
            ncpu=nthreads,
            interpolated=interpolated,
            MWA_PB_file=MWA_PB_file,
            sweet_spot_file=sweet_spot_file,
            iau_order=iau_order,
        )
        if save_pb_file != "":
            print(f"Saving primary beam Jones matrices in: {save_pb_file}\n")
            np.save(save_pb_file, np.array([iau_order, jones_array], dtype="object"))
    elif pb_jones_file != "" and os.path.exists(pb_jones_file):
        print(f"Loading primary beam Jones matrices from : {pb_jones_file}\n")
        pb = np.load(pb_jones_file, allow_pickle=True)
        pb_jones_order = pb[0]
        jones_array = pb[1]
        expected_shape = (alt_az_array["astro_az_rad"].flatten().shape[0], 2, 2)
        if jones_array.shape != expected_shape or pb_jones_order != iau_order:
            if pb_jones_order != iau_order:
                print(
                    "Given primary beam convention does not match with intented convention. Re-estimating primary beam Jones\n"
                )
            else:
                print(
                    "Loaded primary beam Jones are of different shape. Re-estimating primary beam Jones.\n"
                )
            jones_array = get_jones_array(
                90 - np.rad2deg(alt_az_array["za_rad"].flatten()),
                np.rad2deg(alt_az_array["astro_az_rad"].flatten()),
                freq,
                gridpoint,
                ncpu=nthreads,
                interpolated=interpolated,
                MWA_PB_file=MWA_PB_file,
                sweet_spot_file=sweet_spot_file,
                iau_order=iau_order,
            )

    stokes = get_IQUV(fitsfile, stokesaxis=stokesaxis)
    Vij = get_inst_pols(stokes)
    Vij_reshaped = Vij.reshape(Vij.shape[0] * Vij.shape[1], 2, 2)
    jones_array_H = np.transpose(jones_array.conj(), axes=((0, 2, 1)))

    if restore is False:
        Vij_corrected = np.matmul(
            inv(jones_array), np.matmul(Vij_reshaped, inv(jones_array_H))
        )
    else:
        Vij_corrected = np.matmul(jones_array, np.matmul(Vij_reshaped, jones_array_H))

    Vij_reshaped = Vij_corrected.reshape(Vij.shape)
    B = B2IQUV(Vij_reshaped, iau_order=iau_order)

    if stokesaxis == 3:
        imagedata[0, 0, :, :] = np.real(B["I"])
        imagedata[0, 1, :, :] = np.real(B["Q"])
        imagedata[0, 2, :, :] = np.real(B["U"])
        imagedata[0, 3, :, :] = np.real(B["V"])
    elif stokesaxis == 4:
        imagedata[0, 0, :, :] = np.real(B["I"])
        imagedata[1, 0, :, :] = np.real(B["Q"])
        imagedata[2, 0, :, :] = np.real(B["U"])
        imagedata[3, 0, :, :] = np.real(B["V"])
    else:
        imagedata[0, 0, :, :] = np.real(B["I"])

    if os.path.exists(outfile):
        os.system(f"rm -rf {outfile}")
    fits.writeto(outfile, data=imagedata, header=imageheader, overwrite=True)
    print("Output image written to : {outfile}.\n")
    return outfile


def cli():
    parser = argparse.ArgumentParser(
        description="Correct images for MWA primary beam response"
    )

    parser.add_argument(
        "--imagename",
        required=True,
        help="Name of the image file",
    )
    parser.add_argument(
        "--metafits",
        required=True,
        help="Name of the metafits file",
    )
    parser.add_argument(
        "--MWA_PB_file",
        default="",
        help="MWA primary beam file",
    )
    parser.add_argument(
        "--sweetspot_file",
        default="",
        help="MWA primary beam sweetspot file path",
    )
    parser.add_argument(
        "--iau_order",
        action="store_true",
        help="PB Jones in IAU order",
    )
    parser.add_argument(
        "--gridpoint",
        type=int,
        default=-1,
        help="MWA sweet spot pointing number",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of CPU threads to use",
    )
    parser.add_argument(
        "--outfile",
        default="",
        help="Output file name",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the primary beam correction",
    )
    parser.add_argument(
        "--pb_jones_file",
        default="mwapb.npy",
        help="Input NumPy file of PB Jones matrices",
    )
    parser.add_argument(
        "--save_pb",
        default="mwapb.npy",
        help="Save PB Jones matrices to this file",
    )
    parser.add_argument(
        "--interpolated",
        action="store_true",
        help="Use interpolated beam model",
    )
    args = parser.parse_args()

    start_time = time.time()

    try:
        get_pbcor_image(
            args.imagename,
            args.outfile,
            args.metafits,
            MWA_PB_file=args.MWA_PB_file,
            sweet_spot_file=args.sweetspot_file,
            iau_order=args.iau_order,
            pb_jones_file=args.pb_jones_file,
            save_pb_file=args.save_pb,
            interpolated=args.interpolated,
            gridpoint=args.gridpoint,
            nthreads=args.num_threads,
            restore=args.restore,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = main()
    os._exit(result)
