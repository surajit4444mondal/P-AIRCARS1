import os
import time
import traceback
import warnings
import argparse
import sys
import glob
import numpy as np
from numpy.linalg import inv
from astropy.io import fits
from astropy.coordinates import EarthLocation
from paircars.utils.basic_utils import get_datadir
from paircars.utils.mwapb_utils import (
    get_azza_from_fits,
    get_IQUV,
    get_inst_pols,
    B2IQUV,
    get_jones_array,
)
from paircars.utils.selfcal_utils import correct_leakage

warnings.filterwarnings("ignore")

datadir = get_datadir()
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
    leakage_file="",
    iau_order=False,
    pb_jones_file="",
    save_pb=False,
    interpolated=True,
    gridpoint=-1,
    nthreads=1,
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
    leakage_file : str, optional
        Leakage information file
    iau_order : bool
        PB Jones in IAU order or not
    pb_jones_file : str
        Numpy file with primary beam jones matrices
    save_pb : bool
        Save primary beam jones matrices for future use
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
    nthreads = max(1, abs(nthreads))
    if MWA_PB_file == "" or os.path.exists(MWA_PB_file) is False:
        image_header = fits.getheader(imagename)
        if image_header["CTYPE3"] == "FREQ":
            freqres = float(image_header["CDELT3"]) / 1000.0
        elif image_header["CTYPE4"] == "FREQ":
            freqres = float(image_header["CDELT4"]) / 1000.0
        else:
            freqres = 160.0
        beam_files = np.array(
            glob.glob(f"{datadir}/mwa_full_embedded_element_pattern*.h5")
        )
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
        beam_files = beam_files[beam_files_freqs > freqres]
        beam_files_freqs = beam_files_freqs[beam_files_freqs > freqres]
        pos = np.argmin(np.abs(beam_files_freqs - freqres))
        MWA_PB_file = beam_files[pos]
    print(f"Primary beam file: {MWA_PB_file}")
    if sweet_spot_file == "" or os.path.exists(sweet_spot_file) is False:
        sweet_spot_file = sweet_spot_file_paircars
    try:
        if not restore:
            print(
                f"Correcting image : {os.path.basename(imagename)} for MWA primary beam response.\n"
            )
        else:
            print(
                f"Undo the correction image : {os.path.basename(imagename)} for MWA primary beam response.\n"
            )

        ###################################
        # Determining beamformer delays
        ###################################
        if metafits == "" or os.path.isfile(metafits) is False:
            if gridpoint == -1:
                print("Either provide correct metafits file or grid point number.")
                return
            else:
                sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
                gridpoint = sweet_spots[int(gridpoint)][-1]
        else:
            metadata = fits.getheader(metafits)
            gridpoint = metadata["GRIDNUM"]
            sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
            sweet_spots[int(gridpoint)][-1]

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
            freq = imageheader["CRVAL3"] / 10**6  # In MHz
        elif imageheader["CTYPE4"] == "FREQ":
            freq = imageheader["CRVAL4"] / 10**6
        else:
            print(f"No frequency axis in image: {imagename}.")
            return

        ###############################
        # Determining stokes
        ##############################
        if imageheader["CTYPE3"] == "STOKES":
            stokesaxis = 3
            stokes = "IQUV"
        elif imageheader["CTYPE4"] == "STOKES":
            stokesaxis = 4
            stokes = "IQUV"
        else:
            stokesaxis = 1
            stokes = "I"
        if imagedata.shape[0] == 1 and imagedata.shape[1] == 1:
            stokesaxis = 1
            stokes = "I"

        ####################################
        # Preparing data and data grid
        ####################################
        if stokes == "I":
            imagedata = np.repeat(imagedata, 4, 0)
        alt_az_array = get_azza_from_fits(imagename, metafits)

        if os.path.exists(pb_jones_file) is False:
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
            if save_pb:
                if pb_jones_file == "":
                    pb_jones_file = "mwapb.npy"
                else:
                    pb_jones_file = pb_jones_file.rstrip(".npy") + ".npy"
                print(f"Saving primary beam Jones matrices in: {pb_jones_file}\n")
                np.save(
                    pb_jones_file, np.array([iau_order, jones_array], dtype="object")
                )
        elif os.path.exists(pb_jones_file):
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

        if restore is False:
            stokes = get_IQUV(imagename, stokesaxis=stokesaxis)
            Vij = get_inst_pols(stokes, iau_order=not iau_order)
            Vij_reshaped = Vij.reshape(Vij.shape[0] * Vij.shape[1], 2, 2)
            jones_array_H = np.transpose(jones_array.conj(), axes=((0, 2, 1)))
            Vij_corrected = np.matmul(
                inv(jones_array), np.matmul(Vij_reshaped, inv(jones_array_H))
            )
            Vij_reshaped = Vij_corrected.reshape(Vij.shape)
            B = B2IQUV(Vij_reshaped, iau_order=iau_order)
        else:
            stokes = get_IQUV(imagename, stokesaxis=stokesaxis)
            Vij = get_inst_pols(stokes, iau_order=iau_order)
            Vij_reshaped = Vij.reshape(Vij.shape[0] * Vij.shape[1], 2, 2)
            jones_array_H = np.transpose(jones_array.conj(), axes=((0, 2, 1)))
            Vij_corrected = np.matmul(
                jones_array, np.matmul(Vij_reshaped, jones_array_H)
            )
            Vij_reshaped = Vij_corrected.reshape(Vij.shape)
            B = B2IQUV(Vij_reshaped, iau_order=not iau_order)

        if stokesaxis == 3:
            imagedata[0, 0, :, :] = np.real(B["I"])
            imagedata[0, 1, :, :] = np.real(B["Q"])
            imagedata[0, 2, :, :] = np.real(B["U"])
            imagedata[0, 3, :, :] = np.real(B["V"])
            fullpol = True
        elif stokesaxis == 4:
            imagedata[0, 0, :, :] = np.real(B["I"])
            imagedata[1, 0, :, :] = np.real(B["Q"])
            imagedata[2, 0, :, :] = np.real(B["U"])
            imagedata[3, 0, :, :] = np.real(B["V"])
            fullpol = True
        else:
            imagedata[0, 0, :, :] = np.real(B["I"])
            fullpol = False

        if os.path.exists(outfile):
            os.system(f"rm -rf {outfile}")
        imageheader["PBFILE"] = os.path.basename(MWA_PB_file)
        imageheader["HYPEBEAM"] = "0.10.4"
        fits.writeto(outfile, data=imagedata, header=imageheader, overwrite=True)
        if fullpol:
            if leakage_file != "" and os.path.exists(leakage_file):
                try:
                    freq, leakage_info = np.load(leakage_file, allow_pickle=True)
                    max_iter = max(leakage_info.keys())
                    (
                        q_leakage,
                        u_leakage,
                        v_leakage,
                        res_q_leakage,
                        res_u_leakage,
                        res_v_leakage,
                    ) = leakage_info[max_iter]
                    ########################################################################################
                    # Image based leakage correction if polarisation selfcal solutions could not be applied, but leakage information available
                    ########################################################################################
                    header = fits.getheader(outfile)
                    if "POLSELF" in header.keys():
                        if header["POLSELF"] == "FALSE":
                            leakagecor_image, _ = correct_leakage(
                                outfile,
                                modelname="",
                                q_leakage=q_leakage,
                                u_leakage=u_leakage,
                                v_leakage=v_leakage,
                            )
                            if os.path.exists(leakagecor_image):
                                os.system(f"rm -rf {outfile}")
                                os.system(f"mv {leakagecor_image} {outfile}")
                                with fits.open(outfile, mode="update") as hdul:
                                    hdr = hdul[0].header
                                    hdr["IMGLEAKCOR"] = "TRUE"
                    #####################################################
                    # Writing residual leakage information
                    ######################################################
                    with fits.open(outfile, mode="update") as hdul:
                        hdr = hdul[0].header
                        hdr["LEAKUNIT"] = "PERCENT"
                        if np.isnan(res_q_leakage):
                            hdr["QLEAK"] = "NAN"
                        else:
                            hdr["QLEAK"] = abs(round(res_q_leakage * 100.0, 4))
                        if np.isnan(res_u_leakage):
                            hdr["ULEAK"] = "NAN"
                        else:
                            hdr["ULEAK"] = abs(round(res_u_leakage * 100.0, 4))
                        if np.isnan(res_v_leakage):
                            hdr["VLEAK"] = "NAN"
                        else:
                            hdr["VLEAK"] = abs(round(res_v_leakage * 100.0, 4))
                except Exception:
                    traceback.print_exc()
        print(f"Output image written to : {outfile}\n")
        return outfile
    except Exception:
        traceback.print_exc()
        return


def cli():
    parser = argparse.ArgumentParser(
        description="Correct images for MWA primary beam response"
    )
    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "imagename",
        type=str,
        help="Name of the image file",
    )
    basic_args.add_argument(
        "metafits",
        type=str,
        help="Name of the metafits file",
    )
    basic_args.add_argument(
        "outfile",
        type=str,
        help="Output file name",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--MWA_PB_file",
        type=str,
        default="",
        help="MWA primary beam file",
    )
    adv_args.add_argument(
        "--sweetspot_file",
        type=str,
        default="",
        help="MWA primary beam sweetspot file path",
    )
    adv_args.add_argument(
        "--leakage_file",
        type=str,
        default="",
        help="Leakage information file",
    )
    adv_args.add_argument(
        "--iau_order",
        action="store_true",
        help="PB Jones in IAU order",
    )
    adv_args.add_argument(
        "--gridpoint",
        type=int,
        default=-1,
        help="MWA sweet spot pointing number",
    )
    adv_args.add_argument(
        "--restore",
        action="store_true",
        help="Restore the primary beam correction",
    )
    adv_args.add_argument(
        "--pb_jones_file",
        default="mwapb.npy",
        help="NumPy file of PB Jones matrices",
    )
    adv_args.add_argument(
        "--save_pb",
        action="store_true",
        help="Save PB Jones matrices",
    )
    adv_args.add_argument(
        "--interpolated",
        action="store_true",
        help="Use interpolated beam model",
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of CPU threads to use",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    time.time()

    try:
        get_pbcor_image(
            args.imagename,
            args.outfile,
            args.metafits,
            MWA_PB_file=args.MWA_PB_file,
            sweet_spot_file=args.sweetspot_file,
            leakage_file=args.leakage_file,
            iau_order=args.iau_order,
            pb_jones_file=args.pb_jones_file,
            save_pb=args.save_pb,
            interpolated=args.interpolated,
            gridpoint=args.gridpoint,
            nthreads=args.num_threads,
            restore=args.restore,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1
