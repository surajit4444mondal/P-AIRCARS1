import sys
import h5py
import time
import os
import argparse
import traceback
import numpy as np
from scipy.interpolate import CubicSpline
from paircars.utils.logger_utils import SmartDefaultsHelpFormatter

# This script is originally developed by, Puja Majee (NCRA-TIFR), with Devojyoti Kansabanik.


def do_beam_interpolate(original_pb_file, new_freq_res=160, expected_file_size=None):
    """
    Interpolate MWA beam coefficient to obtain beam file at finer spectral resolution

    Parameters
    ----------
    mwapb_file : str
        Original MWA primary beam file
    new_freq_res : int, optional
        Frequency resolution of final beam in kHz
    expected_file_size : int, optional
        Expected file size in bytes (if same named file exist with same size)

    Returns
    -------
    str
        Output file name
    """
    starttime = time.time()
    try:
        time.time()
        new_freq_res = int(new_freq_res)
        new_pb_file = original_pb_file.split(".h5")[0] + f"_{new_freq_res}.h5"

        if os.path.exists(new_pb_file):
            file_size = os.stat(new_pb_file).st_size
            if file_size == expected_file_size:
                return new_pb_file
            os.system("rm -rf " + new_pb_file)

        print(
            f"Interpolating beam coefficients at frequency resolution: {new_freq_res} kHz to primary beam file: {new_pb_file}"
        )
        new_freq_res = new_freq_res * 1000
        f = h5py.File(original_pb_file, "r")
        modes = f["modes"][:]
        old_frequency = np.sort(np.array([int(x[3:]) for x in f.keys() if "X1_" in x]))
        new_frequency = np.arange(old_frequency[0], old_frequency[-1], new_freq_res)

        n_coeffs = []  # Maximum number of coefficients
        for pol in ["X", "Y"]:
            for d in range(1, 17):
                for freq in old_frequency:
                    key = str(pol) + str(d) + "_" + str(freq)
                    n_coeffs.append(f[key].shape[1])
        max_n_coeff = max(n_coeffs)
        coeff_array = np.zeros(
            (2, 16, len(old_frequency), 2, max_n_coeff)
        )  # Old coeff array

        new_coeff_array = np.memmap(
            "new_array.dat",
            dtype=np.float64,
            mode="w+",
            shape=(2, 16, len(new_frequency), 2, max_n_coeff),
        )

        # Making coeff_array
        for pol in range(2):
            for d in range(16):
                for freq in range(len(old_frequency)):
                    key = (
                        str(["X", "Y"][pol])
                        + str(d + 1)
                        + "_"
                        + str(old_frequency[freq])
                    )
                    n_coeff = f[key].shape[1]
                    coeff_array[pol, d, freq, :, :n_coeff] = f[key][:]

        # Interpolation
        for pol in range(2):
            for d in range(16):
                for coeff in range(max_n_coeff):
                    s1 = coeff_array[pol, d, :, 0, coeff]
                    s2 = coeff_array[pol, d, :, 1, coeff]
                    y = s1 * np.exp(1.0j * np.deg2rad(s2))
                    x = old_frequency  # [values!=0]
                    if np.sum(y) != 0.0 + 0.0j:
                        y_real = np.real(y)
                        y_imag = np.imag(y)
                        real_interp = CubicSpline(x, y_real)
                        imag_interp = CubicSpline(x, y_imag)
                        n_freq = np.arange(min(x), max(x), new_freq_res)
                        new_real = real_interp(n_freq)
                        new_imag = imag_interp(n_freq)
                        new_y = new_real + 1.0j * new_imag
                        s1_new = np.abs(new_y)
                        s2_new = np.rad2deg(np.angle(new_y))
                        pos0 = np.argmin(np.abs(new_frequency - min(x)))
                        new_coeff_array[pol, d, pos0 : pos0 + len(new_y), 0, coeff] = (
                            s1_new
                        )
                        new_coeff_array[pol, d, pos0 : pos0 + len(new_y), 1, coeff] = (
                            s2_new
                        )

        # Writing to new coefficients in h5 file
        with h5py.File(new_pb_file, "w") as h5f:
            for pol in range(2):
                for d in range(16):
                    for f in range(len(new_frequency)):
                        freq = new_frequency[f]
                        key = str(["X", "Y"][pol] + str(d + 1) + "_" + str(freq))
                        data = new_coeff_array[pol, d, f, :]
                        pos = np.where(data.mean(0) == 0)
                        data = np.delete(data, pos, axis=-1)
                        h5f.create_dataset(key, data=data, dtype=float)
            h5f.create_dataset("modes", data=modes)
        t_taken = round(time.time() - starttime, 1)
        print(f"Time taken: {t_taken}s")
        os.system("rm -rf new_array.dat")
        return new_pb_file
    except Exception:
        traceback.print_exc()
        return


def cli():
    parser = argparse.ArgumentParser(
        description="Interpolate MWA primary beam coefficients at finer spectral resolution",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mwapb_file",
        type=str,
        help="Original MWA primary beam coefficient file",
    )
    basic_args.add_argument(
        "--freqres",
        type=int,
        default=160,
        help="Frequency resolution of the interpolated file in kHz",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    outfile = do_beam_interpolate(
        args.mwapb_file,
        new_freq_res=int(args.freqres),
    )
    if outfile is not None and os.path.exists(outfile):
        msg = 0
    else:
        msg = 1
    return msg
