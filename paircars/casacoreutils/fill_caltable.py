import sys
import numpy as np
import os
import argparse
import traceback
from casacore.tables import table, maketabdesc, makecoldesc, makearrcoldesc


def make_caltable_columns(msname, caltable, nchan):
    """
    Make blank caltable columns
    Parameters
    ----------
    ms_file : str
        Path to the measurement set (MS).
    output_caltable : str
        Name of the output calibration table.
    nchan :int
        Number of channels to include in the table
    Returns
    -------
    str
        Caltable name
    """
    with table(msname + "/ANTENNA", readonly=True) as antenna_table:
        num_ants = len(antenna_table)  # Number of antennas
    with table(msname + "/SPECTRAL_WINDOW", readonly=True) as spw_table:
        num_spws = len(spw_table)  # Number of SPWs
    coldes = {
        "comment": "",
        "dataManagerGroup": "MSMTAB",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "maxlen": 0,
        "option": 5,
        "valueType": "int",
    }
    coldes_time = {
        "comment": "",
        "dataManagerGroup": "MSMTAB",
        "dataManagerType": "StandardStMan",
        "keywords": {
            "MEASINFO": {"Ref": "UTC", "type": "epoch"},
            "QuantumUnits": np.array(["s"], dtype="|S2"),
        },
        "maxlen": 0,
        "option": 5,
        "valueType": "double",
    }
    coldes_interval = {
        "comment": "",
        "dataManagerGroup": "MSMTAB",
        "dataManagerType": "StandardStMan",
        "keywords": {"QuantumUnits": np.array(["s"], dtype="|S2")},
        "maxlen": 0,
        "option": 5,
        "valueType": "double",
    }
    # Define the table description for the calibration table
    flag = makearrcoldesc(
        "FLAG", np.zeros((nchan, 2), dtype=bool)
    )  # Boolean array column
    table_desc = maketabdesc(
        [
            makecoldesc("ANTENNA1", coldes),  # Integer column for antenna ID
            makecoldesc("ANTENNA2", coldes),
            makecoldesc("SPECTRAL_WINDOW_ID", coldes),  # Integer column for SPW ID
            makecoldesc("FIELD_ID", coldes),
            makecoldesc("SCAN_NUMBER", coldes),
            makecoldesc("TIME", coldes_time),  # Double column for time
            makecoldesc("INTERVAL", coldes_interval),  # Double column for interval
            makearrcoldesc("CPARAM", 1j, 0, [nchan, 2]),
            makearrcoldesc("FLAG", False, 0, [nchan, 2]),
            makearrcoldesc("PARAMERR", 1.0, 0, [nchan, 2]),
            makearrcoldesc("SNR", 1.0, 0, [nchan, 2]),
            makearrcoldesc("WEIGHT", 1.0, 0, [nchan, 2]),
        ]
    )
    # Open the table
    cal_table = table(caltable, readonly=False)
    # Add rows and populate them for each SPW and antenna combination
    for spw in range(num_spws):
        for ant in range(num_ants):
            cal_table.addrows(1)
            row_idx = len(cal_table) - 1
            # Fill scalar columns
            cal_table.putcell("ANTENNA1", row_idx, ant)
            cal_table.putcell("ANTENNA2", row_idx, 1)
            cal_table.putcell("SCAN_NUMBER", row_idx, 1)
            cal_table.putcell("SPECTRAL_WINDOW_ID", row_idx, spw)
            cal_table.putcell("TIME", row_idx, 0.0)  # Placeholder time
            cal_table.putcell("INTERVAL", row_idx, 0.0)  # Placeholder interval
            # Fill array columns
            bandpass_data = np.ones(
                (nchan, 2), dtype=np.complex64
            )  # Normalized bandpass
            flag_data = np.zeros((nchan, 2), dtype=bool)  # No flags
            snr = np.ones((nchan, 2), dtype=np.float64) + 1  # SNR
            err = np.zeros((nchan, 2), dtype=np.float64)  # Error
            weight = np.ones((nchan, 2), dtype=np.float64)
            cal_table.putcell("CPARAM", row_idx, bandpass_data)
            cal_table.putcell("PARAMERR", row_idx, err)
            cal_table.putcell("FLAG", row_idx, flag_data)
            cal_table.putcell("SNR", row_idx, snr)
            cal_table.putcell("WEIGHT", row_idx, weight)
        # Close the table
        cal_table.close()
    return caltable


################################
def cli():
    parser = argparse.ArgumentParser(description="Fill blank caltable columns")
    parser.add_argument(
        "--msname",
        required=True,
        help="Name of the measurement set",
    )
    parser.add_argument(
        "--caltable",
        required=True,
        help="Caltable name",
    )
    parser.add_argument(
        "--nchan",
        type=int,
        default=1,
        help="Number of channels",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()
    try:
        make_caltable_columns(
            args.msname,
            args.caltable,
            args.nchan,
        )
        return 0
    except:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = cli()
    os._exit(result)
