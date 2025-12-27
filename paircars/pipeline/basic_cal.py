from casatasks import bandpass, flagdata
from casatools import table
from calibrate_crossphase import crossphasecal
from optparse import OptionParser
from basic_func import *
import os, gc, traceback, math

os.system("rm -rf casa*log")


def do_flag_cal(
    msname,
    refant,
    caldir,
    uvrange="",
    do_kcross=True,
    kcross_freqavg=-1,
    do_flag=True,
):
    """
    Parameters
    ----------
    msname : str
        Name of the measurement set
    refant : str
        Reference antenna index or name
    caldir : str
        Caltable directory name
    uvrange : str
        UV-range to be used for calibration
    do_kcross : bool
        Perform crosshand phase calibration
    kcross_freqavg : float
        Frequency averaging in crosshand phase estimation in MHz
    do_flag : bool
        Perform flagging or not
    Returns
    -------
    int
        Success or failure message
    str
        Bandpass caltable
    str
        Crosshand phase caltable
    """
    try:
        if do_flag:
            print("Flagging: " + msname)
            flagdata(vis=msname, mode="tfcrop")
        tb = table()
        tb.open(msname + "/SPECTRAL_WINDOW")
        freq = tb.getcol("CHAN_FREQ") / 10**6
        tb.close()
        freqres = freq[1] - freq[0]
        start_coarse_chan = freq_to_MWA_coarse(freq[0])
        end_coarse_chan = freq_to_MWA_coarse(freq[-1])
        caltable_prefix = (
            caldir
            + "/"
            + os.path.basename(msname).split(".ms")[0].split("_")[0]
            + "_ref_"
            + str(refant)
            + "_ch_"
            + str(start_coarse_chan)
            + "_"
            + str(end_coarse_chan)
        )
        print(
            "################\nCalibrating ms :"
            + msname
            + "\n#######################\n"
        )
        if uvrange == "":
            uvrange = get_calibration_uvrange(msname)
        os.system("rm -rf " + caltable_prefix + ".bcal")
        print(
            "bandpass(vis='"
            + msname
            + "',caltable='"
            + caltable_prefix
            + ".bcal',refant='"
            + str(refant)
            + "',solint='inf',uvrange='"
            + uvrange
            + "')\n"
        )
        bandpass(
            vis=msname,
            caltable=caltable_prefix + ".bcal",
            refant=str(refant),
            solint="inf",
            uvrange=uvrange,
        )
        if do_kcross == False:
            gc.collect()
            return 0, caltable_prefix + ".bcal", None
        else:
            if kcross_freqavg > freqres and kcross_freqavg > 0:
                chanwidth = math.ceil(kcross_freqavg / freqres)
            else:
                chanwidth = 1
            print("Estimating crosshand phase...\n")
            crossphase_caltable = f"{caltable_prefix}.kcross"
            bandpass_table = f"{caltable_prefix}.bcal"
            os.system(f"rm -rf {crossphase_caltable}")
            cmd = f"python3 calibrate_crossphase.py --msname {msname} --caltable {crossphase_caltable} --gaintable {bandpass_table} --chanwidth {chanwidth} --uvrange {uvrange}"
            print(cmd)
            os.system(cmd)
            return 0, bandpass_table, crossphase_caltable
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return 1, None, None


################################
def main():
    usage = "Flag and calibrate"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--refant",
        dest="refant",
        default="1",
        help="Reference antenna",
        metavar="String",
    )
    parser.add_option(
        "--do_kcross",
        dest="do_kcross",
        default=True,
        help="Perform crosshand phase calibration or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--do_flag",
        dest="do_flag",
        default=True,
        help="Perform basic flagging on the data or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--uvrange",
        dest="uvrange",
        default="",
        help="UV-range for calibration",
        metavar="String",
    )
    parser.add_option(
        "--kcross_freqavg",
        dest="kcross_freqavg",
        default=-1,
        help="Crosshand phase frequency averaging in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--caldir",
        dest="caldir",
        default=None,
        help="Caltable directory",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.msname == None:
        print("Please provide the measurement set name.\n")
        return 1
    if options.caldir == None:
        caldir = os.path.dirname(options.msname) + "/caltables"
    else:
        caldir = options.caldir
    if os.path.exists(caldir) == False:
        os.makedirs(caldir)
    msg, bcal, kcrosscal = do_flag_cal(
        options.msname,
        options.refant,
        caldir,
        uvrange=str(options.uvrange),
        do_kcross=eval(str(options.do_kcross)),
        kcross_freqavg=float(options.kcross_freqavg),
        do_flag=eval(str(options.do_flag)),
    )
    if msg == 0:
        if kcrosscal != None:
            print("Caltable names: " + bcal + "," + kcrosscal + "\n")
        else:
            print("Caltable names: " + bcal + "\n")
    else:
        print("Issues occured")
    gc.collect()
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
