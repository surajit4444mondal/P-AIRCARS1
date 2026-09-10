import numpy as np
import os
import copy
import time
import socket
import stat
import tempfile
import sys
import astropy.units as u
from io import StringIO
from astropy.time import Time
from astropy.coordinates import Angle
from contextlib import contextmanager


##########################
# Basic utility funactions
##########################
@contextmanager
def capture_all_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    try:
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def print_banner(msg, pad=0, max_width=50, no_print=False):
    width = min(len(msg) + pad, max_width)
    line = "#" * width
    if not no_print:
        print(line)
        print(f"{msg.center(width)}")
        print(line)
    else:
        msg = f"{line}\n{msg.center(width)}\n{line}"
    return msg


def internet_available(timeout=10):
    """
    Check internet connection
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def check_permission(path):
    """
    check read, write, execute permission of a file or directory

    Parameters
    ----------
    path : str
        File or directory path

    Returns
    -------
    bool
        Whether hand permission or not
    """
    if path is None or path == "" or not os.path.exists(path):
        return False
    print(f"Checking permission for: {path}")
    ###########################
    # READ TEST
    ###########################
    try:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                f.read(1)
        elif os.path.isdir(path):
            os.listdir(path)
        else:
            return False
    except Exception:
        return False
    ###########################
    # WRITE TEST
    ###########################
    try:
        if os.path.isfile(path):
            with open(path, "a"):
                pass
        elif os.path.isdir(path):
            with tempfile.NamedTemporaryFile(dir=path, delete=True):
                pass
    except Exception:
        return False
    ###########################
    # EXECUTE TEST
    ###########################
    try:
        if os.path.isdir(path):
            if not os.access(path, os.X_OK):
                return False
        elif os.path.isfile(path):
            mode = os.stat(path).st_mode
            if not (mode & stat.S_IXUSR):
                return False
    except Exception:
        return False
    return True


@contextmanager
def suppress_output():
    """
    Supress CASA terminal output
    """
    with open(os.devnull, "w") as fnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


def get_cachedir():
    """
    Get cache directory
    """
    homedir = os.path.expanduser("~")
    cachedir = f"{homedir}/.paircarspipe"
    os.makedirs(cachedir, exist_ok=True)
    return cachedir


def create_datadir(datadir=""):
    """
    Create data directory

    Parameters
    ----------
    datadir : str, optional
        User provided custom data directory
    """
    cachedir = get_cachedir()
    if datadir == "":
        datadir = f"{cachedir}/paircarspipe_data"
    else:
        datadir = f"{datadir}/paircarspipe_data"
    os.makedirs(datadir, exist_ok=True)
    os.chmod(datadir, 0o755)
    with open(f"{cachedir}/paircarspipe_data_dir.txt", "w") as f:
        f.write(str(datadir) + "\n")
    return


def get_datadir():
    """
    Get package data directory

    Returns
    -------
    str
        Data directory
    """
    cachedir = get_cachedir()
    if not os.path.exists(f"{cachedir}/paircarspipe_data_dir.txt"):
        return None
    with open(f"{cachedir}/paircarspipe_data_dir.txt", "r") as f:
        datadir = f.read().strip()
    os.makedirs(datadir, exist_ok=True)
    return datadir


def wait_for_port(host, port, timeout=60):
    """
    Waiting for port

    Parameters
    ----------
    port : int
        Port number
    timeout : int, optional
        Timeout in seconds
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def get_free_port(start_port=4200, end_port=4300):
    """
    Get free port

    Parameters
    ----------
    start_port : int, optional
        Start port range
    end_port : int, optional
        End port range

    Returns
    -------
    int
        Free port
    """
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue


def check_port_status(port):
    """
    Check port status

    Parameters
    ----------
    port : int
        Port number

    Returns
    -------
    int
        Port free status
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def split_into_chunks(lst, target_chunk_size):
    """
    Split a list into equal number of elements

    Parameters
    ----------
    lst : list
        List of numbers
    target_chunk_size: int
        Number of elements per chunk

    Returns
    -------
    list
        Chunked list
    """
    n = len(lst)
    num_chunks = max(1, round(n / target_chunk_size))
    avg_chunk_size = n // num_chunks
    remainder = n % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0  # Distribute remainder
        end = start + avg_chunk_size + extra
        chunks.append(lst[start:end])
        start = end
    return chunks


def interpolate_nans(data):
    """Linearly interpolate NaNs in 1D array."""
    from scipy.interpolate import interp1d

    nans = np.isnan(data)
    if np.sum(~nans) < 5:
        return data
    x = np.arange(len(data))
    interp_func = interp1d(
        x[~nans],
        data[~nans],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp_func(x)


def average_timestamp(timestamps):
    """
    Compute the average timestamp using astropy from a list of ISO 8601 strings.

    Parameters
    ----------
    timestamps : list
        timestamps (list of str): List of timestamp strings in 'YYYY-MM-DDTHH:MM:SS' format.

    Returns
    --------
    str
        Average timestamp in 'YYYY-MM-DDTHH:MM:SS' format.
    """
    if len(timestamps) == 0:
        return ""
    times = Time(timestamps, format="isot", scale="utc")
    avg_time = Time(np.mean(times.jd), format="jd", scale="utc")
    return avg_time.isot.split(".")[0]  # Strip milliseconds for clean output


def ceil_to_multiple(n, base):
    """
    Round up to the next multiple

    Parameters
    ----------
    n : float
        The number
    base : float
        Whose multiple will be

    Returns
    -------
    float
        The modified number
    """
    return ((n // base) + 1) * base


def average_with_padding(array, chanwidth, axis=0, pad_value=np.nan):
    """
    Averages an array along a specified axis with a given chunk width (chanwidth),
    padding the array if its size along that axis is not divisible by chanwidth.

    Parameters
    ----------
    array : ndarray
        Input array to average.
    chanwidth : int
        Width of chunks to average.
    axis : int
        Axis along which to perform the averaging.
    pad_value : float
        Value to pad with if padding is needed (default: np.nan).

    Returns
    --------
    ndarray
        Array averaged along the specified axis.
    """
    # Compute the shape along the specified axis
    original_size = array.shape[axis]
    pad_size = -original_size % chanwidth
    # If padding is needed, apply it directly along the target axis
    if pad_size > 0:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, pad_size)
        array = np.pad(array, pad_width, constant_values=pad_value)
    # Compute the new shape and reshape the array for chunking
    new_shape = list(array.shape)
    new_shape[axis] = array.shape[axis] // chanwidth
    new_shape.insert(axis + 1, chanwidth)
    reshaped_array = array.reshape(new_shape)
    # Use nanmean along the chunk axis for averaging
    averaged_array = np.nanmean(reshaped_array, axis=axis + 1)
    return averaged_array


def filter_outliers(data, threshold=5, max_iter=3):
    """
    Filter outliers and perform cubic spline fitting

    Parameters
    ----------
    y : numpy.array
        Y values
    threshold : float
        Threshold of filtering
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    numpy.array
        Clean Y-values
    """
    from scipy.ndimage import gaussian_filter1d

    for c in range(max_iter):
        data = np.asarray(data, dtype=np.float64)
        original_nan_mask = np.isnan(data)
        # Interpolate NaNs for smoothing
        interpolated_data = interpolate_nans(data)
        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(
            interpolated_data, sigma=threshold, truncate=3 * threshold
        )
        # Compute residuals and std only on valid original data
        residuals = data - smoothed
        valid_mask = ~original_nan_mask
        std_dev = np.std(residuals[valid_mask])
        if std_dev == 0.0:
            return data
        # Detect outliers
        outlier_mask = np.abs(residuals) > threshold * std_dev
        combined_mask = valid_mask & ~outlier_mask
        # Replace outliers with NaN
        filtered_data = np.where(combined_mask, data, np.nan)
        data = copy.deepcopy(filtered_data)
    return filtered_data


def weighted_mean(x, xerr):
    """
    Calculate weighted mean

    Parameters
    ----------
    x : numpy.array
        Data array
    xerr : numpy.array
        Error array

    Returns
    -------
    float
        Weighted mean
    float
        Weighted error
    """
    w = 1.0 / xerr**2
    mean = np.nansum(w * x) / np.nansum(w)
    err = np.sqrt(1.0 / np.nansum(w))
    return mean, err


def angular_separation_equatorial(ra1, dec1, ra2, dec2):
    """
    Calculate angular seperation between two equatorial coordinates

    Parameters
    ----------
    ra1 : float
        RA of the first coordinate in degree
    dec1 : float
        DEC of the first coordinate in degree
    ra2 : float
        RA of the second coordinate in degree
    dec2 : float
        DEC of the second coordinate in degree

    Returns
    -------
    float
        Angular distance in degree
    """
    # Convert RA and Dec from degrees to radians
    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)
    # Apply the spherical distance formula using NumPy functions
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        ra1 - ra2
    )
    # Calculate the angular separation in radians
    theta_rad = np.arccos(cos_theta)
    # Convert the angular separation from radians to degrees
    theta_deg = np.degrees(theta_rad)
    return round(theta_deg, 2)


def ra_dec_to_deg(ra_hms, dec_dms):
    """
    Convert RA and Dec from hms and dms format to degrees

    Parameters
    ----------
    ra_hms: str
        Right Ascension in 'hms' format
    dec_dms : str
        Declination in 'dms' format

    Returns
    -------
    tuple
        RA and Dec in degrees
    """
    ra = Angle(ra_hms, unit=u.hourangle)
    dec = Angle(dec_dms, unit=u.deg)
    return round(ra.deg, 4), round(dec.deg, 4)


def ra_dec_to_hms_dms(ra_deg, dec_deg):
    """
    Convert RA and Dec in degrees to hms and dms format

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees.
    dec_deg : float
        Declination in degrees.

    Returns
    -------
    tuple
        RA in h:m:s format, Dec in d:m:s format (e.g., '1h5m0s', '1d5m0s').
    """
    # Convert RA to h:m:s
    if ra_deg < 0:
        ra_deg += 360
    ra = Angle(ra_deg, unit=u.deg)
    ra_hms = ra.to_string(unit=u.hourangle, sep=":").split(":")
    ra_hms = ra_hms[0] + "h" + ra_hms[1] + "m" + ra_hms[2] + "s"
    # Convert Dec to d:m:s
    dec = Angle(dec_deg, unit=u.deg)
    dec_dms = dec.to_string(unit=u.deg, sep=":").split(":")
    dec_dms = dec_dms[0] + "d" + dec_dms[1] + "m" + dec_dms[2] + "s"
    return ra_hms, dec_dms


def timestamp_to_mjdsec(timestamp, date_format=0):
    """
    Convert timestamp to mjd second.


    Parameters
    ----------
    timestamp : str
        Time stamp to convert
    date_format : int, optional
        Datetime string format
            0: 'YYYY/MM/DD/hh:mm:ss'

            1: 'YYYY-MM-DDThh:mm:ss'

            2: 'YYYY-MM-DD hh:mm:ss'

            3: 'YYYY_MM_DD_hh_mm_ss'

            4: 'YYYYMMDDhhmmss'

    Returns
    -------
    float
        Return correspondong MJD second of the day
    """
    import julian
    from datetime import datetime as dt

    if date_format == 0:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S")
    elif date_format == 1:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    elif date_format == 2:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    elif date_format == 3:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    elif date_format == 4:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y%m%d%H%M%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y%m%d%H%M%S")
    else:
        print("No proper format of timestamp.\n")
        return
    mjd = float(
        "{: .2f}".format(
            (julian.to_jd(timestamp_datetime) - 2400000.5) * (24.0 * 3600.0)
        )
    )
    return mjd


def mjdsec_to_timestamp(mjdsec, str_format=0):
    """
    Convert CASA MJD seceonds to CASA timestamp

    Parameters
    ----------
    mjdsec : float
        CASA MJD seconds
    str_format : int
        Time stamp format (0: yyyy-mm-ddTHH:MM:SS.ff, 1: yyyy/mm/dd/HH:MM:SS.ff, 2: yyyy-mm-dd HH:MM:SS)

    Returns
    -------
    str
        CASA time stamp in UTC at ISOT format
    """
    from casatools import measures, quanta

    me = measures()
    qa = quanta()
    today = me.epoch("utc", "today")
    mjd = np.array(mjdsec) / 86400.0
    today["m0"]["value"] = mjd
    hhmmss = qa.time(today["m0"], prec=8)[0]
    date = qa.splitdate(today["m0"])
    qa.done()
    if str_format == 0:
        utcstring = "%s-%02d-%02dT%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    elif str_format == 1:
        utcstring = "%s/%02d/%02d/%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    else:
        utcstring = "%s-%02d-%02d %s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    return utcstring


def get_gpstime_to_date(gpstime):
    """
    Get date and time in YYYYMMDD and hhmmssff format from GPS time

    Parameters
    ----------
    gpstime : int
        GPS time

    Returns
    -------
    str
        Date
    str
        Time
    """
    try:
        t = Time(gpstime, format="gps", scale="utc")
        date = t.to_datetime().strftime("%Y%m%d")
        time = t.to_datetime().strftime("%H%M%S%f")[:8]
        return date, time
    except Exception:
        return None, None
