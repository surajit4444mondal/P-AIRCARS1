import secrets
import string
import logging
import argparse
import requests
import time
import os
import getpass
import urllib.request
import urllib.error
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from .basic_utils import get_cachedir, suppress_output


##################################
# Logger related functions
##################################
class SmartDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        # Don't show default for boolean flags
        if isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            return action.help
        return super()._get_help_string(action)


def clean_shutdown(observer):
    if observer:
        observer.stop()
        observer.join(timeout=5)


def generate_password(length=6):
    """
    Generate secure 6-character password with letters, digits, and symbols
    """
    chars = string.ascii_letters + string.digits + "@#$&*"
    return "".join(secrets.choice(chars) for _ in range(length))


def get_remote_logger_link():
    cachedir = get_cachedir()
    username = getpass.getuser()
    link_file = os.path.join(cachedir, f".remotelink_{username}.txt")
    try:
        if os.path.isfile(link_file):
            with open(link_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            remote_link = lines[0]
        else:
            return ""
    except Exception:
        return ""
    try:
        req = urllib.request.Request(remote_link, method="GET")
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status == 200:
                return remote_link
    except (urllib.error.URLError, urllib.error.HTTPError):
        return ""


def get_remote_logger_password():
    cachedir = get_cachedir()
    username = getpass.getuser()
    link_file = os.path.join(cachedir, f".remotelink_password_{username}.txt")
    try:
        if os.path.isfile(link_file):
            with open(link_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            remote_password = lines[0]
            return remote_password
        else:
            return ""
    except Exception:
        return ""


def get_emails():
    cachedir = get_cachedir()
    username = getpass.getuser()
    email_file = os.path.join(cachedir, f".emails_{username}.txt")
    try:
        with open(email_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ""
    if not lines:
        return ""
    else:
        return lines[0]


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self._buffer = ""

    def write(self, message):
        # Remove trailing newlines and skip empty messages
        message = message.rstrip()
        if message:
            self.logger.log(self.log_level, message)

    def flush(self):
        pass  # Required for compatibility


class RemoteLogger(logging.Handler):
    """
    Remote logging handler for posting log messages to a web endpoint.
    """

    def __init__(
        self,
        job_id="default",
        log_id="run_default",
        log_type="master",
        remote_link="",
        password="",
    ):
        super().__init__()
        self.job_id = job_id
        self.log_id = log_id
        self.log_type = log_type
        self.password = password
        self.remote_link = remote_link

    def emit(self, record):
        msg = str(self.format(record))
        level = msg.split("|")[0].strip()
        msg = "|".join(msg.split("|")[1:])
        # Fix numeric levels
        if isinstance(level, int) or level.isdigit():
            level = logging.getLevelName(int(level))
            msg = f"{level} | {msg}"
        try:
            if msg != "" and msg != " " and msg != "\n":
                requests.post(
                    f"{self.remote_link}/api/log",
                    json={
                        "job_id": self.job_id,
                        "log_id": self.log_id,
                        "log_type": self.log_type,
                        "message": f"{msg}\n",
                        "password": self.password,
                        "first": False,
                    },
                    timeout=2,
                )
        except Exception:
            pass  # Fail silently to avoid interrupting the main app


class LogTailHandler(FileSystemEventHandler):
    """
    Continuous logging
    """

    def __init__(self, logfile, logger):
        self.logfile = logfile
        self.logger = logger
        self._position = os.path.getsize(logfile) if os.path.exists(logfile) else 0

    def on_modified(self, event):
        if event.src_path == self.logfile:
            try:
                with open(self.logfile, "r") as f:
                    f.seek(self._position)
                    lines = f.readlines()
                    self._position = f.tell()
                for line in lines:
                    if line != "" and line != " " and line != "\n":
                        self.logger.info(line.strip())
            except Exception:
                pass


def create_logger(logname, logfile):
    """
    Create logger.

    Parameters
    ----------
    logname : str
        Name of the log
    logfile : str, optional
        Log file name

    Returns
    -------
    logger
        Python logging object
    str
        Log file name
    """
    logger = logging.getLogger(logname)
    # If logger already configured, return it
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    filehandle = logging.FileHandler(logfile)
    filehandle.setFormatter(formatter)
    logger.addHandler(filehandle)
    logger.propagate = False
    logger.info(f"Log file: {logfile}\n")
    return logger, logfile


def get_logid(logfile):
    """
    Get log id for remote logger from logfile name
    """
    name = os.path.basename(logfile)
    logmap = {
        "apply_basiccal_target": "Applying basic calibration solutions on targets",
        "apply_basiccal_selfcal": "Applying basic calibration solutions for self-calibration",
        "apply_pbcor": "Applying primary beam corrections",
        "apply_selfcal": "Applying self-calibration solutions",
        "basic_cal": "Basic calibration",
        "cor_phasecenter_target": "Moving phasecenter to solar center",
        "cor_sidereal_selfcal": "Correction of sidereal motion before self-calibration",
        "cor_sidereal_target": "Correction of sidereal motion for target scans",
        "flagging_cal": "Basic flagging of calibrators",
        "flagging_target": "Basic flagging of targets",
        "flagging_selfcal": "Basic flagging of target before self-calibration",
        "modeling": "Simulating visibilities of calibrators",
        "split_calibrator": "Spliting calibrator scans",
        "split_target": "Spliting target",
        "split_selfcal": "Spliting for self-calibration",
        "selfcal_target": "All self-calibrations",
        "imaging_target": "All imaging",
        "ds_target": "Making dynamic spectra",
        "do_overlay": "Making overlays",
        "master": "Master flow",
        "do_msplot": "Diagnistic plot of ms",
    }
    logmap_keys = list(logmap.keys())
    for logmap_key in logmap_keys:
        if name.startswith(logmap_key):
            log_name = logmap[logmap_key]
            if name.startswith("flagging"):
                try:
                    datacolumn = str(name.split(".log")[0].split("_")[2])
                    log_name = f"{log_name}, datacolumn: {datacolumn}"
                except Exception:
                    pass
            try:
                obsid = int(name.split(".log")[0].split("_")[-1])
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{int(time.time())}_time_{log_name}"
    if name.startswith("subflow"):
        if name.startswith("subflow_preprocess"):
            log_name = "Pre-processing"
            try:
                obsid = name.split(".log")[0].split("subflow_preprocess_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{int(time.time())}_time_{log_name}"
        elif name.startswith("subflow_basiccal"):
            log_name = "Basic calibration"
            try:
                obsid = name.split(".log")[0].split("subflow_basiccal_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{int(time.time())}_time_{log_name}"
        elif name.startswith("subflow_selfcal"):
            log_name = "Self-calibration"
            try:
                obsid = name.split(".log")[0].split("subflow_selfcal_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{int(time.time())}_time_{log_name}"
        elif name.startswith("subflow_applysol"):
            log_name = "Apply calibration solutions"
            try:
                obsid = name.split(".log")[0].split("subflow_applysol_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{int(time.time())}_time_{log_name}"
        elif name.startswith("subflow_imaging"):
            log_name = "Imaging"
            try:
                obsid = name.split(".log")[0].split("subflow_imaging_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{int(time.time())}_time_{log_name}"
        else:
            log_name = f"Subflow: {name}"
            return f"{int(time.time())}_time_{log_name}"
    elif name.endswith("_int.log"):
        name = name.split("_int.log")[0].split("selfcal_")[1]
        obsid = name.split("_")[0]
        coarse_chan = name.split("_ch_")[-1]
        return f"{int(time.time())}_time_Intensity self-calibration, OBSID: {obsid}, coarse channel: {coarse_chan}"
    elif name.endswith("_pol.log"):
        name = name.split("_pol.log")[0].split("selfcal_")[1]
        obsid = name.split("_")[0]
        coarse_chan = name.split("_ch_")[-1]
        return f"{int(time.time())}_time_Polarisation self-calibration, OBSID: {obsid}, coarse channel: {coarse_chan}"
    elif "imaging" in name:
        name = name.rstrip(".log").split("imaging_")[1]
        obsid = name.split("_")[0]
        coarse_chan = name.split("_ch_")[-1]
        return f"{int(time.time())}_time_Imaging, OBSID: {obsid}, coarse channel: {coarse_chan}"
    else:
        return f"{int(time.time())}_time_{name}"


def get_logger_safe():
    """
    Returns a logger that
    - Uses Prefect logger if inside a task/flow
    - Falls back to simple print-style logger otherwise
    """
    try:
        with suppress_output():
            from prefect import get_run_logger

            logger = get_run_logger()
            logger.setLevel(logging.INFO)
            return logger
    except Exception:
        name = f"task_{os.getpid()}"
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        return logger


def init_logger(logname, logfile, log_type="task", jobname="", password=""):
    """
    Initialize a remote logger with watchdog-based tailing.

    Parameters
    ----------
    logname : str
        Logger name.
    logfile : str
        Path to the local logfile to also monitor.
    logtype : str, optional
        Log type (only allowed values: master | subflow | task)
    jobname : str, optional
        Remote logger job ID.
    password : str
        Password used for remote authentication.

    Returns
    -------
    observer
        Observer object
    """
    try:
        timeout = 30
        waited = 0
        while True:
            if not os.path.exists(logfile):
                time.sleep(1)
                waited += 1
            elif waited >= timeout:
                return
            else:
                break
        logger = logging.getLogger(logname)
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter("%(message)s")
        remote_link = get_remote_logger_link()
        if log_type not in ["master", "subflow", "task"]:
            log_type = "task"
        if remote_link != "":
            if jobname:
                job_id = jobname
                log_id = get_logid(logfile)
                remote_handler = RemoteLogger(
                    job_id=job_id,
                    log_id=log_id,
                    log_type=log_type,
                    remote_link=remote_link,
                    password=password,
                )
                logger.setLevel(logging.DEBUG)
                remote_handler.setFormatter(formatter)
                logger.addHandler(remote_handler)
                try:
                    requests.post(
                        f"{remote_link}/api/log",
                        json={
                            "job_id": job_id,
                            "log_id": log_id,
                            "log_type": log_type,
                            "message": "",
                            "password": password,
                            "first": True,
                        },
                        timeout=2,
                    )
                except Exception:
                    pass
            if os.path.exists(logfile):
                if os.path.islink(logfile):
                    logfile = os.readlink(logfile)
                event_handler = LogTailHandler(logfile, logger)
                observer = Observer()
                observer.schedule(
                    event_handler, path=os.path.dirname(logfile), recursive=False
                )
                observer.start()
                return observer
            else:
                return
        else:
            return
    except Exception:
        return
