import ctypes
import platform
import os
import sys
import argparse
import numpy as np
import logging
import traceback
import glob
import getpass
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTabWidget,
    QSizeGrip,
    QHBoxLayout,
    QGridLayout,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

LOG_DIR = None
POSIX_FADV_DONTNEED = 4
libc = ctypes.CDLL("libc.so.6")

username = getpass.getuser()


#####################################
# Resource management
#####################################
def drop_file_cache(filepath, verbose=False):
    """
    Advise the OS to drop the given file from the page cache.
    Safe, per-file, no sudo required.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    try:
        if not os.path.isfile(filepath):
            return
        fd = os.open(filepath, os.O_RDONLY)
        result = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        os.close(fd)
        if verbose:
            if result == 0:
                print(f"[cache drop] Released: {filepath}")
            else:
                print(f"[cache drop] Failed ({result}) for: {filepath}")
    except Exception as e:
        if verbose:
            print(f"[cache drop] Error for {filepath}: {e}")
            traceback.print_exc()


def drop_cache(path, verbose=False):
    """
    Drop file cache for a file or all files under a directory.

    Parameters
    ----------
    path : str
        File or directory path
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    if os.path.isfile(path):
        drop_file_cache(path, verbose=verbose)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                full_path = os.path.join(root, f)
                drop_file_cache(full_path, verbose=verbose)
    else:
        if verbose:
            print(f"[cache drop] Path does not exist or is not valid: {path}")


def get_cachedir():
    homedir = os.path.expanduser("~")
    cachedir = f"{homedir}/.paircarspipe"
    os.makedirs(cachedir, exist_ok=True)
    return cachedir


def get_datadir():
    cachedir = get_cachedir()
    if not os.path.exists(f"{cachedir}/paircarspipe_data_dir.txt"):
        return None
    with open(f"{cachedir}/paircarspipe_data_dir.txt", "r") as f:
        datadir = f.read().strip()
    os.makedirs(datadir, exist_ok=True)
    return datadir


class SmartDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        # Don't show default for boolean flags
        if isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            return action.help
        return super()._get_help_string(action)


def classify_log(name):
    if name.startswith("main"):
        return "master"
    elif name.startswith("subflow"):
        return "subflow"
    else:
        return "task"


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
            return f"{log_name}"
    if name.startswith("subflow"):
        if name.startswith("subflow_preprocess"):
            log_name = "Pre-processing"
            try:
                obsid = name.split(".log")[0].split("subflow_preprocess_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{log_name}"
        elif name.startswith("subflow_basiccal"):
            log_name = "Basic calibration"
            try:
                obsid = name.split(".log")[0].split("subflow_basiccal_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{log_name}"
        elif name.startswith("subflow_selfcal"):
            log_name = "Self-calibration"
            try:
                obsid = name.split(".log")[0].split("subflow_selfcal_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{log_name}"
        elif name.startswith("subflow_applysol"):
            log_name = "Apply calibration solutions"
            try:
                obsid = name.split(".log")[0].split("subflow_applysol_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{log_name}"
        elif name.startswith("subflow_imaging"):
            log_name = "Imaging"
            try:
                obsid = name.split(".log")[0].split("subflow_imaging_")[-1]
                log_name = f"{log_name} [{obsid}]"
            except Exception:
                pass
            return f"{log_name}"
        else:
            log_name = f"Subflow: {name}"
            return f"{log_name}"
    elif name.endswith("_int.log"):
        name = name.split("_int.log")[0].split("selfcal_")[1]
        obsid = name.split("_")[0]
        coarse_chan = name.split("_ch_")[-1]
        return (
            f"Intensity self-calibration, OBSID: {obsid}, coarse channel: {coarse_chan}"
        )
    elif name.endswith("_pol.log"):
        name = name.split("_pol.log")[0].split("selfcal_")[1]
        obsid = name.split("_")[0]
        coarse_chan = name.split("_ch_")[-1]
        return f"Polarisation self-calibration, OBSID: {obsid}, coarse channel: {coarse_chan}"
    elif "imaging" in name:
        name = name.rstrip(".log").split("imaging_")[1]
        obsid = name.split("_")[0]
        coarse_chan = name.split("_ch_")[-1]
        return f"Imaging, OBSID: {obsid}, coarse channel: {coarse_chan}"
    else:
        return name


def format_log_block(text):
    """Convert numeric log levels + add spacing + color"""
    formatted = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        color = "#dddddd"
        if len(parts) > 1:
            level = parts[0].strip()
            msg = "|".join(parts[1:]).strip()
            if level.isdigit():
                level = logging.getLevelName(int(level))
            if "CRITICAL" in level:
                color = "#ff55ff"
            elif "ERROR" in level:
                color = "#ff5555"
            elif "WARNING" in level:
                color = "#ffff55"
            elif "DEBUG" in level:
                color = "#8be9fd"
            elif "INFO" in level:
                color = "#55ff55"
            line = f"{level} | {msg}"
        # Color only up to the first " - "
        pos = line.rfind("|")
        if pos != -1:
            colored_part = line[: pos + 1]
            remaining_part = line[pos + 1 :]

            html_line = (
                f'<span style="color:{color};">{colored_part}</span>'
                f"{remaining_part}<br><br>"
            )
        else:
            html_line = f'<span style="color:{color};">{line}</span><br><br>'

        formatted.append(html_line)
    return "".join(formatted)


# -----------------------------
# Tail watcher (LIVE)
# -----------------------------
class TailWatcher(FileSystemEventHandler, QObject):
    new_line = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._running = True
        self._position = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.observer = Observer()

    def start(self):
        self.observer.schedule(
            self, path=os.path.dirname(self.file_path), recursive=False
        )
        self.observer.start()
        # Do not emit initial content to avoid duplication

    def stop(self):
        self._running = False
        self.observer.stop()
        self.observer.join()

    def on_modified(self, event):
        if event.src_path == self.file_path and self._running:
            try:
                with open(self.file_path, "r") as f:
                    f.seek(self._position)
                    new_data = f.read()
                    self._position = f.tell()
                    if new_data:
                        # Remove blank/whitespace-only lines
                        filtered_lines = "\n".join(
                            line for line in new_data.splitlines() if line.strip()
                        )
                        if filtered_lines:
                            self.new_line.emit(format_log_block(filtered_lines))
            except Exception as e:
                self.new_line.emit(f"\n[watcher error] {e}\n")


# -----------------------------
# UI
# -----------------------------
class LogViewer(QWidget):
    def __init__(self, max_lines=10000):
        super().__init__()
        self.setWindowTitle("P-AIRCARS Dashboard")
        self.resize(1500, 950)

        self.max_lines = max_lines
        self.buffer = []
        self.tail_watcher = None
        self.current_log_path = None

        # ONLY call setup here (no UI creation here)
        self.setup_ui()
        self.refresh_logs()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_logs)
        self.refresh_timer.start(2000)

    #############################################
    # Categorization
    #############################################
    def categorize_log(self, name):
        if name.startswith("master"):
            return "master"
        elif name.startswith("subflow"):
            return "subflow"
        else:
            return "task"

    #############################################
    # UI (ONLY place where layout is created)
    #############################################
    def setup_ui(self):
        outer_layout = QGridLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        inner_layout = QVBoxLayout()

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # ---- Tabs ----
        self.tabs = QTabWidget()

        self.master_list = QListWidget()
        self.subflow_list = QListWidget()
        self.task_list = QListWidget()

        self.tabs.addTab(self.master_list, "Master Flow")
        self.tabs.addTab(self.subflow_list, "Subflows")
        self.tabs.addTab(self.task_list, "Tasks")

        # connect
        self.master_list.itemClicked.connect(self.load_log)
        self.subflow_list.itemClicked.connect(self.load_log)
        self.task_list.itemClicked.connect(self.load_log)

        # ---- Log viewer (HTML capable) ----
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(self.log_view)
        self.splitter.setSizes([350, 1150])

        inner_layout.addWidget(self.splitter)

        # ---- resize grip ----
        grip_layout = QHBoxLayout()
        grip_layout.addStretch()
        grip_layout.addWidget(QSizeGrip(self))
        inner_layout.addLayout(grip_layout)

        # ---- container ----
        inner_container = QWidget()
        inner_container.setObjectName("InnerContainer")
        inner_container.setLayout(inner_layout)

        outer_layout.addWidget(inner_container, 0, 0)

        # ---- styling (your dark UI) ----
        self.setStyleSheet(
            """
        QWidget {
            background-color: #1e1e1e;
            color: #dddddd;
            font-size: 16px;
        }
        QListWidget {
            background-color: #252526;
            border: none;
            padding: 6px;
        }
        QTextEdit {
            background-color: #1e1e1e;
            border: none;
            font-family: Consolas, monospace;
            font-size: 15px;
        }
        QTabBar::tab {
            background: #2d2d2d;
            padding: 12px 18px;
            margin: 2px;
            border-radius: 6px;
        }
        QTabBar::tab:selected {
            background: #007acc;
        }
        """
        )

    #############################################
    # Refresh logs
    #############################################
    def refresh_logs(self):
        lists = {
            "master": self.master_list,
            "subflow": self.subflow_list,
            "task": self.task_list,
        }

        existing_paths = {
            key: {
                lists[key].item(i).data(Qt.UserRole) for i in range(lists[key].count())
            }
            for key in lists
        }

        if os.path.isdir(LOG_DIR):
            log_files = [
                fname for fname in os.listdir(LOG_DIR) if fname.endswith(".log")
            ]

            log_files.sort(key=lambda f: os.path.getctime(os.path.join(LOG_DIR, f)))

            for fname in log_files:
                full_path = os.path.join(LOG_DIR, fname)
                category = self.categorize_log(fname)

                if full_path not in existing_paths[category]:
                    item = QListWidgetItem(get_logid(fname))
                    item.setData(Qt.UserRole, full_path)
                    lists[category].addItem(item)

    #############################################
    # Load log
    #############################################
    def load_log(self, item):
        new_log_path = item.data(Qt.UserRole)

        if self.tail_watcher:
            self.tail_watcher.stop()

        self.current_log_path = new_log_path
        self.buffer.clear()
        self.log_view.clear()

        try:
            with open(new_log_path, "r") as f:
                data = f.read()
                formatted = format_log_block(data)
                self.buffer = [formatted]
                self.log_view.setHtml(formatted)
                self.log_view.moveCursor(QTextCursor.End)
        except Exception as e:
            self.log_view.setPlainText(str(e))

        self.tail_watcher = TailWatcher(self.current_log_path)
        self.tail_watcher.new_line.connect(self.append_log_line)
        self.tail_watcher.start()

    #############################################
    # Append log (live)
    #############################################
    def append_log_line(self, text):
        scrollbar = self.log_view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 5  # tolerance

        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(text)

        # 🔥 FORCE scroll AFTER Qt updates layout
        if at_bottom:
            QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

    #############################################
    def closeEvent(self, event):
        if self.tail_watcher:
            self.tail_watcher.stop()
        QApplication.quit()


def cli():
    global LOG_DIR

    parser = argparse.ArgumentParser(
        description="P-AIRCARS Logger", formatter_class=SmartDefaultsHelpFormatter
    )
    parser.add_argument("--jobid", type=str, help="P-AIRCARS Jobid")
    parser.add_argument("--logdir", type=str, help="Log direcotory")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    cachedir = f"{get_datadir()}/{username}"
    os.makedirs(cachedir, exist_ok=True)

    try:
        if args.jobid is None and args.logdir is None:
            print("Provide jobid or logdir")
            sys.exit(1)

        if args.jobid:
            jobfile = f"{cachedir}/main_pids_{args.jobid}.txt"
            if not os.path.exists(jobfile):
                print("Invalid jobid")
                sys.exit(1)

            results = np.loadtxt(jobfile, dtype="str", unpack=True)
            workdir = str(results[4])
            log_dirs = glob.glob(f"{workdir}/*_{args.jobid}_target/logs")
            if len(log_dirs) > 0:
                LOG_DIR = log_dirs[0]
            else:
                print(
                    f"No log directory is present: {workdir}/*_{args.jobid}_target/logs"
                )
                sys.exit(1)
        else:
            if not os.path.exists(args.logdir):
                print("Invalid logdir")
                sys.exit(1)
            LOG_DIR = args.logdir

        os.environ["QT_OPENGL"] = "software"
        os.environ["QT_XCB_GL_INTEGRATION"] = "none"
        os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
        os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
        os.makedirs(f"{LOG_DIR}/xdgtmp", exist_ok=True)
        os.chmod(f"{LOG_DIR}/xdgtmp", 0o700)
        os.environ.setdefault("XDG_RUNTIME_DIR", f"{LOG_DIR}/xdgtmp")
        os.environ["XDG_RUNTIME_DIR"] = f"{LOG_DIR}/xdgtmp"
        os.environ["TMPDIR"] = f"{LOG_DIR}/xdgtmp"
        os.environ["QT_OPENGL"] = "software"
        os.environ["QT_STYLE_OVERRIDE"] = "Fusion"

        app = QApplication(sys.argv)
        viewer = LogViewer()
        viewer.show()
        sys.exit(app.exec_())
    except Exception:
        traceback.print_exc()
    finally:
        if LOG_DIR is not None and os.path.exists(LOG_DIR):
            drop_cache(LOG_DIR)
        os.system(f"rm -rf {LOG_DIR}/xdgtmp")
