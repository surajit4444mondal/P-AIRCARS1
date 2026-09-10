import psutil
import argparse
import traceback
import glob
import sys
import os
import subprocess
import getpass
from paircars.utils.basic_utils import get_datadir, print_banner
from paircars.utils.logger_utils import SmartDefaultsHelpFormatter
from paircars.utils.proc_manage_utils import get_scheduler_name

username = getpass.getuser()


def is_slurm_job_running(job_id, node_name=None):
    """
    Returns True if job_id is RUNNING on node.

    Parameters
    ----------
    job_id : int
        Slurm job ID
    node_name : str, optional
        Node name
    """
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T %N"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        return False  # job not active
    # Output format: "RUNNING node45"
    parts = output.split()
    state = parts[0]
    nodes = " ".join(parts[1:])  # handles multi-node jobs
    if node_name is not None:
        return state == "RUNNING" and node_name in nodes
    else:
        return state == "RUNNING"


def show_local_job_status(clean_old_jobs=False):
    """
    Show P-AIRCARS local cluster jobs status

    Parameters
    ----------
    clean_old_jobs : bool, optional
        Clean old informations for stopped jobs

    Returns
    -------
    int
        Number of jobs running
    """
    cachedir = f"{get_datadir()}/{username}"
    os.makedirs(cachedir, exist_ok=True)
    msg = 0
    try:
        main_pid_files = glob.glob(f"{cachedir}/main_pids_*.txt")
        if len(main_pid_files) == 0:
            print("No P-AIRCARS jobs is running.")
        else:
            for pid_file in main_pid_files:
                with open(pid_file, "r") as f:
                    line = f.read().split(" ")
                jobid = line[0]
                pid = line[1]
                workdir = line[4]
                outdir = line[5]
                if psutil.pid_exists(int(pid)):
                    running = "Running/Waiting"
                    msg += 1
                else:
                    running = "Done/Stopped"
                print_banner(
                    f"Job ID: {jobid}, Work directory: {workdir}, Output directory: {outdir}, Status: {running}"
                )
                if clean_old_jobs and running == "Done/Stopped":
                    print(f"Removed {jobid}")
                    os.system(f"rm -rf {pid_file}")
    except Exception:
        traceback.print_exc()
    finally:
        return msg


def show_slurm_job_status(clean_old_jobs=False, node_name=None, print_status=True):
    """
    Show P-AIRCARS slurm cluster jobs status

    Parameters
    ----------
    clean_old_jobs : bool, optional
        Clean old informations for stopped jobs
    node_name : str, optional
        Node name of slurm cluster
    print_status : bool, optional
        Print status on terminal

    Returns
    -------
    int
        Number of jobs running
    """
    cachedir = f"{get_datadir()}/{username}"
    os.makedirs(cachedir, exist_ok=True)
    msg = 0
    try:
        main_pid_files = glob.glob(f"{cachedir}/main_pids_*.txt")
        if len(main_pid_files) == 0 and print_status:
            print("No P-AIRCARS jobs is running.")
        else:
            for pid_file in main_pid_files:
                with open(pid_file, "r") as f:
                    line = f.read().split(" ")
                jobid = line[0]
                pid = line[1]
                workdir = line[4]
                outdir = line[5]
                if node_name is not None:
                    if is_slurm_job_running(int(pid), node_name=node_name):
                        running = f"Running/Waiting in node: {node_name}"
                        msg += 1
                    elif is_slurm_job_running(int(pid)):
                        running = "Running/Waiting in different node"
                    else:
                        running = "Done/Stopped"
                elif is_slurm_job_running(int(pid)):
                    running = "Running/Waiting in any node"
                    msg += 1
                else:
                    running = "Done/Stopped"
                if print_status:
                    print_banner(
                        f"Job ID: {jobid}, Work directory: {workdir}, Output directory: {outdir}, Status: {running}"
                    )
                if clean_old_jobs and running == "Done/Stopped":
                    print(f"Removed {jobid}")
                    os.system(f"rm -rf {pid_file}")
    except Exception:
        traceback.print_exc()
    finally:
        return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Show P-AIRCARS jobs status.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        dest="show",
        help="Show job status",
    )
    parser.add_argument(
        "--clean_old_jobs",
        action="store_true",
        help="Clean old jobs",
    )
    parser.add_argument(
        "--node_name",
        type=str,
        default=None,
        help="Slurm node name",
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    scheduler_name = get_scheduler_name()
    print("####################")
    print("P-AIRCARS Job status")
    print("####################")
    try:
        args = parser.parse_args()
        if args.show:
            if scheduler_name == "local":
                show_local_job_status(clean_old_jobs=args.clean_old_jobs)
            elif scheduler_name == "slurm":
                show_slurm_job_status(
                    clean_old_jobs=args.clean_old_jobs, node_name=args.node_name
                )
            else:
                print(f"P-AIRCARS is not ready for job scheduler: {scheduler_name}")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    cli()
