import psutil
import numpy as np
import time
import os
import signal
import traceback
import subprocess
import getpass
from distributed import Client
from .basic_utils import get_datadir, check_port_status
from .resource_utils import drop_cache


username = getpass.getuser()


def kill_port(port):
    """
    Kill a running port

    Parameters
    ----------
    port : int
        Port number
    """
    if check_port_status(port) is False:
        print(f"Closing port : {port}.")
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
        )
        for pid in result.stdout.split():
            os.kill(int(pid), signal.SIGKILL)


def terminate_process_and_children(pid, grace_period=3.0):
    """
    Try to gracefully terminate a process tree, then force kill if needed.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        parent.terminate()
        gone, alive = psutil.wait_procs([parent] + children, timeout=grace_period)
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass


def kill_localscheduler(jobid):
    """
    Kill local scheduler

    Parameters
    ----------
    jobid : int
        P-AIRCARS job ID
    """
    try:
        cachedir = f"{get_datadir()}/{username}"
        os.makedirs(cachedir, exist_ok=True)
        jobfile_name = f"{cachedir}/main_pids_{jobid}.txt"
        if os.path.exists(jobfile_name) is False:
            print(f"No P-AIRCARS job information available for job ID; {jobfile_name}")
            return
        try:
            results = np.loadtxt(jobfile_name, dtype="str", unpack=True)
            main_pid = int(results[1])
            scheduler_address = str(results[2])
            msdir = str(results[3])
            workdir = str(results[4])
            outdir = str(results[5])
        except Exception:
            print("Could not read job file.")
            traceback.print_exc()
            return

        try:
            print("Closing dask cluster....")
            client = Client(address=scheduler_address, timeout=30)
            client.shutdown()
            client.close()
        except Exception:
            print(f"Dask cluster at: {scheduler_address} is already closed.")

        time.sleep(1)
        print(f"Attempting to terminate main PID: {main_pid}")
        terminate_process_and_children(main_pid)

        print("Dropping caches...")
        drop_cache(msdir)
        drop_cache(workdir)
        drop_cache(outdir)
        drop_cache(cachedir)
        print("Cleanup complete.")
        os.system(f"rm -rf {jobfile_name}")
        return
    except Exception:
        print(f"Error in killing P-AIRCARS job: {jobid}")
        traceback.print_exc()
        return


def kill_slurmscheduler(jobid):
    """
    Kill local scheduler

    Parameters
    ----------
    jobid : int
        P-AIRCARS job ID
    """
    try:
        cachedir = f"{get_datadir()}/{username}"
        os.makedirs(cachedir, exist_ok=True)
        jobfile_name = f"{cachedir}/main_pids_{jobid}.txt"
        if os.path.exists(jobfile_name) is False:
            print(f"No P-AIRCARS job information available for job ID; {jobfile_name}")
            return
        try:
            results = np.loadtxt(jobfile_name, dtype="str", unpack=True)
            main_jobid = int(results[1])
            scheduler_address = str(results[2])
            msdir = str(results[3])
            workdir = str(results[4])
            outdir = str(results[5])
        except Exception:
            print("Could not read job file.")
            traceback.print_exc()
            return

        try:
            print("Closing dask cluster....")
            client = Client(address=scheduler_address, timeout=30)
            client.shutdown()
            client.close()
        except Exception:
            print(f"Dask cluster at: {scheduler_address} is already closed.")
        time.sleep(1)

        print(f"Attempting to terminate main slurm jobid: {main_jobid}")
        subprocess.run(["scancel", f"{main_jobid}"])

        print("Dropping caches...")
        drop_cache(msdir)
        drop_cache(workdir)
        drop_cache(outdir)
        drop_cache(cachedir)
        print("Cleanup complete.")
        os.system(f"rm -rf {jobfile_name}")
        return
    except Exception:
        print(f"Error in killing P-AIRCARS job: {jobid}")
        traceback.print_exc()
        return
