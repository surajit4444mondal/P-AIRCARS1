import dask
import gc
import time
import os
import subprocess
import sys
import traceback
import logging
import shlex
import re
import numpy as np
from dotenv import load_dotenv
from dask.distributed import Client
from dask_jobqueue import PBSCluster
from pyfiglet import Figlet
from collections import deque
from paircars.utils.basic_utils import get_cachedir
from paircars.utils.proc_manage_utils import (
    get_scheduler_name,
    detect_best_interface,
    get_jobid,
    get_total_nodes,
)

def is_pbs_job():
    """
    Check whether the current process is running as a PBS job.
    """
    return any(
        var in os.environ
        for var in [
            "PBS_JOBID",
            "PBS_JOBNAME",
            "PBS_NODEFILE",
        ]
    )
    
def get_available_nodes(queue=None):
    """
    Get available nodes of a PBS queue.

    Parameters
    ----------
    queue : str, optional
        PBS queue name.

    Returns
    -------
    list
        Available node names in the given queue.
    list 
        All available node names
    """
    cmd = ["pbsnodes", "-a", "-S"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    available = []
    all_available = []
    for line in result.stdout.splitlines():
        fields = line.split()
        # Skip header/empty lines
        if not fields or fields[0].lower() in ["vnode", "node"]:
            continue
        name = fields[0]
        state = fields[1] if len(fields) > 2 else ""
        q = fields[5]
        if state in ["free", "job-busy"]:
            all_available.append(name)
            if queue is None or q==queue:
                available.append(name)
    return available, all_available
    
    
def get_pbs_node_resources(queue=None, cpu_frac=0.8, mem_frac=0.8):
    """
    Get node resources for a PBS/OpenPBS cluster.

    Parameters
    ----------
    queue : str, optional
        PBS queue name. If specified, try to restrict nodes to this queue.
    cpu_frac : float, optional
        Fraction of CPUs to use.
    mem_frac : float, optional
        Fraction of memory to use.

    Returns
    -------
    ncpu : int
        Number of CPU threads to use.
    mem : float
        Memory in GB to use.
    """
    cmd = ["pbsnodes", "-a"]
    out = subprocess.check_output(cmd, text=True)
    cores = []
    mems = []
    current_node = None
    node_data = {}
    for line in out.splitlines():
        # New node
        if line and not line.startswith((" ", "\t")):
            current_node = line.strip()
            node_data[current_node] = {}
        elif current_node is not None:
            line = line.strip()
            # Example:
            # resources_available.ncpus = 64
            # resources_available.mem = 250gb
            if "=" in line:
                key, value = [x.strip() for x in line.split("=", 1)]
                node_data[current_node][key] = value
    for node, data in node_data.items():
        # If queue information exists on the node, filter it
        if queue is not None:
            node_queue = data.get("queue")
            if node_queue is not None:
                node_queue = node_queue.lstrip("@")
                if node_queue != queue:
                    continue
        try:
            cpu = int(data["resources_available.ncpus"])
            mem_string = data["resources_available.mem"].lower()
            if mem_string.endswith("gb"):
                mem = float(mem_string[:-2])
            elif mem_string.endswith("g"):
                mem = float(mem_string[:-1])
            elif mem_string.endswith("mb"):
                mem = float(mem_string[:-2]) / 1024
            elif mem_string.endswith("kb"):
                mem = float(mem_string[:-2]) / (1024 ** 2)
            else:
                # PBS memory is commonly reported in bytes if no unit
                mem = float(mem_string) / (1024 ** 3)
            cores.append(cpu)
            mems.append(mem)
        except (KeyError, ValueError):
            continue
    if not cores:
        print(
            f"No PBS nodes with usable resources found"
            + (f" for queue '{queue}'" if queue else "")
        )
        return None, None
    # Use minimum node resources so that a job fits on every candidate node
    total_cpu = min(cores)
    total_mem = min(mems)
    cpu_frac = min(0.8, cpu_frac)
    mem_frac = min(0.8, mem_frac)
    ncpu = max(1, int(total_cpu * cpu_frac))
    mem = round(total_mem * mem_frac, 1)
    return ncpu, mem
    

def pbs_time_to_seconds(time_str):
    """
    Convert PBS walltime to seconds.

    Parameters
    ----------
    time_str : str
        PBS time string
        Supported formats:
            HH:MM:SS
            MM:SS
            DD:HH:MM:SS
            
    Returns
    -------
    float
        Time in seconds
    """
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 4:
        days, hours, minutes, seconds = parts
        return (
            days * 86400
            + hours * 3600
            + minutes * 60
            + seconds
        )
    else:
        print(f"Invalid PBS walltime: {time_str}")
        return 


def get_max_walltime(queue):
    """
    Get maximum wall time for a PBS queue.

    Parameters
    ----------
    queue : str
        PBS queue name.

    Returns
    -------
    max_time : str
        Maximum wall time in PBS format.
    max_time_seconds : int
        Maximum wall time in seconds.
    """
    result = subprocess.run(
        ["qstat", "-Q", "-f", queue],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"Failed to query PBS queue '{queue}': "
            f"{result.stderr.strip()}"
        )
        return None, None
    output = result.stdout
    # Look for:
    # resources_max.walltime = 48:00:00
    match = re.search(
        r"resources_max\.walltime\s*=\s*(\S+)",
        output
    )
    if not match:
        print(
            f"Maximum walltime not defined for PBS queue '{queue}'."
        )
        return "UNLIMITED", np.inf
    max_time = match.group(1)
    return max_time, pbs_time_to_seconds(max_time)
    
    

    
    
