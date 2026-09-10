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
from dask_jobqueue import SLURMCluster
from pyfiglet import Figlet
from collections import deque
from paircars.utils.basic_utils import get_cachedir
from paircars.utils.proc_manage_utils import (
    get_scheduler_name,
    detect_best_interface,
    get_jobid,
    get_total_nodes,
)


def is_slurm_job():
    """
    Check whether a job is submitted as slurm job or not
    """
    return any(
        var in os.environ
        for var in [
            "SLURM_JOB_ID",
            "SLURM_JOBID",  # older systems
            "SLURM_STEP_ID",
        ]
    )


def get_available_nodes(partition=None):
    """
    Get available nodes of the partition

    Parameters
    ----------
    partition : str, optional
        Partition name

    Returns
    -------
    list
        Available node names in the partition
    list
        All available node names
    """
    cmd = ["sinfo", "-h", "-N", "-o", "%N %t"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    all_available = []
    for line in result.stdout.splitlines():
        name, state = line.split()
        if state.startswith("idl") or state.startswith("mix"):
            all_available.append(name)
    available = []
    if partition:
        cmd.extend(["-p", partition])
            result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            name, state = line.split()
            if state.startswith("idl") or state.startswith("mix"):
                available.append(name)
    return available, all_available


def get_slurm_node_resources(partition=None, cpu_frac=0.8, mem_frac=0.8):
    """
    Get node resources for SLURM cluster

    Parameters
    ----------
    partition : str, optional
        Partition name
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Number of CPU threads
    float
        Memory in GB
    """
    if partition is not None:
        cmd = ["sinfo", "-h", "-p", partition, "-o", "%c %m"]
    else:
        cmd = ["sinfo", "-h", "-o", "%c %m"]
    out = subprocess.check_output(cmd).decode().strip().split("\n")
    cores = []
    mems = []
    for line in out:
        c, m = line.split()
        cores.append(int(c.rstrip("+")))
        mems.append(int(m.rstrip("+")))
    total_cpu = min(cores)
    total_mem = min(mems) / (1024)  # In GB
    cpu_frac = min(0.8, cpu_frac)
    mem_frac = min(0.8, mem_frac)
    ncpu = max(1, int(total_cpu * cpu_frac))
    mem = round(total_mem * mem_frac, 1)
    return ncpu, mem


def slurm_time_to_seconds(timestr):
    """
    Convert SLURM time format (D-HH:MM:SS or HH:MM:SS) to seconds.

    Parameters
    ----------
    timestr : str
        Time string in SLURM format

    Returns
    -------
    float
        Time in seconds
    """
    if timestr.lower() in ["infinite", "unlimited"]:
        return float("inf")
    if "-" in timestr:
        days, hms = timestr.split("-")
        h, m, s = map(int, hms.split(":"))
        return int(days) * 86400 + h * 3600 + m * 60 + s
    else:
        h, m, s = map(int, timestr.split(":"))
        return h * 3600 + m * 60 + s


def get_max_walltime(partition):
    """
    Get maximum wall time for the partition

    Parameters
    ----------
    partition : str
        Partition name

    Returns
    -------
    str
        Maximum wall time
    """
    result = subprocess.run(
        ["scontrol", "show", "partition"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to query SLURM partitions.")
    output = result.stdout
    partitions = {}
    blocks = output.split("\n\n")
    for block in blocks:
        name_match = re.search(r"PartitionName=(\S+)", block)
        time_match = re.search(r"MaxTime=(\S+)", block)
        if name_match and time_match:
            part_name = name_match.group(1)
            max_time = time_match.group(1)
            partitions[part_name] = max_time
    if partition not in partitions:
        raise ValueError(f"Partition {partition} not found.")
    max_time = partitions[partition]
    return max_time, slurm_time_to_seconds(max_time)


def get_slurm_dask_cluster(
    dask_dir,
    jobid=None,
    cpu_frac=0.8,
    mem_frac=0.8,
    min_mem=2,
    max_worker=1,
    partition=None,
    account=None,
    walltime=None,
    num_node=-1,
    python_path=None,
    spill_frac=0.7,
    verbose=True,
):
    """
    Launch a SLURMCluster

    Parameters
    ----------
    dask_dir : str
        Dask working directory (for temporary files)
    jobid : int
        JobID of P-AIRCARS to avoid mixup of cluster configurration with other P-AIRCARS jobs.
    cpu_frac : float, optional
        CPU fraction to use per node
    mem_frac : float, optional
        Memory fraction to use per node
    min_mem : float, optional
        Minimum per job memory in GB
    max_worker : int, optional
        Maximum number of parallel processes
    partition : str, optional
        SLURM partition name
        Note: If your cluster requires this, you should provide. Otherwise, error will occur.
    account : str, optional
        SLURM account name
        Note: If your cluster requires this, you should provide. Otherwise, error will occur.
    walltime : str, optional
        Job walltime, maximum time the SLURM job can run (HH:MM:SS)
    num_node : int, optional
        Maximum number of nodes to use (default: -1, use all nodes available in the account)
    spill_frac : float
        Fraction of memory to spill to disk
    verbose : bool
        Print Dask dashboard URL and diagnostics

    Returns
    -------
    client : dask.distributed.Client
        Connected Dask client
    cluster : dask_jobqueue.SLURMCluster
        SLURM Dask cluster
    str
        Dask directory used
    """
    logging.getLogger("distributed").setLevel(logging.ERROR)
    scheduler_name = get_scheduler_name()
    if scheduler_name != "slurm":
        print("SLURM is not avilable as job scheduler in your cluster.")
        return

    cpu_frac = min(0.8, abs(cpu_frac))
    mem_frac = min(0.8, abs(mem_frac))

    if jobid is None:
        jobid = get_jobid()

    os.makedirs(dask_dir, exist_ok=True)
    log_dir = f"{dask_dir}/slurm_logs"
    os.makedirs(log_dir, exist_ok=True)

    dask_dir = os.path.join(dask_dir.rstrip("/"), f"dask_{int(time.time())}")
    dask_dir_tmp = os.path.join(dask_dir, "tmp")
    os.makedirs(dask_dir_tmp, exist_ok=True)

    try:
        dask.config.set(
            {
                "temporary-directory": dask_dir_tmp,
                "distributed.worker.memory.target": spill_frac,
                "distributed.worker.memory.spill": spill_frac + 0.1,
                "distributed.worker.memory.pause": spill_frac + 0.2,
                "distributed.worker.memory.terminate": spill_frac + 0.25,
                "distributed.worker.daemon": False,
            }
        )
        if python_path is None:
            python_path = sys.executable
        interface = detect_best_interface()

        max_worker = max(1, max_worker)
        per_node_cpu, per_node_mem = get_slurm_node_resources(
            partition=partition, cpu_frac=cpu_frac, mem_frac=mem_frac
        )
        total_nodes = get_total_nodes(partition=partition)
        if num_node < 0:
            num_node = total_nodes
        elif num_node > total_nodes:
            num_node = total_nodes

        workers_per_node_mem = int(per_node_mem / min_mem)
        if workers_per_node_mem < 1:
            print(
                "Minimum available memory per node is not sufficient for at-least one worker on a node."
            )
            return
        workers_per_node_cpu = per_node_cpu
        workers_per_node = min(workers_per_node_mem, workers_per_node_cpu)
        max_workers_cluster = workers_per_node * num_node
        if max_worker > 0:
            max_workers_cluster = min(max_workers_cluster, max_worker)
            max_workers_cluster = max(2, max_workers_cluster)
        mem_limit = round(per_node_mem / workers_per_node, 2)
        ncpu = max(1, int(per_node_cpu / workers_per_node))

        env_extra = [
            "export PYTHONUNBUFFERED=1",
            f"export OMP_NUM_THREADS={ncpu}",
            f"export MKL_NUM_THREADS={ncpu}",
            f"export OPENBLAS_NUM_THREADS={ncpu}",
            f"export NUMEXPR_NUM_THREADS={ncpu}",
            f"export RAYON_NUM_THREADS={ncpu}",
            "export MALLOC_TRIM_THRESHOLD_=0",
            f"export TMPDIR={dask_dir_tmp}",
            f"export TMP={dask_dir_tmp}",
            f"export TEMP={dask_dir_tmp}",
            f"export DASK_TEMPORARY_DIRECTORY={dask_dir_tmp}",
            "export PYTHONWARNINGS=ignore::UserWarning:contextlib",
        ]

        job_extra = [
            f"--cpus-per-task={ncpu}",
            f"-J paircars_{jobid}",
            f"-o {log_dir}/paircars_{jobid}-%j.out",
            f"-e {log_dir}/paircars_{jobid}-%j.err",
        ]

        if walltime is None:
            walltime, _ = get_max_walltime(partition)
        else:
            max_time, max_time_second = get_max_walltime(partition)
            wall_time_second = slurm_time_to_seconds(walltime)
            if wall_time_second > max_time_second:
                print(
                    f"Walltime : {walltime} is larger than maximum allowed time: {max_time}."
                )
                walltime = max_time

        try:
            cluster = SLURMCluster(
                queue=partition,
                account=account,
                cores=1,  # This is important for CASA tasks
                n_workers=1,
                walltime=walltime,
                memory=f"{mem_limit}GiB",
                processes=1,
                python=python_path,
                local_directory=dask_dir_tmp,
                death_timeout=300,
                log_directory=log_dir,
                name=f"paircars_{jobid}",
                shared_temp_directory=dask_dir_tmp,
                job_extra_directives=job_extra,
                job_script_prologue=env_extra,
            )
            client = Client(cluster, heartbeat_interval="5s")
            client.run_on_scheduler(gc.collect)
            print("Using interface: auto-detected")
        except Exception:
            cluster = SLURMCluster(
                queue=partition,
                account=account,
                cores=1,  # This is important for CASA tasks
                n_workers=1,
                walltime=walltime,
                memory=f"{mem_limit}GiB",
                processes=1,
                python=python_path,
                local_directory=dask_dir_tmp,
                death_timeout=300,
                log_directory=log_dir,
                name=f"paircars_{jobid}",
                shared_temp_directory=dask_dir_tmp,
                job_extra_directives=job_extra,
                job_script_prologue=env_extra,
                interface=interface,
            )
            client = Client(cluster, heartbeat_interval="5s")
            client.run_on_scheduler(gc.collect)
            print(f"Using interface: {interface}")

        if verbose:
            print("####################################################")
            print(f"Dask dashboard available at: {client.dashboard_link}")
            print(f"Total usable cpu per node: {per_node_cpu}")
            print(f"Total usable memory per node: {per_node_mem} GB")
            print(f"CPU per worker: {ncpu}")
            print(f"Memory per worker: {mem_limit}GB")
            print(f"Maximum number of workers: {max_workers_cluster}")
            print("####################################################")
        return client, cluster, dask_dir, max_workers_cluster
    except Exception:
        print("Error occured in creating SLURM cluster.")
        traceback.print_exc()
        os.system(f"rm -rf {log_dir} {dask_dir}")
        return
        

def submit_slurm_master_flow(args, jobid):
    """
    Submit P-AIRCARS master flow to a slurm cluster

    Parameters
    ----------
    args : dict
        Arparser dictionary
    jobid : int
        P-AIRCARS jobid

    Returns
    -------
    int
        Success message
    """
    scheduler_name = get_scheduler_name()
    if scheduler_name != "slurm":
        print("SLURM job scheduler is not available.")
        return 1
    args_list = [shlex.quote(arg) for arg in sys.argv[1:]]
    if "--log2term" in args_list:
        args_list.remove("--log2term")
    cli_cmd = "run-mwa-masterflow " + " ".join(args_list) + f" --jobid {jobid}"

    if hasattr(args, "log2term"):
        log2term = args.log2term
    else:
        log2term = False

    if hasattr(args, "partition") and args.partition is not None:
        max_time, max_time_second = get_max_walltime(args.partition)
    else:
        print("Please provide partition name to run SLURM jobs.")
        return 1
    if hasattr(args, "workdir") and args.workdir is not None:
        os.makedirs(args.workdir, exist_ok=True)
    else:
        print("Please provide a work directory.")
        return 1

    cachedir = f"{get_cachedir()}/prefect_{scheduler_name}"
    config_file = f"{cachedir}/prefect.config.npy"
    config = np.load(config_file, allow_pickle=True).all()
    if os.path.exists(config_file) is False:
        print(
            f"Prefect server configuration is not availble. It is required for P-AIRCARS to run in cluster with scheduler: {scheduler_name}"
        )
        return 1
    load_dotenv(dotenv_path=config["ENV_FILE"], override=True)
    envlist = os.environ
    envlist["PREFECT_API_URL"] = config["NODE_URL"]

    prefect_env_list = []
    for env in envlist:
        if "PREFECT" in env:
            prefect_env_list.append(f"export {env}={envlist.get(env)}")

    log_file = f"{args.workdir}/main_paircars_{jobid}.log"

    try:
        #################################
        # Determining wall time
        #################################
        if hasattr(args, "walltime"):
            if args.walltime is None:
                walltime = max_time
            else:
                wall_time_second = slurm_time_to_seconds(args.walltime)
                if wall_time_second > max_time_second:
                    print(
                        f"Walltime : {args.walltime} is larger than maximum allowed time: {max_time}."
                    )
                    walltime = max_time
                else:
                    walltime = args.walltime
        else:
            walltime = max_time
        #############################
        # Determining cpu and memory
        #############################
        if hasattr(args, "cpu_frac") is False:
            cpu_frac = 0.8
        else:
            cpu_frac = args.cpu_frac
        if hasattr(args, "mem_frac") is False:
            mem_frac = 0.8
        else:
            mem_frac = args.mem_frac
        ncpu, mem = get_slurm_node_resources(
            partition=args.partition, cpu_frac=cpu_frac, mem_frac=mem_frac
        )
        script_args = [
            "#!/bin/bash",
            f"#SBATCH --job-name=paircars_{jobid}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --output={log_file}",
            f"#SBATCH --error={log_file}",
            f"#SBATCH --partition={args.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={min(8,ncpu)}",
            f"#SBATCH --mem={min(16,mem)}G",
        ]
        if hasattr(args, "account") and args.account is not None:
            script_args.append(f"#SBATCH --account={args.account}\n")
        if len(prefect_env_list) > 0:
            for i in prefect_env_list:
                script_args.append(f"{i}")
        script_args.append("unset PYTHONPATH\n")
        script_args.append("export PYTHONNOUSERSITE=1\n")
        script_args.append("export PYTHONUNBUFFERED=1\n")
        script_args.append(cli_cmd)
        script_path = os.path.join(args.workdir, f"paircars_slurm_{jobid}.sh")
        with open(script_path, "w") as f:
            for script_arg in script_args:
                f.write(f"{script_arg}\n")
        f = Figlet(font="big")
        print(f.renderText("P-AIRCARS"))
        print("######################################################")
        print(f"P-AIRCARS Job ID: {jobid}")
        print(f"Batch script: {script_path} is ready for submission.")
        print(f"Main logger: {log_file}")
        print("######################################################")
        result = subprocess.run(["sbatch", script_path], stderr=subprocess.DEVNULL)
        exit_code = result.returncode
        if exit_code == 0:
            print(f"P-AIRCARS job with Job ID: {jobid} is submitted successfully.")
            if not log2term:
                return exit_code
            else:
                print("Streaming logs to terminal...\n")
                last_lines = deque(maxlen=500)
                only_run_print = False
                printing_traceback = False
                traceback_waittime = None
                last_write_time = time.time()
                with open(log_file, "r") as log:
                    log.seek(0, os.SEEK_END)
                    while True:
                        line = log.readline()
                        wait_time = time.time() - last_write_time
                        if (
                            traceback_waittime is not None
                            and wait_time > traceback_waittime
                        ):
                            return 1
                        if not line:
                            time.sleep(0.5)
                            continue
                        last_lines.append(line)
                        lower = line.lower()
                        upper = line.upper()
                        if (
                            "task run" in lower or "flow run" in lower
                        ) and "p-aircars execution is finished" not in lower:
                            only_run_print = True
                        if (
                            "killed" in lower
                            or (
                                "ERROR" in upper
                                and "flow run" in lower
                                and f"paircars_{jobid}" in lower
                            )
                        ) and not printing_traceback:
                            printing_traceback = True
                            traceback_waittime = 300
                        if (
                            printing_traceback
                            or not only_run_print
                            or ("task run" in lower or "flow run" in lower)
                            or "p-aircars execution is finished" in lower
                        ):
                            sys.stdout.write(line)
                            sys.stdout.flush()
                            last_write_time = time.time()
                        if (
                            "p-aircars execution is finished" in lower
                            or "cluster closed" in lower
                        ):
                            return 0
        else:
            print(f"P-AIRCARS job with Job ID: {jobid} could not be submitted.")
        return 0 if exit_code == 0 else 1
    except Exception:
        traceback.print_exc()
        return 1
