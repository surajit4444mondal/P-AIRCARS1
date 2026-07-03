import resource
import psutil
import dask
import numpy as np
import gc
import logging
import time
import glob
import os
import subprocess
import sys
import shutil
import socket
import shlex
import traceback
from dotenv import load_dotenv
from dask.distributed import Client, LocalCluster
from datetime import datetime as dt, timedelta
from pyfiglet import Figlet
from collections import deque
from .basic_utils import get_cachedir


#################################
# Process management
#################################
def get_jobid():
    """
    Get Job ID with millisecond-level uniqueness.

    Returns
    -------
    int
        Job ID in the format YYYYMMDDHHMMSSmmm (milliseconds)
    """
    cachedir = get_cachedir()
    jobid_file = os.path.join(cachedir, "jobids.txt")
    if os.path.exists(jobid_file):
        prev_jobids = np.loadtxt(jobid_file, unpack=True, dtype="int64")
        if prev_jobids.size == 0:
            prev_jobids = []
        elif prev_jobids.size == 1:
            prev_jobids = [str(prev_jobids)]
        else:
            prev_jobids = [str(jid) for jid in prev_jobids]
    else:
        prev_jobids = []

    if len(prev_jobids) > 0:
        FORMAT = "%Y%m%d%H%M%S%f"
        CUTOFF = dt.utcnow() - timedelta(days=15)
        filtered_prev_jobids = []
        for job_id in prev_jobids:
            job_time = dt.strptime(job_id.ljust(20, "0"), FORMAT)  # pad if truncated
            if job_time >= CUTOFF or job_id == 0:  # Job ID 0 is always kept
                filtered_prev_jobids.append(job_id)
        prev_jobids = filtered_prev_jobids

    now = dt.utcnow()
    cur_jobid = (
        now.strftime("%Y%m%d%H%M%S") + f"{int(now.microsecond/1000):03d}"
    )  # ms = first 3 digits of microseconds
    prev_jobids.append(cur_jobid)

    job_ids_int = np.array(prev_jobids, dtype=np.int64)
    np.savetxt(jobid_file, job_ids_int, fmt="%d")

    return int(cur_jobid)


def save_main_process_info(
    pid, jobid, scheduler_address, msdir, workdir, outdir, cpu_frac, mem_frac
):
    """
    Save main processes info

    Parameters
    ----------
    pid : int
        Main job process id for local cluster or scheduler job id
    jobid : int
        Job ID
    scheduler_address : str
        Dask scheduler address
    msdir : str
        Measurement set directory
    workdir : str
        Work directory
    outdir : str
        Output directory
    cpu_frac : float
        CPU fraction of the job
    mem_frac : float
        Memory fraction of the job

    Returns
    -------
    str
        Job info file name
    """
    cachedir = get_cachedir()
    prev_main_pids = glob.glob(f"{cachedir}/main_pids_*.txt")
    prev_jobids = [
        str(os.path.basename(i).rstrip(".txt").split("main_pids_")[-1])
        for i in prev_main_pids
    ]
    if len(prev_jobids) > 0:
        FORMAT = "%Y%m%d%H%M%S%f"
        CUTOFF = dt.utcnow() - timedelta(days=15)
        filtered_prev_jobids = []
        for i in range(len(prev_jobids)):
            job_id = prev_jobids[i]
            job_time = dt.strptime(job_id.ljust(20, "0"), FORMAT)  # pad if truncated
            if job_time > CUTOFF or job_id == 0:  # Job ID 0 is always kept
                filtered_prev_jobids.append(job_id)
            else:
                os.system(f"rm -rf {prev_main_pids[i]}")
    main_job_file = f"{cachedir}/main_pids_{jobid}.txt"
    main_str = f"{jobid} {pid} {scheduler_address} {msdir} {workdir} {outdir} {cpu_frac} {mem_frac}"
    with open(main_job_file, "w") as f:
        f.write(main_str)
    return main_job_file


def get_total_worker(client):
    """
    Get total workers in the cluster

    Parameters
    ----------
    client : dask.client
        Dask client for the cluster

    Returns
    -------
    int
        Number of workers
    """
    return len(client.scheduler_info()["workers"])


def scale_worker_and_wait(
    dask_cluster,
    dask_client,
    nworker,
    timeout=60,
):
    """
    Scale worker and wait until it is done

    Parameters
    ----------
    dask_cluster : dask.cluster
        Dask cluster
    dask_client : dask.client
        Dask client for the same cluster
    nworker : int
        Number of worker
    timeout : float, optional
        Timeout, show a warning and move
    """
    print(f"Start scaling to {nworker} workers")
    nworker = max(2, nworker)  # Safety, never scale to 1 worker
    dask_cluster.scale(nworker)
    try:
        dask_client.wait_for_workers(nworker, timeout=timeout)
        print(f"Successfully scaled to {nworker} workers")
        return 0
    except TimeoutError:
        workers = get_total_worker(dask_client)
        print(f"Scaling timeout. Current workers: {workers}")
        return 1


def get_local_dask_cluster(
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    min_mem=2,
    max_worker=-1,
    spill_frac=0.7,
    wait_time=10.0,
    verbose=True,
):
    """
    Create a local Dask cluster

    Parameters
    ----------
    dask_dir : str
        Dask temporary directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Fraction of total memory to use
    min_mem : float, optional
        Minimum required per job memory in GB
    max_worker : int, optional
        Maximum worker
    spill_frac : float, optional
        Spill to disk at this fraction
    wait_time : float, optional
        Wait time in seconds
    verbose : bool, optional
        Verbose (details of cluster)

    Returns
    -------
    client : dask.distributed.Client
        Dask client
    cluster : dask.distributed.LocalCluster
        Dask cluster
    str
        Dask directory
    int
        Number of workers
    """
    cpu_frac = min(abs(cpu_frac), 0.8)
    mem_frac = min(abs(mem_frac), 0.8)
    max_worker = max(2, max_worker)
    logging.getLogger("distributed").setLevel(logging.ERROR)
    print("Creating local cluster on the current node.")
    # Set up Dask working directories
    dask_dir = f"{dask_dir.rstrip('/')}/dask_{int(time.time())}"
    os.makedirs(dask_dir, exist_ok=True)
    dask_dir_tmp = f"{dask_dir}/tmp"
    os.makedirs(dask_dir_tmp, exist_ok=True)
    min_mem = max(0.1, min_mem)
    try:
        # Raise file descriptor limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < int(hard * 0.8):
            resource.setrlimit(resource.RLIMIT_NOFILE, (int(hard * 0.8), hard))
        dask.config.set(
            {
                "temporary-directory": dask_dir,
                "distributed.worker.memory.target": spill_frac,
                "distributed.worker.memory.spill": spill_frac + 0.1,
                "distributed.worker.memory.pause": spill_frac + 0.2,
                "distributed.worker.memory.terminate": spill_frac + 0.25,
            }
        )
        min_mem /= spill_frac  # Accounting for spill fraction
        usable_cpu = max(1, int(psutil.cpu_count() * cpu_frac))
        total_mem = psutil.virtual_memory().total / 1024**3  # In GB
        usable_mem = round(total_mem * mem_frac, 2)
        n_worker_mem = int(usable_mem / min_mem)
        if n_worker_mem < 2:
            print(
                f"Minimum available memory: {usable_mem}GB is not sufficient for at-least 2 workers."
            )
            return None, None, dask_dir, n_worker_mem

        n_worker_cpu = usable_cpu
        n_worker = min(n_worker_cpu, n_worker_mem)
        if max_worker > 0:
            n_worker = min(n_worker, max_worker)
            n_worker = max(2, n_worker)

        mem_limit = round(usable_mem / n_worker, 2)
        n_worker = max(1, int(usable_mem / mem_limit))
        ncpu = max(1, int(usable_cpu / n_worker))
        env = {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": f"{ncpu}",
            "MKL_NUM_THREADS": f"{ncpu}",
            "OPENBLAS_NUM_THREADS": f"{ncpu}",
            "NUMEXPR_NUM_THREADS": f"{ncpu}",
            "RAYON_NUM_THREADS": f"{ncpu}",
            "MALLOC_TRIM_THRESHOLD_": "0",
            "TMPDIR": f"{dask_dir_tmp}",
            "TMP": f"{dask_dir_tmp}",
            "TEMP": f"{dask_dir_tmp}",
            "DASK_TEMPORARY_DIRECTORY": f"{dask_dir_tmp}",
            "PYTHONWARNINGS": "ignore::UserWarning:contextlib",
        }
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            memory_limit=f"{mem_limit}GiB",
            local_directory=dask_dir,
            dashboard_address=":0",
            processes=True,
            env=env,
        )
        client = Client(cluster, heartbeat_interval="5s")
        client.run_on_scheduler(gc.collect)
        if verbose:
            print("####################################################")
            print(f"Dask dashboard available at: {client.dashboard_link}")
            print(f"Total usable cpu: {usable_cpu}")
            print(f"Total usable memory: {usable_mem}GB")
            print(f"CPU per worker: {ncpu}")
            print(f"Memory per worker: {mem_limit}GB")
            print(f"Maximum number of workers: {n_worker}")
            print("####################################################")
        os.environ.update(env)
        return client, cluster, dask_dir, n_worker
    except Exception:
        print("Error occured in creating local cluster.")
        traceback.print_exc()
        os.system(f"rm -rf {dask_dir_tmp}")
        return


def submit_local_master_flow(args, jobid):
    """
    Submit P-AIRCARS master flow to a local cluster

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
    if scheduler_name != "local":
        print(
            f"Job scheduler is not local. Available job scheduler is : {scheduler_name}"
        )
        cont=input("Do you want to continue to run P-AIRCARS in current node only? Y/N")
        if cont.lower()=="n":
            return 1
            
    args_list = [shlex.quote(arg) for arg in sys.argv[1:]]
    if "--log2term" in args_list:
        args_list.remove("--log2term")

    cli_cmd = "run-mwa-masterflow " + " ".join(args_list) + f" --jobid {jobid}"

    if hasattr(args, "workdir") and args.workdir is not None:
        os.makedirs(args.workdir, exist_ok=True)
    else:
        print("Please provide a work directory.")
        return 1

    if hasattr(args, "log2term"):
        log2term = args.log2term
    else:
        log2term = False

    cachedir = f"{get_cachedir()}/prefect_{scheduler_name}"
    config_file = f"{cachedir}/prefect.config.npy"
    prefect_env_list = []
    if os.path.exists(config_file):
        config = np.load(config_file, allow_pickle=True).all()
        load_dotenv(dotenv_path=config["ENV_FILE"], override=True)
        envlist = os.environ
        envlist["PREFECT_API_URL"] = config["NODE_URL"]
        for env in envlist:
            if "PREFECT" in env:
                prefect_env_list.append(f"export {env}={envlist.get(env)}")
    log_file = f"{args.workdir}/main_paircars_{jobid}.log"

    try:
        script_args = ["#!/bin/bash\n"]
        script_args.append("unset PYTHONPATH\n")
        script_args.append("export PYTHONNOUSERSITE=1\n")
        if len(prefect_env_list) > 0:
            for i in prefect_env_list:
                script_args.append(f"{i}")
        script_args.append("export PYTHONUNBUFFERED=1\n")
        script_args.append(cli_cmd)
        script_path = os.path.join(args.workdir, f"paircars_local_{jobid}.sh")
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
        try:
            # Always run job in background
            with open(log_file, "a", buffering=1) as log:
                proc = subprocess.Popen(
                    ["bash", script_path],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    bufsize=1,
                )
            print("Master flow started in background")
            print(f"Main log: {log_file}")
            if not log2term:
                return 0
            print("Streaming logs to terminal...\n")
            last_lines = deque(maxlen=500)
            only_run_print = False
            printing_traceback = False
            traceback_waittime = None
            last_write_time = time.time()
            with open(log_file, "r") as log:
                log.seek(0, os.SEEK_END)
                while True:
                    exit_code = proc.poll()
                    line = log.readline()
                    wait_time = time.time() - last_write_time
                    if exit_code is not None and (
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
        except Exception:
            traceback.print_exc()
            return 1
    except Exception:
        traceback.print_exc()
        return 1


##############################################
# Scheduler and hardware architecture related
##############################################
def detect_best_interface(scheduler_ip=None):
    """
    Automatically detect best IPv4 network interface for Dask.

    Parameters
    ----------
    scheduler_ip : str, optional
        If provided, ensures selected interface can route to this IP.

    Returns
    -------
    str or None
        Best interface name or None if not found.
    """
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()
    candidates = []
    for iface, iface_addrs in addrs.items():
        if iface == "lo":
            continue
        if iface.startswith(("docker", "veth", "br", "wl")):
            continue
        # Interface must be UP
        if iface not in stats or not stats[iface].isup:
            continue
        # Must have IPv4
        ipv4 = None
        for addr in iface_addrs:
            if addr.family == socket.AF_INET:
                if not addr.address.startswith("127."):
                    ipv4 = addr.address
                    break
        if ipv4:
            candidates.append((iface, ipv4))
    if not candidates:
        return None
    # If scheduler_ip provided, check routing
    if scheduler_ip:
        for iface, ip in candidates:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((ip, 0))
                s.settimeout(1)
                s.connect((scheduler_ip, 80))
                s.close()
                return iface
            except Exception:
                continue
    # Prefer InfiniBand if available
    for iface, _ in candidates:
        if iface.startswith("ib"):
            return iface
    # Otherwise return first valid interface
    return candidates[0][0]


def get_scheduler_name():
    """
    Get job scheduler available

    Returns
    -------
    str
        Scheduler name (local, pbs, slurm)
    """
    if shutil.which("sbatch"):
        return "slurm"
    elif shutil.which("bsub"):
        return "lsf"
    elif shutil.which("qhost"):
        return "sge"
    elif shutil.which("qsub"):
        return "pbs"
    elif shutil.which("condor_submit"):
        return "htcondor"
    elif shutil.which("msub"):
        return "mab"
    elif shutil.which("oarsub"):
        return "oar"
    else:
        return "local"


def get_total_nodes(partition=None):
    """
    Get total nodes

    Parameters
    ----------
    partitiion : str, optional
        Partition or queue (depending on type of scheduler)

    Returns
    -------
    int
        Total node number
    """
    if partition is None:
        print("No partition is given. Providing nodes of entire cluster.")
    scheduler_name = get_scheduler_name()
    if scheduler_name == "slurm":
        if partition is None:
            cmd = "sinfo -h -o '%D'"
        else:
            cmd = f"sinfo -p {partition} -h -o '%D'"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split()
        return sum(int(x) for x in output)
    elif scheduler_name == "pbs":
        if partition is None:
            cmd = "pbsnodes -a | grep 'Mom =' | wc -l"
        else:
            cmd = f"pbsnodes -a | grep 'queue = {partition}' | wc -l"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(output)
    elif scheduler_name == "lsf":
        cmd = "bhosts -noheader | wc -l"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(output)
    elif scheduler_name == "sge":
        cmd = "qhost | grep lx | wc -l"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(output)
    elif scheduler_name == "htcondor":
        cmd = "condor_status -noheader | wc -l"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(output)
    elif scheduler_name == "oar":
        if partition:
            cmd = f"oarnodes -l | grep {partition} | wc -l"
        else:
            cmd = "oarnodes -s | grep Alive | wc -l"

        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(output)
    elif scheduler_name == "moab":
        cmd = "mdiag -n"
        output = subprocess.check_output(cmd, shell=True).decode()
        for line in output.splitlines():
            if "Total Nodes" in line:
                return int(line.split(":")[1].strip())
        return
    elif scheduler_name == "local":
        return 1
    else:
        return None
