import os
import subprocess
import time
import socket
import signal
import toml
import traceback
import numpy as np
import getpass
from dotenv import load_dotenv
from .basic_utils import get_datadir
from .killjob_utils import kill_port
from .udocker_utils import run_postgres, kill_postgres


# === CONFIG ===
username = getpass.getuser()


def prefect_config(port, postgres_port, scheduler_name="local"):
    """
    Configure prefect

    Parameters
    ----------
    port : int
        Free port
    postgres_port : int
        Postgres server  port
    scheduler_name : str, optional
        Scheduler name

    Returns
    -------
    str
        Configuration file name
    dict
        Configuration dictionary
    """
    ##################################
    # Postgres information
    ##################################
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    postgres_url_file = f"{datadir}/postgres.url"
    if os.path.exists(postgres_url_file) is False:
        print("Start postgres server first.")
        return

    ###################################
    # Prefect configuration
    ###################################
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)

    config_file = f"{cachedir}/prefect.config.npy"

    PREFECT_HOME = f"{cachedir}/prefect_home"
    os.makedirs(PREFECT_HOME, exist_ok=True)

    with open(postgres_url_file, "r") as f:
        postgres_url = f.read().strip()
    PREFECT_API_DATABASE_CONNECTION_URL = postgres_url

    LOG_FILE = os.path.join(PREFECT_HOME, "server.log")
    profile_path = os.path.join(PREFECT_HOME, "profiles.toml")
    memo_path = os.path.join(PREFECT_HOME, "memo_store.toml")
    storage = os.path.join(PREFECT_HOME, "storage")
    os.makedirs(storage, exist_ok=True)

    ENV_FILE = os.path.join(cachedir, "paircars_prefect.env")

    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = f"{port}"

    hostname = socket.gethostname()

    SERVER_URL = f"http://0.0.0.0:{SERVER_PORT}/api"
    NODE_URL = f"http://{hostname}:{SERVER_PORT}/api"

    SERVER_DASHBOARD = f"http://localhost:{SERVER_PORT}/dashboard"

    PREFECT_SERVER_API_HOST = "127.0.0.1"
    profile_name = f"paircarspipe_{scheduler_name}"
    pid_file = os.path.join(PREFECT_HOME, "server.pid")
    logging_path = os.path.join(PREFECT_HOME, "logging.yml")

    config = {
        "CACHEDIR": cachedir,
        "PREFECT_HOME": PREFECT_HOME,
        "PREFECT_API_DATABASE_CONNECTION_URL": PREFECT_API_DATABASE_CONNECTION_URL,
        "LOG_FILE": LOG_FILE,
        "PROFILE_PATH": profile_path,
        "MEMO_PATH": memo_path,
        "STORAGE": storage,
        "ENV_FILE": ENV_FILE,
        "SERVER_HOST": SERVER_HOST,
        "SERVER_PORT": SERVER_PORT,
        "SERVER_URL": SERVER_URL,
        "NODE_URL": NODE_URL,
        "SERVER_DASHBOARD": SERVER_DASHBOARD,
        "PROFILE_NAME": profile_name,
        "PID_FILE": pid_file,
        "LOGGING_PATH": logging_path,
        "PREFECT_SERVER_API_HOST": PREFECT_SERVER_API_HOST,
    }
    np.save(config_file, config)
    return config_file, config


####################################
# Start and save
####################################
def write_prefect_profile(scheduler_name="local"):
    """
    Save prefect profile

    Parameters
    ----------
    scheduler_name : str, optional
        Scheduler name

    Returns
    -------
    str
        Profile file
    """
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)
    config_file = f"{cachedir}/prefect.config.npy"
    if os.path.exists(config_file) is False:
        print(f"Configuration file for job ID: {scheduler_name} does not exist.")
        return False
    config = np.load(config_file, allow_pickle=True).all()
    # Load existing TOML config or start new
    profile_path = config["PROFILE_PATH"]
    if os.path.exists(profile_path):
        data = toml.load(profile_path)
    else:
        data = {}
    # Set active profile
    profile_name = config["PROFILE_NAME"]
    data["active"] = profile_name
    # Set config under [profiles.<profile_name>]
    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][profile_name] = {
        "PREFECT_API_URL": config["SERVER_URL"],
        "PREFECT_HOME": config["PREFECT_HOME"],
        "PREFECT_API_DATABASE_CONNECTION_URL": config[
            "PREFECT_API_DATABASE_CONNECTION_URL"
        ],
    }
    with open(profile_path, "w") as f:
        toml.dump(data, f)
    print(f"Prefect profile '{profile_name}' written to {profile_path}")
    return profile_path


def save_prefect_env_to_file(scheduler_name="local"):
    """
    Save current Prefect server env config to a .env file for reuse.

    Parameters
    ----------
    scheduler_name : str, optional
        Scheduler name

    Returns
    -------
    str
        Profile file
    str
        Environment file
    str
        Dashboard file
    """
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)
    config_file = f"{cachedir}/prefect.config.npy"
    if os.path.exists(config_file) is False:
        print(f"Configuration file for job ID: {scheduler_name} does not exist.")
        return False
    config = np.load(config_file, allow_pickle=True).all()
    cachedir = config["CACHEDIR"]
    env_file = config["ENV_FILE"]
    dashboard = f"{cachedir}/prefect.dashboard"
    with open(env_file, "w") as f:
        f.write(f"PREFECT_HOME={config['PREFECT_HOME']}\n")
        f.write("PREFECT_API_MODE=server\n")
        f.write(
            f"PREFECT_API_DATABASE_CONNECTION_URL={config['PREFECT_API_DATABASE_CONNECTION_URL']}\n"
        )
        f.write("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=false\n")
        f.write(f"PREFECT_API_URL={config['SERVER_URL']}\n")
        f.write(f"PREFECT_PROFILE={config['PROFILE_NAME']}\n")
        f.write(f"PREFECT_PROFILES_PATH={config['PROFILE_PATH']}\n")
        f.write(f"PREFECT_LOCAL_STORAGE_PATH={config['STORAGE']}\n")
        f.write(f"PREFECT_LOGGING_SETTINGS_PATH={config['LOGGING_PATH']}\n")
        f.write(f"PREFECT_MEMO_STORE_PATH={config['MEMO_PATH']}\n")
    print(f"Saved Prefect server environment to {env_file}")
    if os.path.exists(dashboard) is not True:
        with open(dashboard, "w") as f:
            f.write(f"{config['SERVER_DASHBOARD']}")
    profile_path = write_prefect_profile(scheduler_name=scheduler_name)
    return profile_path, env_file, dashboard


def start_prefect_server(
    port, postgres_port, show_config=False, scheduler_name="local"
):
    """
    Start prefect server if it is not running

    Parameters
    ----------
    port : int
        Free port number
    postgres_port : int
        Postgres port
    show_config : bool, optional
        Show configuration of prefect server
    scheduler_name : str, optional
        Scheduler name

    Returns
    -------
    0, config_file, profile_path, env_file, dashboard, pid_file
    int
        Success message
    str
        Configuration file
    str
        Profile file
    str
        Environment file
    str
        Dashboard file
    str
        Server process ID file
    """
    kill_port(int(port))
    trial = 0
    while trial < 5:
        running = run_postgres(
            postgres_port=postgres_port,
            verbose=True,
        )
        if running:
            break
        else:
            trial += 1
    config_file, config = prefect_config(
        port, postgres_port, scheduler_name=scheduler_name
    )
    pid_file = config["PID_FILE"]
    env = get_prefect_env(scheduler_name=scheduler_name)
    print("Starting Prefect server...")
    if prefect_server_status(scheduler_name=scheduler_name):
        print(f"Server is already running at port: {port}")
        server_started = True
        new_start = False
    else:
        with open(config["LOG_FILE"], "w") as f:
            server_proc = subprocess.Popen(
                [
                    "prefect",
                    "server",
                    "start",
                    "--host",
                    config["SERVER_HOST"],
                    "--port",
                    config["SERVER_PORT"],
                    "--log-level",
                    "INFO",
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        server_started = False
        new_start = True
    os.makedirs(config["PREFECT_HOME"], exist_ok=True)
    profile_path, env_file, dashboard = save_prefect_env_to_file(
        scheduler_name=scheduler_name
    )
    if server_started is False:
        for _ in range(180):  # wait up to 180s for the server to respond
            if prefect_server_status(scheduler_name=scheduler_name):
                server_started = True
                break
            else:
                time.sleep(5)
    if server_started:
        if new_start:
            with open(pid_file, "w") as pf:
                pf.write(str(server_proc.pid))
        print(
            "##########################################################################"
        )
        if scheduler_name != "local":
            print(
                f"First tunnel to prefect from your local machine: ssh -N -L {port}:localhost:{port} <username>@<remote.cluster.name>"
            )
        print(
            f"Prefect server dashboard for monitoring is available at local machine: http://localhost:{port}/dashboard"
        )
        print(
            "##########################################################################"
        )
        if os.path.exists(dashboard) is not True:
            with open(dashboard, "w") as f:
                f.write(f"{config['SERVER_DASHBOARD']}")
        if show_config:
            show_prefect_config(scheduler_name=scheduler_name)
        return 0, config_file, profile_path, env_file, dashboard, pid_file
    else:
        print(
            f"Server did not respond within 3 minutes. Check logs at {config['LOG_FILE']} for more details"
        )
        return 0, config_file, profile_path, env_file, dashboard, pid_file


#########################################
# Stop prefect server
##########################################
def stop_prefect_server(scheduler_name="local"):
    """
    Stop prefect server running in the current installation
    Note: it will only stop prefect server which is running from the current installation
    For this pipeline, a port between 4260 to 5250 is chosen for P-AIRCARS.

    Parameters
    ----------
    scheduler_name : str, optional
        Scheduler name

    Returns
    -------
    int
        Success message
    """
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)
    config_file = f"{cachedir}/prefect.config.npy"
    if os.path.exists(config_file) is False:
        print(f"Configuration file for job ID: {scheduler_name} does not exist.")
        os.system(f"rm -rf {cachedir}")
        return 1
    config = np.load(config_file, allow_pickle=True).all()
    config = np.load(config_file, allow_pickle=True).all()
    load_dotenv(dotenv_path=config["ENV_FILE"], override=True)
    env = os.environ.copy()
    print("Prefect config in current environment ...")
    result = subprocess.run(
        ["prefect", "server", "stop"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    exit_code = result.returncode
    if exit_code == 0:
        print("Prefect server stopped.")
    else:
        print("Could not stop prefect server.")
    postgres_port = int(
        config["PREFECT_API_DATABASE_CONNECTION_URL"].split(":")[-1].split("/")[0]
    )
    pid_file = config["PID_FILE"]
    cachedir = config["CACHEDIR"]
    try:
        if os.path.exists(pid_file):
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            print(f"Stopping Prefect server with PID {pid} ...")
            os.kill(pid, signal.SIGTERM)
        try:
            print(f"Closing prefect server port: {config['SERVER_PORT']}")
            kill_port(int(config["SERVER_PORT"]))
            msg = 0
            print(f"Closed port: {config['SERVER_PORT']}")
        except Exception:
            traceback.print_exc()
            msg = 1
            print(f"Could not close prefect server port: {config['SERVER_PORT']}")
        try:
            print(f"Closing postgreSQL server port: {postgres_port}")
            killed = kill_postgres(
                postgres_port=postgres_port,
            )
            if not killed:
                kill_port(int(postgres_port))
            print(f"Closed port: {postgres_port}")
        except Exception:
            traceback.print_exc()
            print(f"Could not close postgreSQL server port: {postgres_port}")
        print(f"Server stopped and {cachedir} removed.")
        msg = 0
    except ProcessLookupError:
        print(f"No such process with PID {pid}. Removing stale {cachedir} directory.")
        msg = 0
    except Exception:
        print("Error stopping server")
        traceback.print_exc()
        msg = 1
    finally:
        os.system(f"rm -rf {cachedir}")
        return msg


############################################
# Prefect server status
############################################
def prefect_server_status(scheduler_name="local"):
    """
    Get prefect server status

    Parameters
    ----------
    scheduler_name : str, optional
        Scheduler name
    """
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)
    config_file = f"{cachedir}/prefect.config.npy"
    if os.path.exists(config_file) is False:
        print(f"Configuration file for job ID: {scheduler_name} does not exist.")
        return False
    config = np.load(config_file, allow_pickle=True).all()
    try:
        with socket.create_connection(
            (config["SERVER_HOST"], config["SERVER_PORT"]), timeout=60
        ):
            return True
    except OSError:
        return False


def get_prefect_env(scheduler_name="local"):
    """
    Get environment variables of prefect

    Parameters
    ----------
    scheduler_name : str, optional
        Scheduler name

    Returns
    -------
    dict
        Environment dictionary
    """
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)
    config_file = f"{cachedir}/prefect.config.npy"
    if os.path.exists(config_file) is False:
        print(f"Configuration file for job ID: {scheduler_name} does not exist.")
        return
    config = np.load(config_file, allow_pickle=True).all()
    env = os.environ.copy()
    env["PREFECT_HOME"] = config["PREFECT_HOME"]
    env["PREFECT_API_MODE"] = "server"
    env["PREFECT_API_DATABASE_CONNECTION_URL"] = config[
        "PREFECT_API_DATABASE_CONNECTION_URL"
    ]
    env["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "false"
    env["PREFECT_API_URL"] = config["SERVER_URL"]
    env["PREFECT_PROFILE"] = config["PROFILE_NAME"]
    env["PREFECT_PROFILES_PATH"] = config["PROFILE_PATH"]
    env["PREFECT_LOCAL_STORAGE_PATH"] = config["STORAGE"]
    env["PREFECT_LOGGING_SETTINGS_PATH"] = config["LOGGING_PATH"]
    env["PREFECT_MEMO_STORE_PATH"] = config["MEMO_PATH"]
    return env


def show_prefect_config(scheduler_name="local"):
    """
    Print the effective Prefect config in this environment.

    Parameters
    ----------
    scheduler_name : str, optional
        Scheduler name
    """
    datadir = f"{get_datadir()}/{username}"
    os.makedirs(datadir, exist_ok=True)
    cachedir = f"{datadir}/prefect_{scheduler_name}"
    os.makedirs(cachedir, exist_ok=True)
    config_file = f"{cachedir}/prefect.config.npy"
    if os.path.exists(config_file) is False:
        print(f"Configuration file for job ID: {scheduler_name} does not exist.")
        return
    config = np.load(config_file, allow_pickle=True).all()
    load_dotenv(dotenv_path=config["ENV_FILE"], override=True)
    env = os.environ.copy()
    print("Prefect config in current environment ...")
    subprocess.run(["prefect", "config", "view"], env=env)
