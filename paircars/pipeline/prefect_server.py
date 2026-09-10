import argparse
import sys
import os
import numpy as np
import getpass
from paircars.utils.basic_utils import get_datadir, check_port_status, get_free_port
from paircars.utils.logger_utils import SmartDefaultsHelpFormatter
from paircars.utils.proc_manage_utils import get_scheduler_name
from paircars.utils.prefect_setup_utils import (
    start_prefect_server,
    stop_prefect_server,
    prefect_server_status,
    save_prefect_env_to_file,
    show_prefect_config,
)


def cli():
    parser = argparse.ArgumentParser(
        description="Manage a local Prefect server. Only for single-node work station.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    # Start
    start_parser = subparsers.add_parser("start", help="Start the Prefect server")
    start_parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display Prefect config after startup",
    )
    # Stop
    subparsers.add_parser("stop", help="Stop the Prefect server")

    # Status
    subparsers.add_parser("status", help="Check if the Prefect server is running")

    # Env
    subparsers.add_parser(
        "save_env", help="Save the Prefect environment to a .env file"
    )

    # Config
    subparsers.add_parser("config", help="Print the current Prefect config")

    parser.add_argument("--port", type=int, default=4260, help="Prefect port")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    scheduler_name = get_scheduler_name()

    port = int(args.port)
    postgres_port = port + 1000

    if check_port_status(port) is False:
        if scheduler_name != "local":
            port = get_free_port(start_port=port, end_port=port + 990)

    if check_port_status(postgres_port) is False:
        if scheduler_name != "local":
            get_free_port(start_port=postgres_port, end_port=postgres_port + 990)

    if args.command == "start":
        msg, config_file, profile_path, env_file, dashboard, pid_file = (
            start_prefect_server(
                port,
                postgres_port,
                show_config=args.show_config,
                scheduler_name=scheduler_name,
            )
        )
        if msg == 0:
            print(f"Prefect server started at port: {port}")
            print(f"Profile file: {profile_path}")
            print(f"Environment file: {env_file}")
            print(f"Dashboard file: {dashboard}")
            print(f"Server process ID file: {pid_file}")
        else:
            print(f"Error in starting prefect server at port: {port}")
    elif args.command == "save_env":
        profile_path, env_file, dashboard = save_prefect_env_to_file(
            scheduler_name=scheduler_name
        )
        print(f"Profile file: {profile_path}")
        print(f"Environment file: {env_file}")
        print(f"Dashboard file: {dashboard}")
    elif args.command == "config":
        show_prefect_config(scheduler_name=scheduler_name)
    else:
        username = getpass.getuser()
        datadir = f"{get_datadir()}/{username}"
        os.makedirs(datadir, exist_ok=True)
        cachedir = f"{datadir}/prefect_{scheduler_name}"
        os.makedirs(cachedir, exist_ok=True)
        config_file = f"{cachedir}/prefect.config.npy"
        if os.path.exists(config_file) is False:
            print(f"Configuration file for job ID: {scheduler_name} does not exist.")
            return 1
        config = np.load(config_file, allow_pickle=True).all()
        if args.command == "stop":
            stop_msg = stop_prefect_server(scheduler_name=scheduler_name)
            if stop_msg == 0:
                print(f"Prefect server stopped at: {config['SERVER_DASHBOARD']}.")
            else:
                print(
                    f"Error in stopping prefect server at: {config['SERVER_DASHBOARD']}."
                )
        elif args.command == "status":
            if prefect_server_status(scheduler_name=scheduler_name):
                print(
                    "##########################################################################"
                )
                if scheduler_name != "local":
                    print(
                        f"First tunnel to prefect from your local machine: ssh -N -L {config['SERVER_PORT']}:localhost:{config['SERVER_PORT']} <username>@<remote.cluster.name>"
                    )
                    print(
                        f"Prefect server dashboard for remote monitoring is available at local machine: http://localhost:{config['SERVER_PORT']}/dashboard"
                    )
                else:
                    print(
                        f"Prefect server dashboard for monitoring is available at: http://localhost:{config['SERVER_PORT']}/dashboard"
                    )
                print(
                    "##########################################################################"
                )
            else:
                print(
                    f"Prefect server is not running at: {config['SERVER_DASHBOARD']}."
                )
        else:
            parser.print_help()


if __name__ == "__main__":
    cli()
