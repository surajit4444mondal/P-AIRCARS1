import logging
import psutil
import argparse
import requests
import sys
import os
import getpass
import time
from datetime import datetime as dt
from parfive import Downloader
from paircars.utils.basic_utils import (
    create_datadir,
    get_datadir,
    get_cachedir,
    check_port_status,
    get_free_port,
)
from paircars.utils.logger_utils import SmartDefaultsHelpFormatter, generate_password
from paircars.utils.prefect_setup_utils import start_prefect_server, stop_prefect_server
from paircars.utils.resource_utils import has_space
from paircars.utils.proc_manage_utils import get_scheduler_name
from paircars.utils.udocker_utils import (
    init_udocker,
    initialize_wsclean_container,
    initialize_quartical_container,
    initialize_shadems_container,
    initialize_hyperdrive_container,
    initialize_hyperbeam_container,
    initialize_postgres_container,
)
from paircars.utils.killjob_utils import kill_port
from paircars.pipeline.beam_interpolate import do_beam_interpolate

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)

all_filenames = [
    "udocker-englib-1.2.11.tar.gz",
    "de440s.bsp",
    "GGSM.txt",
    "haslam_map.fits",
    "MWA_sweet_spots.npy",
    "Ref_mean_bandpass_final.npy",
    "postgres_credentials.npy",
    "mwa_full_embedded_element_pattern.h5",
]


def get_zenodo_file_urls(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return [(f["links"]["self"], f["key"]) for f in data.get("files", [])]


def download_with_parfive(record_id, update=False, output_dir="zenodo_download"):
    print("####################################")
    print("Downloading P-AIRCARS data files ...")
    print("####################################")
    urls = get_zenodo_file_urls(record_id)
    urls.append(
        (
            "http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5",
            "mwa_full_embedded_element_pattern.h5",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    total_cpu = psutil.cpu_count()
    dl = Downloader(max_conn=min(total_cpu, len(all_filenames) + 1))
    for file_url, filename in urls:
        if filename in all_filenames:
            if not os.path.exists(f"{output_dir}/{filename}") or update:
                if os.path.exists(f"{output_dir}/{filename}"):
                    os.system(f"rm -rf {output_dir}/{filename}")
                dl.enqueue_file(file_url, path=output_dir, filename=filename)
    results = dl.download()
    retry_count = 0
    while True:
        if len(results.errors) > 0:
            dl.retry(results)
            retry_count += 1
            time.sleep(10)
        else:
            break
        if retry_count > 10:
            print(
                "Maximum 10 times retied to download. Check your internet connection."
            )
            break
    for f in results:
        os.chmod(f, 0o755)


def init_paircars_data(
    update=False, remote_link=None, remotelink_password=None, emails=None
):
    """
    Initiate P-AIRCARS data

    Parameters
    ----------
    update : bool, optional
        Update data, if already exists
    remote_link : str, optional
        Remote logger link to save in database
    remotelink_password : str, optional
        Remote link password
    emails : str, optional
        Email addresses to send remote logger JobID and password
    """
    datadir = get_datadir()
    os.makedirs(datadir, exist_ok=True)
    cachedir = get_cachedir()
    username = getpass.getuser()
    linkfile = f"{cachedir}/.remotelink_{username}.txt"
    linkpassword = f"{cachedir}/.remotelink_password_{username}.txt"
    emailfile = f"{cachedir}/.emails_{username}.txt"
    if not os.path.exists(linkfile):
        with open(linkfile, "w") as f:
            f.write("")

    if remote_link is not None:
        with open(linkfile, "w") as f:
            f.write(str(remote_link))

        if remotelink_password is None:
            remotelink_password = generate_password()

        with open(linkpassword, "w") as f:
            f.write(str(remotelink_password))

    if emails is not None:
        with open(emailfile, "w") as f:
            f.write(str(emails))

    unavailable_files = [
        f for f in all_filenames if not os.path.exists(f"{datadir}/{f}")
    ]

    if unavailable_files or update:
        record_id = "18640418"
        download_with_parfive(record_id, update=update, output_dir=datadir)
        timestr = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"P-AIRCARS data are updated in: {datadir} at time: {timestr}")

    freqres_list = [40, 80, 160, 320, 640]
    file_size_list = [6520103392, 3223921432, 1575254952, 750739184, 338476000]
    mwapb_file = f"{datadir}/mwa_full_embedded_element_pattern.h5"
    for i in range(len(freqres_list)):
        freqres = freqres_list[i]
        file_size = file_size_list[i]
        outfile = mwapb_file.split(".h5")[0] + f"_{freqres}.h5"
        if os.path.exists(outfile) is False or update:
            print(f"Making interpolated beam at frequency resolution: {freqres} kHz")
            do_beam_interpolate(
                mwapb_file, new_freq_res=int(freqres), expected_file_size=file_size
            )


def main(
    init=False,
    port=4260,
    do_kill_port=True,
    datadir="",
    update=False,
    link=None,
    password=None,
    emails=None,
):
    """
    Initiate P-AIRCARS setup

    Parameters
    ----------
    init : bool, optional
        Initiate setup
    port : int, optional
        Prefect port
    do_kill_port : bool, optional
        Try to kill port job if it is occupied
    datadir : str, optional
        User provided custom data directory
    update : bool, optional
        Update existing data (if corrupted by somehow)
    link : str, optional
        Remote link
    password : str, optional
        Remote logger password
    emails : str, optional
        E-mails for notifications
    """
    required_gb = 20
    postgres_port = port + 1000
    scheduler_name = get_scheduler_name()

    if check_port_status(port) is False:
        if do_kill_port:
            try:
                kill_port(int(port))
                check_free_port = False
            except Exception:
                check_free_port = True
        else:
            check_free_port = True
        if scheduler_name != "local" and check_free_port:
            port = get_free_port(start_port=port, end_port=port + 990)

    if check_port_status(postgres_port) is False:
        if do_kill_port:
            try:
                kill_port(int(postgres_port))
                check_free_port = False
            except Exception:
                check_free_port = True
        else:
            check_free_port = True
        if scheduler_name != "local" and check_free_port:
            postgres_port = get_free_port(
                start_port=postgres_port, end_port=postgres_port + 990
            )

    if init:
        ######################################
        # Downloading data files
        ######################################
        create_datadir(datadir=datadir)
        datadir = get_datadir()
        print(f"P-AIRCARS configuration data directory: {datadir}")
        if has_space(datadir, required_gb) is False:
            print(
                f"Minimum {required_gb}GB disk space is required in data directory: {datadir}. Please check disk space."
            )
            return 1
        init_paircars_data(
            update=update, remote_link=link, remotelink_password=password, emails=emails
        )
        print("P-AIRCARS data are initiated.")

        #########################################
        # Docker containers initiation
        #########################################
        init_udocker()
        print("uDOCKER is inititalized")

        #####################
        # Wsclean
        #####################
        trial = 0
        while trial < 2:
            wsclean_container_name = initialize_wsclean_container(
                update=update, verbose=True
            )
            if (
                wsclean_container_name is not None
                and wsclean_container_name == "paircarswsclean"
            ):
                print("WSClean container is initialized")
                break
            else:
                trial += 1
                if trial == 2:
                    print("WSClean container is not initialized.")
                    print("Check you internet connectivity.")
                    return 1

        ##########################
        # Quartical
        ##########################
        trial = 0
        while trial < 2:
            quartical_container_name = initialize_quartical_container(
                update=update, verbose=True
            )
            if (
                quartical_container_name is not None
                and quartical_container_name == "paircarsquartical"
            ):
                print("Quartical container is initialized")
                break
            else:
                trial += 1
                if trial == 2:
                    print("Quartical container is not initialized.")
                    print("Check you internet connectivity.")
                    return 1

        #############################
        # Hyperdrive
        #############################
        trial = 0
        while trial < 2:
            hyperdrive_container_name = initialize_hyperdrive_container(
                update=update, verbose=True
            )
            if (
                hyperdrive_container_name is not None
                and hyperdrive_container_name == "paircarshyperdrive"
            ):
                print("Hyperdrive container is initialized")
                break
            else:
                trial += 1
                if trial == 2:
                    print("Hyperdrive container is not initialized.")
                    print("Check you internet connectivity.")
                    return 1

        #############################
        # Hyperbeam
        #############################
        trial = 0
        while trial < 2:
            hyperbeam_container_name = initialize_hyperbeam_container(
                update=update, verbose=True
            )
            if (
                hyperbeam_container_name is not None
                and hyperbeam_container_name == "paircarshyperbeam"
            ):
                print("Hyperbeam container is initialized")
                break
            else:
                trial += 1
                if trial == 2:
                    print("Hyperbeam container is not initialized.")
                    print("Check you internet connectivity.")
                    return 1

        #############################
        # PostgreSQL
        ##############################
        trial = 0
        while trial < 2:
            postgres_container_name = initialize_postgres_container(
                update=update, verbose=True
            )
            if (
                postgres_container_name is not None
                and postgres_container_name == "paircarspostgres"
            ):
                print("PostgreSQL container is initialized")
                break
            else:
                trial += 1
                if trial == 2:
                    print("PostgreSQL container is not initialized.")
                    print("Check you internet connectivity.")
                    return 1

        ##############################
        # Shadems
        ##############################
        trial = 0
        while trial < 2:
            shadems_container_name = initialize_shadems_container(
                update=update, verbose=True
            )
            if (
                shadems_container_name is not None
                and shadems_container_name == "paircarsshadems"
            ):
                print("Shadems container is initialized")
                break
            else:
                trial += 1
                if trial == 2:
                    print("Shadems container is not initialized.")
                    print("Check you internet connectivity.")

        #########################################
        # prefect server setup
        #########################################
        print("Prefect setup....")
        msg, config_file, profile_path, env_file, dashboard, pid_file = (
            start_prefect_server(port, postgres_port, scheduler_name=scheduler_name)
        )
        if msg != 0:
            if scheduler_name != "local":
                print(
                    f"P-AIRCARS will not work in prefect ephemeral mode in cluster environment with job scheduler: {scheduler_name}"
                )
                return 1
            else:
                print(
                    "Error in starting prefect server at port. P-AIRCARS will use ephemeral mode in local cluster."
                )
                try:
                    stop_prefect_server(scheduler_name=scheduler_name)
                except Exception:
                    pass
                os.system(f"rm -rf {config_file} {profile_path} {env_file} {pid_file}")
        return 0
    else:
        return 1


def cli():
    usage = "Initiate P-AIRCARS data"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )
    parser.add_argument("--init", action="store_true", help="Initiate data")
    parser.add_argument("--port", type=int, default=4260, help="Prefect port")
    parser.add_argument(
        "--no_kill_port",
        action="store_false",
        dest="kill_port",
        help="Do not kill occupied port",
    )
    parser.add_argument(
        "--configdir",
        type=str,
        default="",
        help="User provided custom configuration data directory",
    )
    parser.add_argument("--update", action="store_true", help="Update existing data")
    parser.add_argument(
        "--remotelink", dest="link", default=None, help="Set remote log link"
    )
    parser.add_argument(
        "--remote_password",
        dest="password",
        default=None,
        help="Set remote log password",
    )
    parser.add_argument(
        "--emails",
        dest="emails",
        default=None,
        help="Email addresses (comma separated) to send Job ID and password for remote logger",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        init=args.init,
        datadir=args.configdir,
        port=args.port,
        do_kill_port=args.kill_port,
        update=args.update,
        link=args.link,
        password=args.password,
        emails=args.emails,
    )
    if msg != 0:
        print("Error in initial setup.")
    return msg


if __name__ == "__main__":
    msg = cli()
    os._exit(msg)
