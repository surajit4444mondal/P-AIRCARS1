import traceback
import tempfile
import os
import subprocess
import numpy as np
import socket
from .basic_utils import get_datadir, wait_for_port
from .killjob_utils import terminate_process_and_children, kill_port
from .resource_utils import limit_threads

####################
# uDOCKER related
####################


def set_udocker_env():
    datadir = get_datadir()
    if (
        datadir is None
        or os.path.exists(datadir) is False
        or os.path.exists(f"{datadir}/udocker-englib-1.2.11.tar.gz") is False
    ):
        print("P-AIRCARS data directory and docker environment is not setup yet")
        return
    udocker_dir = f"{datadir}/udocker"
    os.makedirs(udocker_dir, exist_ok=True)
    os.environ["UDOCKER_DIR"] = udocker_dir
    os.environ["UDOCKER_TARBALL"] = f"{datadir}/udocker-englib-1.2.11.tar.gz"
    return datadir


def init_udocker():
    set_udocker_env()
    os.system("udocker install")


def check_udocker_container(name):
    """
    Check whether a docker container is present or not

    Parameters
    ----------
    name : str
        Container name

    Returns
    -------
    bool
        Whether present or not
    """
    init_udocker()
    env = os.environ.copy()
    try:
        result = subprocess.run(
            ["udocker", "--insecure", "--quiet", "inspect", name],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception:
        return False


def initialize_container(image_name, name, update=False, verbose=False):
    """
    Initialize container

    Parameters
    ----------
    image_name: str
        Docker image name
    name : str
        Container name
    update : bool, optional
        Update or not
    verbose : bool, optional
        Verbose output

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    init_udocker()
    env = os.environ.copy()
    check_cmd = f"udocker images | grep -q {image_name}"
    image_exists = os.system(check_cmd)
    container_exists = check_udocker_container(name)
    if image_exists != 0:
        if container_exists:
            if verbose:
                subprocess.run(
                    ["udocker", "rm", f"{name}"],
                    env=env,
                )
            else:
                subprocess.run(
                    ["udocker", "rm", f"{name}"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        if verbose:
            result = subprocess.run(
                ["udocker", "pull", f"{image_name}"],
                env=env,
            )
        else:
            result = subprocess.run(
                ["udocker", "pull", f"{image_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        a = result.returncode
        if a == 0:
            create_container = True
        else:
            create_container = False
    else:
        if update:
            if verbose:
                if container_exists:
                    subprocess.run(
                        ["udocker", "rm", f"{name}"],
                        env=env,
                    )
                subprocess.run(
                    ["udocker", "rmi", f"{image_name}"],
                    env=env,
                )
            else:
                if container_exists:
                    subprocess.run(
                        ["udocker", "rm", f"{name}"],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                subprocess.run(
                    ["udocker", "rmi", f"{image_name}"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            print("Re-downloading docker image.")
            if verbose:
                result = subprocess.run(
                    ["udocker", "pull", f"{image_name}"],
                    env=env,
                )
            else:
                result = subprocess.run(
                    ["udocker", "pull", f"{image_name}"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            a = result.returncode
            if a == 0:
                print("Re-downloaded docker image.")
                create_container = True
            else:
                print("Re-downloading container image is failed.")
                create_container = False
                return
        else:
            print(f"Image {image_name} already present.")
            if container_exists is False:
                create_container = True
            else:
                return name
    if create_container:
        if verbose:
            result = subprocess.run(
                ["udocker", "create", f"--name={name}", f"{image_name}"],
                env=env,
            )
        else:
            result = subprocess.run(
                ["udocker", "create", f"--name={name}", f"{image_name}"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        a = result.returncode
        print(f"Container started with name : {name}")
        return name
    else:
        print(f"Container could not be created with name : {name}")
        return


def initialize_wsclean_container(name="paircarswsclean", update=False, verbose=False):
    """
    Initialize WSClean container

    Parameters
    ----------
    name : str, optional
        Name of the container
    update : bool, optional
        Update container

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    print("Initializing wsclean container.")
    image_name = "devojyoti96/wsclean-solar:latest"
    msg = initialize_container(image_name, name, update=update, verbose=verbose)
    return msg


def initialize_quartical_container(
    name="paircarsquartical", update=False, verbose=True
):
    """
    Initialize quartical container

    Parameters
    ----------
    name : str, optional
        Name of the container
    update : bool, optional
        Update container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    print("Initializing quartical container.")
    image_name = "devojyoti96/quartical:0.2.6"
    msg = initialize_container(image_name, name, update=update, verbose=verbose)
    return msg


def initialize_shadems_container(name="paircarsshadems", update=False, verbose=False):
    """
    Initialize shadems container

    Parameters
    ----------
    name : str, optional
        Name of the container
    update : bool, optional
        Update container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    print("Initializing shadems container.")
    image_name = "devojyoti96/shadems:v0.5.4"
    msg = initialize_container(image_name, name, update=update, verbose=verbose)
    return msg


def initialize_hyperdrive_container(
    name="paircarshyperdrive", update=False, verbose=False
):
    """
    Initialize hyperdrive container

    Parameters
    ----------
    name : str, optional
        Name of the container
    update : bool, optional
        Update container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    print("Initializing hyperdrive container.")
    image_name = "devojyoti96/paircarshyperdrive:latest"
    msg = initialize_container(image_name, name, update=update, verbose=verbose)
    return msg


def initialize_hyperbeam_container(
    name="paircarshyperbeam", update=False, verbose=False
):
    """
    Initialize hyperbeam container

    Parameters
    ----------
    name : str, optional
        Name of the container
    update : bool, optional
        Update container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    print("Initializing hyperbeam container.")
    image_name = "devojyoti96/paircarshyperbeam:latest"
    msg = initialize_container(image_name, name, update=update, verbose=verbose)
    return msg


def initialize_postgres_container(name="paircarspostgres", update=False, verbose=False):
    """
    Initialize postgres container

    Parameters
    ----------
    name : str, optional
        Name of the container
    update : bool, optional
        Update container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    print("Initializing postgres container.")
    image_name = "postgres"
    msg = initialize_container(image_name, name, update=update, verbose=verbose)
    return msg


def run_wsclean(
    wsclean_cmd,
    container_name="paircarswsclean",
    check_container=False,
    verbose=False,
):
    """
    Run WSClean inside a udocker container (no root permission required).

    Parameters
    ----------
    wsclean_cmd : str
        Full WSClean command as a string.
    container_name : str, optional
        Container name
    check_container : bool, optional
        Check container presence or not
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Success message
    """
    init_udocker()
    if check_container:
        container_present = check_udocker_container(container_name)
        if not container_present:
            print(f"Initializing {container_name}...")
            container_name = initialize_wsclean_container(
                name=container_name, verbose=True
            )
            if container_name is None:
                print(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1
    msname = wsclean_cmd.split(" ")[-1]
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_name = "wsclean_udocker_" + next(tempfile._get_candidate_names())
    temp_docker_path = os.path.join(mspath, temp_name)
    wsclean_cmd_args = wsclean_cmd.split(" ")[:-1]
    if "-fits-mask" in wsclean_cmd_args:
        index = wsclean_cmd_args.index("-fits-mask")
        name = wsclean_cmd_args[index + 1]
        os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-name" not in wsclean_cmd_args:
        wsclean_cmd_args.append(
            "-name " + temp_docker_path + "/" + os.path.basename(msname).split(".ms")[0]
        )
    else:
        index = wsclean_cmd_args.index("-name")
        name = wsclean_cmd_args[index + 1]
        os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-temp-dir" not in wsclean_cmd_args:
        wsclean_cmd_args.append("-temp-dir " + temp_docker_path)
    else:
        index = wsclean_cmd_args.index("-temp-dir")
        name = os.path.abspath(wsclean_cmd_args[index + 1])
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path)
    wsclean_cmd = (
        " ".join(wsclean_cmd_args)
        + " "
        + temp_docker_path
        + "/"
        + os.path.basename(msname)
    )
    wsclean_cmd_args = wsclean_cmd.split(" ")
    try:
        full_command = [
            "udocker",
            "run",
            "--nobanner",
            f"--volume={mspath}:{temp_docker_path}",
            "--workdir",
            f"{temp_docker_path}",
            f"{container_name}",
        ] + wsclean_cmd_args
        if verbose:
            print(f"{wsclean_cmd}\n")
            result = subprocess.run(
                full_command,
            )
        else:
            result = subprocess.run(
                full_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        exit_code = result.returncode
        return 0 if exit_code == 0 else 1
    except Exception:
        traceback.print_exc()
        return 1


def run_solar_sidereal_cor(
    msname="",
    only_uvw=False,
    ncpu=1,
    container_name="paircarswsclean",
    check_container=False,
    verbose=False,
):
    """
    Run chgcenter inside a udocker container to correct solar sidereal motion (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator to track the Sun,
        but while creating the MS, UVW values are estimated using the first phasecenter of the Sun.
    ncpu : int, optional
        Number of CPU threads to use
    check_container : bool, optional
        Check container
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Success message
    """
    with limit_threads(n_threads=ncpu):
        env = os.environ.copy()
        env_vars = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
        ]
        init_udocker()
        if check_container:
            container_present = check_udocker_container(container_name)
            if not container_present:
                print(f"Initializing {container_name}...")
                container_name = initialize_wsclean_container(
                    name=container_name, verbose=True
                )
                if container_name is None:
                    print(
                        f"Container {container_name} is not initiated. First initiate container and then run."
                    )
                    return 1
        msname = os.path.abspath(msname)
        mspath = os.path.dirname(msname)
        temp_name = "chgcenter_udocker_" + next(tempfile._get_candidate_names())
        temp_docker_path = os.path.join(mspath, temp_name)
        if only_uvw:
            cmd = (
                "chgcentre -only-uvw -solarcenter "
                + temp_docker_path
                + "/"
                + os.path.basename(msname)
            )
        else:
            cmd = (
                "chgcentre -solarcenter "
                + temp_docker_path
                + "/"
                + os.path.basename(msname)
            )
        cmd_args = cmd.split(" ")
        try:
            full_command = ["udocker", "--quiet", "run", "--nobanner"]
            env_keys = list(env.keys())
            for var in env_vars:
                if var in env_keys:
                    full_command.append(f"--env={var}={env[var]}")
            full_command = (
                full_command
                + [
                    f"--volume={mspath}:{temp_docker_path}",
                    "--workdir",
                    f"{temp_docker_path}",
                    "paircarswsclean",
                ]
                + cmd_args
            )
            if verbose:
                print(f"{cmd}\n")
                result = subprocess.run(
                    full_command,
                )
            else:
                result = subprocess.run(
                    full_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            exit_code = result.returncode
            return 0 if exit_code == 0 else 1
        except Exception:
            traceback.print_exc()
            return 1


def run_chgcenter(
    msname,
    ra,
    dec,
    only_uvw=False,
    ncpu=1,
    container_name="paircarswsclean",
    check_container=False,
    verbose=False,
):
    """
    Run chgcenter inside a udocker container (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    ra : str
        RA can either be 00h00m00.0s or 00:00:00.0
    dec : str
        Dec can either be 00d00m00.0s or 00.00.00.0
    ncpu : bool, optional
        Number of CPU threads to use
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator,
        but while creating the MS, UVW values are estimated using a wrong phase center.
    check_container : bool, optional
        Check container
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    with limit_threads(n_threads=ncpu):
        env = os.environ.copy()
        env_vars = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
        ]
        init_udocker()
        if check_container:
            container_present = check_udocker_container(container_name)
            if not container_present:
                print(f"Initializing {container_name}...")
                container_name = initialize_wsclean_container(
                    name=container_name, verbose=True
                )
                if container_name is None:
                    print(
                        f"Container {container_name} is not initiated. First initiate container and then run."
                    )
                    return 1
        msname = os.path.abspath(msname)
        mspath = os.path.dirname(msname)
        temp_name = "chgcenter_udocker_" + next(tempfile._get_candidate_names())
        temp_docker_path = os.path.join(mspath, temp_name)
        if only_uvw:
            cmd = (
                "chgcentre -only-uvw "
                + temp_docker_path
                + "/"
                + os.path.basename(msname)
                + " "
                + ra
                + " "
                + dec
            )
        else:
            cmd = (
                "chgcentre "
                + temp_docker_path
                + "/"
                + os.path.basename(msname)
                + " "
                + ra
                + " "
                + dec
            )
        cmd_args = cmd.split(" ")
        try:
            full_command = ["udocker", "--quiet", "run", "--nobanner"]
            env_keys = list(env.keys())
            for var in env_vars:
                if var in env_keys:
                    full_command.append(f"--env={var}={env[var]}")
            full_command = (
                full_command
                + [
                    f"--volume={mspath}:{temp_docker_path}",
                    "--workdir",
                    f"{temp_docker_path}",
                    f"{container_name}",
                ]
                + cmd_args
            )
            if verbose:
                print(f"{cmd}\n")
                result = subprocess.run(
                    full_command,
                )
            else:
                result = subprocess.run(
                    full_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            exit_code = result.returncode
            return 0 if exit_code == 0 else 1
        except Exception:
            traceback.print_exc()
            return 1


def run_shadems(
    cmd,
    container_name="paircarsshadems",
    check_container=False,
    verbose=False,
):
    """
    Run shadems inside a udocker container (no root permission required).

    Parameters
    ----------
    cmd : str
        Shadems command
    container_name : str, optional
        Container name
    check_container : bool, optional
        Check container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    init_udocker()
    if check_container:
        container_present = check_udocker_container(container_name)
        if not container_present:
            print(f"Initializing {container_name}...")
            container_name = initialize_shadems_container(
                name=container_name, verbose=True
            )
            if container_name is None:
                print(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1
    splited_cmd = cmd.split(" ")
    if splited_cmd[-1] in ["-h", "--help"]:
        verbose = True
        datapath = None
    else:
        msname = splited_cmd[-1]
        datapath = os.path.dirname(os.path.abspath(msname))
    temp_name = "shadems_udocker_" + next(tempfile._get_candidate_names())
    if datapath is not None:
        temp_docker_path = os.path.join(datapath, temp_name)
    else:
        temp_docker_path = os.path.join(os.getcwd(), temp_name)
    if splited_cmd[-1] not in ["-h", "--help"]:
        cmd = f"{' '.join(splited_cmd[:-1])} {temp_docker_path}/{os.path.basename(msname)}"
    cmd_args = cmd.split(" ")
    try:
        full_command = ["udocker", "--quiet", "run", "--nobanner"]
        if datapath is not None:
            full_command.append(f"--volume={datapath}:{temp_docker_path}")
        full_command += [
            "--workdir",
            f"{temp_docker_path}",
            f"{container_name}",
        ] + cmd_args
        if verbose:
            print(f"{cmd}\n")
            result = subprocess.run(
                full_command,
            )
        else:
            result = subprocess.run(
                full_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        exit_code = result.returncode
        return 0 if exit_code == 0 else 1
    except Exception:
        traceback.print_exc()
        return 1


def run_quartical(
    cmd,
    container_name="paircarsquartical",
    check_container=False,
    verbose=False,
):
    """
    Run quartical inside a udocker container (no root permission required).

    Parameters
    ----------
    cmd : str
        Quartical command
    container_name : str, optional
        Container name
    check_container : bool, optional
        Check container
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    init_udocker()
    if check_container:
        container_present = check_udocker_container(container_name)
        if not container_present:
            print(f"Initializing {container_name}...")
            container_name = initialize_quartical_container(
                name=container_name, verbose=True
            )
            if container_name is None:
                print(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1
    splited_cmd = cmd.split(" ")
    datapath = None
    gain_path = None
    if len(splited_cmd) == 1 and "goquartical" in cmd:
        temp_name = "quartical_udocker_" + next(tempfile._get_candidate_names())
        if datapath is not None:
            temp_docker_path = os.path.join(datapath, temp_name)
        else:
            temp_docker_path = os.path.join(os.getcwd(), temp_name)
    elif len(splited_cmd) > 1:
        for i in range(len(splited_cmd)):
            cmd_arg = splited_cmd[i]
            if "input_ms.path" in cmd_arg:
                msname = cmd_arg.split("input_ms.path=")[-1]
                datapath = os.path.dirname(os.path.abspath(msname))
                temp_name = "quartical_udocker_" + next(tempfile._get_candidate_names())
                temp_docker_path = os.path.join(datapath, temp_name)
                temp_msname = f"{temp_docker_path}/{os.path.basename(msname)}"
                cmd_arg = f"input_ms.path={temp_msname}"
                splited_cmd[i] = cmd_arg
            if "output.gain_directory" in cmd_arg:
                caltable = cmd_arg.split("output.gain_directory=")[-1]
                temp_caltable = f"{temp_docker_path}/{os.path.basename(caltable)}"
                cmd_arg = f"output.gain_directory={temp_caltable}"
                splited_cmd[i] = cmd_arg
            if "output.log_directory" in cmd_arg:
                log = cmd_arg.split("output.log_directory=")[-1]
                temp_log = f"{temp_docker_path}/{os.path.basename(log)}"
                cmd_arg = f"output.log_directory={temp_log}"
                splited_cmd[i] = cmd_arg
            if "load_from" in cmd_arg:
                gaintable = cmd_arg.split("load_from=")[-1]
                gain_path = os.path.dirname(os.path.dirname(gaintable))
                soltype = f"{cmd_arg.split('=')[0].split('.load_from')[0]}"
                gaintable = os.path.basename(os.path.dirname(gaintable))
                temp_name = "quartical_udocker_" + next(tempfile._get_candidate_names())
                temp_gain_path = os.path.join(gain_path, temp_name)
                temp_gaintable = f"{temp_gain_path}/{gaintable}/{soltype}"
                cmd_arg = f"{cmd_arg.split('=')[0]}={temp_gaintable}"
                splited_cmd[i] = cmd_arg
        cmd = " ".join(splited_cmd)
    else:
        print("Please provide valid command.")
        return 1
    cmd_args = cmd.split(" ")
    try:
        full_command = ["udocker", "--quiet", "run", "--nobanner"]
        if datapath is not None:
            full_command.append(f"--volume={datapath}:{temp_docker_path}")
        if gain_path is not None:
            full_command.append(f"--volume={gain_path}:{temp_gain_path}")
        full_command += [
            "--workdir",
            f"{temp_docker_path}",
            f"{container_name}",
        ] + cmd_args
        if verbose:
            print(f"{cmd}\n")
            result = subprocess.run(
                full_command,
            )
        else:
            result = subprocess.run(
                full_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        exit_code = result.returncode
        return 0 if exit_code == 0 else 1
    except Exception:
        traceback.print_exc()
        return 1


def run_hyperdrive(
    hyperdrive_cmd,
    ncpu=1,
    container_name="paircarshyperdrive",
    check_container=False,
    verbose=False,
):
    """
    Run chgcenter inside a udocker container (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    ra : str
        RA can either be 00h00m00.0s or 00:00:00.0
    dec : str
        Dec can either be 00d00m00.0s or 00.00.00.0
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator,
        but while creating the MS, UVW values are estimated using a wrong phase center.
    check_container : bool, optional
        Check container
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    with limit_threads(n_threads=ncpu):
        env = os.environ.copy()
        env_vars = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
        ]
        init_udocker()
        if check_container:
            container_present = check_udocker_container(container_name)
            if not container_present:
                print(f"Initializing {container_name}...")
                container_name = initialize_hyperdrive_container(
                    name=container_name, verbose=True
                )
                if container_name is None:
                    print(
                        f"Container {container_name} is not initiated. First initiate container and then run."
                    )
                    return 1
        cmd_args = hyperdrive_cmd.split(" ")
        outpath = None
        beampath = None
        sourcepath = None
        metapath = None
        for i in range(len(cmd_args)):
            cmd = cmd_args[i]
            if cmd == "-m":
                metafits_name = cmd_args[i + 1]
                metapath = os.path.dirname(os.path.abspath(metafits_name))
                temp_name = "hyperdrive_udocker_" + next(tempfile._get_candidate_names())
                temp_docker_metapath = os.path.join(metapath, temp_name)
                cmd_args[i + 1] = (
                    f"{temp_docker_metapath}/{os.path.basename(metafits_name)}"
                )
            if cmd == "--output-model-files":
                outfile_name = cmd_args[i + 1]
                outpath = os.path.dirname(os.path.abspath(outfile_name))
                temp_name = "hyperdrive_udocker_" + next(tempfile._get_candidate_names())
                temp_docker_outpath = os.path.join(outpath, temp_name)
                cmd_args[i + 1] = f"{temp_docker_outpath}/{os.path.basename(outfile_name)}"
            if cmd == "--beam-file":
                beamfile = cmd_args[i + 1]
                beampath = os.path.dirname(os.path.abspath(beamfile))
                temp_name = "hyperdrive_udocker_" + next(tempfile._get_candidate_names())
                temp_docker_beampath = os.path.join(beampath, temp_name)
                cmd_args[i + 1] = f"{temp_docker_beampath}/{os.path.basename(beamfile)}"
            if cmd == "-s":
                sourcefile = cmd_args[i + 1]
                sourcepath = os.path.dirname(os.path.abspath(sourcefile))
                temp_name = "hyperdrive_udocker_" + next(tempfile._get_candidate_names())
                temp_docker_sourcepath = os.path.join(sourcepath, temp_name)
                cmd_args[i + 1] = f"{temp_docker_sourcepath}/{os.path.basename(sourcefile)}"
        try:
            full_command = ["udocker", "--quiet", "run", "--nobanner"]
            env_keys = list(env.keys())
            for var in env_vars:
                if var in env_keys:
                    full_command.append(f"--env={var}={env[var]}")
            if outpath is not None:
                full_command.append(f"--volume={outpath}:{temp_docker_outpath}")
            if beampath is not None:
                full_command.append(f"--volume={beampath}:{temp_docker_beampath}")
            if sourcepath is not None:
                full_command.append(f"--volume={sourcepath}:{temp_docker_sourcepath}")
            if metapath is not None:
                full_command.append(f"--volume={metapath}:{temp_docker_metapath}")
            full_command += [
                f"{container_name}",
            ] + cmd_args
            if verbose:
                print(f"{hyperdrive_cmd}\n")
                result = subprocess.run(
                    full_command,
                    env=env,
                )
            else:
                result = subprocess.run(
                    full_command,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            exit_code = result.returncode
            return 0 if exit_code == 0 else 1
        except Exception:
            traceback.print_exc()
            return 1


def run_postgres(
    postgres_port=5432,
    container_name="paircarspostgres",
    verbose=False,
):
    """
    Start postgres server

    Parameters
    ----------
    postgres_port : int, optional
        Postgres port
    container_name : str, optional
        container name
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Whether postgres server started or not
    """
    init_udocker()

    datadir = get_datadir()
    pg_credentials = f"{datadir}/postgres_credentials.npy"
    pgdata_dir = f"{datadir}/pgdata"

    postgres_user, postgres_pass, postgres_db = np.load(
        pg_credentials, allow_pickle=True
    )
    hostname = socket.gethostname()
    postgrs_addr = socket.gethostbyname(hostname)

    pid_file = f"{datadir}/postgres.pid"
    log_file = f"{datadir}/postgres.log"
    url_file = f"{datadir}/postgres.url"

    if os.path.exists(log_file):
        os.system(f"rm -rf {log_file}")
    if os.path.exists(url_file):
        os.system(f"rm -rf {url_file}")
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            lines = f.readlines()
        pids = [int(p) for p in lines]
        for pid in pids:
            terminate_process_and_children(pid)
        os.system(f"rm -rf {pid_file}")
    kill_port(int(postgres_port))

    env = os.environ.copy()

    ########################################################
    # Deleting any running postgres container and reinitiate
    ########################################################
    container_present = check_udocker_container(container_name)
    if not container_present:
        container_name = initialize_postgres_container(
            name=container_name, verbose=verbose
        )
        if container_name is None:
            print(
                f"Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1

    ###########################################################################
    # Setup udocker execution mode. Execmode Pn is required for port publishing
    ###########################################################################
    cmd = ["udocker", "setup", "--execmode=P1", f"{container_name}"]
    if verbose:
        subprocess.run(
            cmd,
            env=env,
        )
    else:
        subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    ##########################
    # Starting postgres server
    ##########################
    print("Starting postgres server....")
    os.makedirs(pgdata_dir, exist_ok=True)
    cmd = [
        "udocker",
        "run",
        f"--publish={postgres_port}:5432",
        f"--volume={pgdata_dir}:/var/lib/postgresql/data",
        f"--env=POSTGRES_PASSWORD={postgres_pass}",
        f"--env=POSTGRES_USER={postgres_user}",
        f"--env=POSTGRES_DB={postgres_db}",
        f"{container_name}",
        "-c",
        "listen_addresses=*",
        "-c",
        "max_connections=1000",
        "-c",
        "shared_buffers=1024MB",
    ]

    with open(log_file, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    pid = proc.pid
    with open(pid_file, "w") as f:
        f.write(f"{pid}\n")

    postgres_url = f"postgresql+asyncpg://{postgres_user}:{postgres_pass}@{postgrs_addr}:{postgres_port}/{postgres_db}"
    print("Waiting for PostgreSQL...")
    if not wait_for_port("127.0.0.1", postgres_port, timeout=300):
        print("PostgreSQL failed to start.")
        return False
    else:
        print("PostgreSQL running.")
        with open(url_file, "w") as f:
            f.write(postgres_url.strip())
        return True


def kill_postgres(
    postgres_port=5432,
    container_name="paircarspostgres",
    verbose=False,
):
    """
    Kill postgres server

    Parameters
    ----------
    postgres_port : int, optional
        Postgres server
    container_name : str, optional
        Container name

    Returns
    -------
    int
        Whether closed or not

    """
    init_udocker()
    datadir = get_datadir()
    pid_file = f"{datadir}/postgres.pid"
    log_file = f"{datadir}/postgres.log"
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            lines = f.readlines()
        pids = [int(p) for p in lines]
        for pid in pids:
            terminate_process_and_children(pid)
        os.system(f"rm -rf {pid_file}")
    kill_port(int(postgres_port))
    env = os.environ.copy()
    ########################################################
    # Deleting any running postgres container and reinitiate
    ########################################################
    container_present = check_udocker_container(container_name)
    if container_present:
        if verbose:
            subprocess.run(
                ["udocker", "rm", f"{container_name}"],
                env=env,
            )
        else:
            subprocess.run(
                ["udocker", "rm", f"{container_name}"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # os.system(f"rm -rf {pgdata_dir}")

    print("Checking for PostgreSQL status...")
    if wait_for_port("127.0.0.1", postgres_port, timeout=10):
        print("PostgreSQL kill is failed.")
        return False
    else:
        print("PostgreSQL is killed.")
        if os.path.exists(log_file):
            os.system(f"rm -rf {log_file}")
        return True
