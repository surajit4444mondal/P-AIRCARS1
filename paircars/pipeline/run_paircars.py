import os
import sys
import traceback
import argparse
import shlex
from paircars.utils.basic_utils import (
    check_port_status,
    get_free_port,
    check_permission,
)
from paircars.utils.logger_utils import SmartDefaultsHelpFormatter
from paircars.utils.proc_manage_utils import (
    get_scheduler_name,
    submit_local_master_flow,
    get_jobid,
)
from paircars.utils.prefect_setup_utils import (
    prefect_server_status,
    start_prefect_server,
    stop_prefect_server,
)
from paircars.clusterutils.slurm_cluster import submit_slurm_master_flow

def cli():
    parser = argparse.ArgumentParser(
        description="Run P-AIRCARS for calibration and imaging of solar observations.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    # === Essential parameters ===
    essential = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    essential.add_argument(
        "target_datadir", type=str, help="Target measurement set directory"
    )
    essential.add_argument(
        "--workdir",
        type=str,
        dest="workdir",
        required=True,
        help="Working directory",
    )
    essential.add_argument(
        "--outdir",
        type=str,
        dest="outdir",
        required=True,
        help="Output products directory",
    )
    essential.add_argument(
        "--target_metafits",
        type=str,
        default="",
        dest="target_metafits",
        help="Target metafits file",
    )
    essential.add_argument(
        "--cal_datadir",
        type=str,
        default="",
        dest="cal_datadir",
        help="Calibrator measurement set directory",
    )
    essential.add_argument(
        "--cal_metafits",
        type=str,
        default="",
        dest="cal_metafits",
        help="Calibrator metafits file",
    )

    # === Advanced calibration parameters ===
    advanced_cal = parser.add_argument_group(
        "###################\nAdvanced calibration parameters\n###################"
    )
    advanced_cal.add_argument(
        "--solint",
        type=str,
        default="60s",
        help="Solution interval for calibration (e.g. 'int', '10s', '5min', 'inf')",
    )
    advanced_cal.add_argument(
        "--cal_uvrange",
        type=str,
        default="",
        help="UV range to filter data for calibration (e.g. '>100klambda', '100~10000lambda')",
    )
    advanced_cal.add_argument(
        "--no_polcal",
        action="store_false",
        dest="do_polcal",
        help="Disable polarization calibration",
    )
    advanced_cal.add_argument(
        "--only_amplitude",
        action="store_true",
        help="Apply only amplitude part of gain solution from calibrator or not",
    )
    advanced_cal.add_argument(
        "--redo_basic_cal",
        action="store_true",
        help="Redo basic calibration or not",
    )
    advanced_cal.add_argument(
        "--redo_selfcal",
        action="store_true",
        help="Redo self-calibration or not",
    )
    advanced_cal.add_argument(
        "--use_solarflagger",
        action="store_true",
        help="Use solar flagger",
    )

    # === Advanced imaging parameters ===
    advanced_image = parser.add_argument_group(
        "###################\nAdvanced imaging parameters\n###################"
    )
    advanced_image.add_argument(
        "--freqrange",
        type=str,
        default="",
        help="Frequency range in MHz to select during imaging (comma-separated, e.g. '100~110,130~140')",
    )
    advanced_image.add_argument(
        "--timerange",
        type=str,
        default="",
        help="Time range to select during imaging (comma-separated, e.g. '2014/09/06/09:30:00~2014/09/06/09:45:00,2014/09/06/10:30:00~2014/09/06/10:45:00')",
    )
    advanced_image.add_argument(
        "--image_freqres",
        type=float,
        default=1.28,
        help="Output image frequency resolution in MHz (-1 = full)",
    )
    advanced_image.add_argument(
        "--image_timeres",
        type=float,
        default=10.0,
        help="Output image time resolution in seconds (-1 = full)",
    )
    advanced_image.add_argument(
        "--pol",
        type=str,
        default="IQUV",
        help="Stokes parameter(s) to image ('I' or 'IQUV')",
    )
    advanced_image.add_argument(
        "--minuv",
        type=float,
        default=0,
        help="Minimum baseline length (in wavelengths) to include in imaging",
    )
    advanced_image.add_argument(
        "--weight",
        type=str,
        default="briggs",
        help="Imaging weighting scheme (e.g. 'briggs', 'natural', 'uniform')",
    )
    advanced_image.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust parameter for Briggs weighting (-2 to +2)",
    )
    advanced_image.add_argument(
        "--no_multiscale",
        action="store_false",
        dest="use_multiscale",
        help="Disable multiscale CLEAN for extended structures",
    )
    advanced_image.add_argument(
        "--clean_threshold",
        type=float,
        default=1.0,
        help="Clean threshold in sigma for final deconvolution",
    )
    advanced_image.add_argument(
        "--no_pbcor",
        action="store_false",
        dest="do_pbcor",
        help="Do not apply primary beam correction after imaging",
    )
    advanced_image.add_argument(
        "--cutout_rsun",
        type=float,
        default=10.0,
        help="Field of view cutout radius in solar radii",
    )
    advanced_image.add_argument(
        "--no_solar_mask",
        action="store_false",
        dest="use_solar_mask",
        help="Disable use solar disk mask during deconvolution",
    )
    advanced_image.add_argument(
        "--do_overlay",
        action="store_true",
        dest="make_overlay",
        help="Make overlay plot on EUV images",
    )

    # === Advanced options ===
    advanced = parser.add_argument_group(
        "###################\nAdvanced pipeline parameters\n###################"
    )
    advanced.add_argument(
        "--make_msplot",
        action="store_true",
        help="Make diagnostic plots of measurement sets",
    )
    advanced.add_argument(
        "--non_solar_data",
        action="store_false",
        dest="solar_data",
        help="Disable solar data mode",
    )
    advanced.add_argument(
        "--no_ds",
        action="store_false",
        dest="make_ds",
        help="Disable making solar dynamic spectra",
    )
    advanced.add_argument(
        "--do_forcereset_weightflag",
        action="store_true",
        help="Force reset of weights and flags (disabled by default)",
    )
    advanced.add_argument(
        "--no_cal_flag",
        action="store_false",
        dest="do_cal_flag",
        help="Disable initial flagging of calibrators",
    )
    advanced.add_argument(
        "--no_import_model",
        action="store_false",
        dest="do_import_model",
        help="Disable model import",
    )
    advanced.add_argument(
        "--no_basic_cal",
        action="store_false",
        dest="do_basic_cal",
        help="Disable basic gain calibration",
    )
    advanced.add_argument(
        "--do_sidereal_cor",
        action="store_true",
        dest="do_sidereal_cor",
        help="Sidereal motion correction for Sun (disabled by default)",
    )
    advanced.add_argument(
        "--no_solarcenter_move",
        action="store_false",
        dest="do_move_solarcenter",
        help="Disable moving phasecenter to solar center",
    )
    advanced.add_argument(
        "--no_selfcal",
        action="store_false",
        dest="do_selfcal",
        help="Disable self-calibration",
    )
    advanced.add_argument(
        "--no_ap_selfcal",
        action="store_false",
        dest="do_ap_selfcal",
        help="Disable amplitude-phase self-calibration",
    )
    advanced.add_argument(
        "--no_solar_selfcal",
        action="store_false",
        dest="solar_selfcal",
        help="Disable solar-specific self-calibration parameters",
    )
    advanced.add_argument(
        "--no_applycal",
        action="store_false",
        dest="do_applycal",
        help="Disable application of basic calibration solutions",
    )
    advanced.add_argument(
        "--no_apply_selfcal",
        action="store_false",
        dest="do_apply_selfcal",
        help="Disable application of self-calibration solutions",
    )
    advanced.add_argument(
        "--no_imaging",
        action="store_false",
        dest="do_imaging",
        help="Disable final imaging",
    )
    advanced.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs",
    )

    # === Advanced local system/ per node hardware resource parameters ===
    advanced_resource = parser.add_argument_group(
        "###################\nAdvanced hardware resource parameters for local system or per node on HPC cluster\n###################"
    )
    advanced_resource.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="Fraction of CPU usuage per node",
    )
    advanced_resource.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Fraction of memory usuage per node",
    )
    advanced_resource.add_argument(
        "--max_worker",
        type=int,
        default=-1,
        help="Maximum number of workers",
    )
    advanced_resource.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of intermediate steps",
    )
    advanced_resource.add_argument(
        "--no_calibrated_ms",
        action="store_false",
        dest="keep_calibrated_ms",
        help="Keep calibrated measurement sets or not",
    )
    advanced_resource.add_argument(
        "--no_remote_logger",
        action="store_false",
        dest="remote_logger",
        help="Disable remote logger",
    )
    advanced_resource.add_argument(
        "--log2term",
        action="store_true",
        help="Show logs in terminal",
    )
    advanced_resource.add_argument(
        "--job_password",
        type=str,
        default=None,
        help="User specified job password",
    )
    advanced_resource.add_argument(
        "--cluster",
        action="store_true",
        dest="cluster",
        help="Running in cluster environment",
    )
    advanced_resource.add_argument(
        "--port",
        type=int,
        default=4260,
        help="Prefect port",
    )

    # === Advanced job scheduler parameters ===
    advanced_slurm = parser.add_argument_group(
        "###################\nAdvanced slurm cluster settings\n###################"
    )
    advanced_slurm.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Partition name (Required)",
    )
    advanced_slurm.add_argument(
        "--account",
        type=str,
        default=None,
        help="Account name (If your cluster requires this, you should provide. Otherwise job can not be started)",
    )
    advanced_slurm.add_argument(
        "--walltime",
        type=str,
        default=None,
        help="Wall time, each slurm job can execute in maximum this time",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.target_datadir.startswith("~"):
        print("Please provide full path of target directory.")
        return 1
    if args.workdir.startswith("~"):
        print("Please provide full path of work directory.")
        return 1
    if args.outdir.startswith("~"):
        print("Please provide full path of output directory.")
        return 1

    target_datadir_permission = check_permission(args.target_datadir)
    if args.target_datadir is not None and args.target_datadir!="" and target_datadir_permission is False:
        print(
            f"Do not have permission for target data directory: {args.target_datadir}"
        )
        return

    cal_datadir_list = args.cal_datadir.split(",")
    filtered_cal_datadir_list = []
    for cal_datadir in cal_datadir_list:
        cal_datadir_permission = check_permission(cal_datadir)
        if cal_datadir is not None and cal_datadir!="" and cal_datadir_permission is False:
            print(
                f"Do not have permission for calibrator data directory: {cal_datadir}"
            )
        else:
            filtered_cal_datadir_list.append(cal_datadir)
    cal_datadir_list = filtered_cal_datadir_list
    if len(cal_datadir_list) > 0:
        args.cal_datadir = ",".join(cal_datadir_list)
    else:
        args.cal_datadir = ""

    jobid = get_jobid()
    scheduler_name = get_scheduler_name()

    prefect_status = prefect_server_status(scheduler_name=scheduler_name)
    if prefect_status is False:
        print("Prefect server is not running.")
        print("Prefect setup is initiating ....")
        port = args.port
        postgres_port = port + 1000

        if check_port_status(port) is False:
            if scheduler_name != "local":
                port = get_free_port(start_port=port, end_port=port + 990)

        if check_port_status(postgres_port) is False:
            if scheduler_name != "local":
                postgres_port = get_free_port(
                    start_port=postgres_port, end_port=postgres_port + 990
                )

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
    try:
        ############################################
        # Submitting batch script
        ############################################
        args_list = [shlex.quote(arg) for arg in sys.argv[1:]]
        cli_cmd = "run-mwa-paircars " + " ".join(args_list)
        os.makedirs(f"{args.workdir}",exist_ok=True)
        if os.path.exists(f"{args.workdir}/inputs.txt"):
            os.system(f"rm -rf {args.workdir}/inputs.txt")
        with open(f"{args.workdir}/inputs.txt","w") as f:
            f.write(cli_cmd)
        
        if not args.cluster or scheduler_name == "local":
            msg = submit_local_master_flow(args, jobid)
            if msg != 0:
                print(
                    "Some error may occured in batch script execution, while P-AIRCARS may be successully completed."
                )
            return msg
        elif scheduler_name == "slurm":
            msg = submit_slurm_master_flow(args, jobid)
            if msg != 0:
                print(
                    "Some error may occured in batch script execution, while P-AIRCARS may be successully completed."
                )
            return msg
        else:
            print(
                f"P-AIRCARS currently does not support {scheduler_name} job scheduler."
            )
            return 1
    except Exception:
        print("Error occured in executing P-AIRCARS master flow.")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    cli()
