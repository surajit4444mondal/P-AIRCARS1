import pytest
from unittest.mock import patch, MagicMock
from paircars.pipeline.do_apply_basiccal import *


@pytest.mark.parametrize(
    "exists_applied, force_apply, quartical_present, quartical_msg, overwrite, raise_exc",
    [
        (True, False, False, 0, False, False),  # already applied
        (True, True, False, 0, False, False),  # force apply
        (False, False, False, 0, False, False),  # quartical missing
        (False, False, True, 0, False, False),  # quartical success
        (False, False, True, 1, False, False),  # quartical failure
        (False, False, False, 0, True, False),  # overwrite branch
        (False, False, False, 0, False, True),  # exception branch
    ],
)
def test_applysol(
    exists_applied,
    force_apply,
    quartical_present,
    quartical_msg,
    overwrite,
    raise_exc,
):
    msname = "test.ms"
    qc_table = "qc_dir"
    with (
        patch("paircars.pipeline.do_apply_basiccal.limit_threads"),
        patch("paircars.pipeline.do_apply_basiccal.os.path.exists") as m_exists,
        patch("paircars.pipeline.do_apply_basiccal.os.system") as m_system,
        patch("paircars.pipeline.do_apply_basiccal.os.listdir", return_value=["term1"]),
        patch(
            "paircars.pipeline.do_apply_basiccal.glob.glob",
            return_value=[f"{msname}/.touch"],
        ),
        patch(
            "paircars.pipeline.do_apply_basiccal.run_quartical",
            return_value=quartical_msg,
        ),
        patch("paircars.pipeline.do_apply_basiccal.suppress_output") as m_suppress,
        patch("paircars.pipeline.do_apply_basiccal.traceback.print_exc"),
    ):
        m_suppress.return_value.__enter__.return_value = None
        m_suppress.return_value.__exit__.return_value = None
        with patch.dict(
            "sys.modules",
            {
                "casatasks": MagicMock(
                    applycal=MagicMock(),
                    flagdata=MagicMock(),
                    split=MagicMock(),
                    clearcal=MagicMock(),
                )
            },
        ):

            def exists_side_effect(path):
                if raise_exc:
                    raise Exception("boom")
                if path == msname + "/.applied_sol":
                    return exists_applied
                if path == qc_table:
                    return quartical_present
                if path.endswith(".flagversions"):
                    return True
                return True

            m_exists.side_effect = exists_side_effect
            result = applysol(
                msname,
                "/mock/workdir",
                gaintable=["g1"],
                gainfield=[""],
                interp=["nearest"],
                quartical_table=[qc_table],
                overwrite_datacolumn=overwrite,
                force_apply=force_apply,
                soltype="basic",
            )
            msg = result[0]
            qc_msg = result[1]
            if raise_exc:
                assert msg == 1
                return
            if exists_applied and not force_apply:
                assert msg == 0
                return
            assert msg == 0


def mock_glob_pattern(pattern):
    if "attval_scan" in pattern:
        return ["myms_attval_scan_9.npy"]
    elif "calibrator_caltable_scan" in pattern:
        return ["/mock/caldir/calibrator_caltable_scan_9.bcal"]
    elif "bcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.bcal"]
    elif "kcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.kcal"]
    elif "gcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.gcal"]
    elif "dcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.dcal"]
    elif "kcrosscal" in pattern:
        return ["/mock/caldir/calibrator_caltable.kcrosscal"]
    elif "xfcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.xfcal"]
    elif "panglecal" in pattern:
        return ["/mock/caldir/calibrator_caltable.panglecal"]
    return []


@pytest.mark.parametrize(
    "has_bandpass, has_crossphase, valid_ms, results_sum, raise_exc",
    [
        (False, False, True, 0, False),  # no bandpass -> []
        (True, False, True, 0, False),  # bandpass only, success
        (True, True, True, 0, False),  # bandpass + crossphase success
        (True, True, True, 1, False),  # failure results
        (True, True, False, 0, False),  # no valid ms
        (True, True, True, 0, True),  # exception branch
    ],
)
def test_run_all_applysol(
    has_bandpass,
    has_crossphase,
    valid_ms,
    results_sum,
    raise_exc,
):

    mslist = np.array(["a.ms", "b.ms"])
    fake_client = MagicMock()

    with (
        patch("paircars.pipeline.do_apply_basiccal.os.chdir"),
        patch("paircars.pipeline.do_apply_basiccal.np.unique", return_value=mslist),
        patch("paircars.pipeline.do_apply_basiccal.fits.getheader") as m_header,
        patch("paircars.pipeline.do_apply_basiccal.glob.glob") as m_glob,
        patch(
            "paircars.pipeline.do_apply_basiccal.check_datacolumn_valid"
        ) as m_checkcol,
        patch("paircars.pipeline.do_apply_basiccal.msmetadata") as m_msmd,
        patch(
            "paircars.pipeline.do_apply_basiccal.get_nearest_bandpass_table",
            return_value="nearest.bcal",
        ),
        patch("paircars.pipeline.do_apply_basiccal.delayed", side_effect=lambda f: f),
        patch("paircars.pipeline.do_apply_basiccal.applysol_wrapper") as m_apply,
        patch("paircars.pipeline.do_apply_basiccal.traceback.print_exc"),
        patch("paircars.pipeline.do_apply_basiccal.os.system") as m_system,
    ):

        # ---- FITS headers ----
        m_header.side_effect = [
            {"GPSTIME": "123", "ATTEN_DB": 10},  # calibrator
            {"ATTEN_DB": 20},  # target
        ]

        # ---- glob tables ----
        if not has_bandpass:
            m_glob.side_effect = [[], []]
        else:
            bpass = ["123.bcal"]
            cross = ["123.kcrosscal"] if has_crossphase else []
            m_glob.side_effect = [bpass, cross]

        # ---- MS validity ----
        if valid_ms:
            m_checkcol.return_value = True
        else:
            m_checkcol.return_value = False

        # ---- msmetadata ----
        msmd_inst = MagicMock()
        m_msmd.return_value = msmd_inst
        msmd_inst.meanfreq.return_value = 150

        # ---- Dask ----
        if raise_exc:
            fake_client.compute.side_effect = Exception("boom")
        else:
            fake_client.compute.side_effect = lambda x: x
            patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
            fake_client.gather.side_effect = lambda x: [
                (ms, patterns[i % len(patterns)], f"out_{ms}", f"err_{ms}")
                for i, ms in enumerate(mslist)
            ]
        m_system.return_value = True
        result = run_all_applysol(
            mslist=mslist,
            target_metafits="target.fits",
            dask_client=fake_client,
            workdir="/tmp",
            caldir="/cal",
        )


@pytest.mark.parametrize(
    "start_remote_log, provide_dask, caldir_exists, raise_exc, run_result",
    [
        (False, False, True, False, (0, 2, 0)),  # normal, local dask
        (True, False, True, False, (0, 2, 0)),  # remote logging
        (False, True, True, False, (0, 2, 0)),  # external dask client
        (False, False, False, False, (1, 0, 2)),  # caldir missing
        (False, False, True, True, (1, 0, 2)),  # exception branch
    ],
)
def test_main_applysol(
    start_remote_log,
    provide_dask,
    caldir_exists,
    raise_exc,
    run_result,
):
    mslist = "a.ms,b.ms"

    fake_client = MagicMock()
    fake_cluster = MagicMock()

    with (
        patch(
            "paircars.pipeline.do_apply_basiccal.get_local_dask_cluster"
        ) as m_cluster,
        patch("paircars.pipeline.do_apply_basiccal.scale_worker_and_wait"),
        patch("paircars.pipeline.do_apply_basiccal.run_all_applysol") as m_run,
        patch("paircars.pipeline.do_apply_basiccal.drop_cache") as m_drop,
        patch("paircars.pipeline.do_apply_basiccal.clean_shutdown") as m_clean,
        patch("paircars.pipeline.do_apply_basiccal.init_logger") as m_logger,
        patch("paircars.pipeline.do_apply_basiccal.os.makedirs"),
        patch("paircars.pipeline.do_apply_basiccal.os.path.exists") as m_exists,
        patch("paircars.pipeline.do_apply_basiccal.os.system"),
        patch("paircars.pipeline.do_apply_basiccal.time.sleep"),
        patch("paircars.pipeline.do_apply_basiccal.traceback.print_exc"),
        patch("paircars.pipeline.do_apply_basiccal.os.chdir") as m_chdir,
        patch("paircars.pipeline.do_apply_basiccal.msmetadata") as m_msmd,
        patch("paircars.pipeline.do_apply_basiccal.get_ncoarse") as m_ncoarse,
    ):

        def exists_side_effect(path):
            if "jobname_password.npy" in path:
                return start_remote_log
            if path == "/cal":
                return caldir_exists
            return True

        m_chdir.return_value = 0
        m_exists.side_effect = exists_side_effect
        m_cluster.return_value = (fake_client, fake_cluster, "/tmp/daskdir", 1)
        mock_msmd = MagicMock()
        m_msmd.return_value = mock_msmd
        mock_msmd.meanfreq.return_value = 150.0
        m_ncoarse.return_value = 1

        if start_remote_log:
            with patch(
                "paircars.pipeline.basic_cal.np.load", return_value=("job", "pass")
            ):
                m_logger.return_value = MagicMock()

                m_run.return_value = run_result
                result, succeed, failed = main(
                    mslist=mslist,
                    target_metafits="tar.fits",
                    workdir="",
                    caldir="/cal",
                    start_remote_log=True,
                    logfile="/tmp/log.txt",
                    dask_client=None if not provide_dask else fake_client,
                )
        else:
            if raise_exc:
                m_run.side_effect = Exception("boom")
            else:
                m_run.return_value = run_result

            result, succeed, failed = main(
                mslist=mslist,
                target_metafits="tar.fits",
                workdir="",
                caldir="/cal",
                start_remote_log=False,
                logfile=None,
                dask_client=None if not provide_dask else fake_client,
            )
            fake_client.scheduler_info.return_value = {
                "workers": {
                    "tcp://worker-1": {"memory_limit": 8 * 1024**3},  # 8 GB
                    "tcp://worker-2": {"memory_limit": 4 * 1024**3},  # 4 GB
                }
            }
        assert result == run_result[0]
        assert succeed == run_result[1]
        assert failed == run_result[2]
        if not provide_dask:
            fake_client.close.assert_called()
            fake_cluster.close.assert_called()


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),
        (
            [
                "prog.py",
                "ms1.ms,ms2.ms",
                "--target_metafits",
                "target.metafits",
                "--workdir",
                "/mock/work",
                "--caldir",
                "/mock/caltables",
                "--force_apply",
            ],
            False,
        ),
    ],
)
@patch("paircars.pipeline.do_apply_basiccal.main", return_value=(0, 1, 0))
@patch("paircars.pipeline.do_apply_basiccal.sys.exit")
@patch("paircars.pipeline.do_apply_basiccal.argparse.ArgumentParser.print_help")
def test_cli(mock_print_help, mock_exit, mock_main, argv, should_exit):
    with patch("sys.argv", argv):
        from paircars.pipeline import do_apply_basiccal

        result = do_apply_basiccal.cli()
        assert result == should_exit
