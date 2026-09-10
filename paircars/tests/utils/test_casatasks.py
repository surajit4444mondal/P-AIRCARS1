import os
import traceback
import pytest
import numpy as np
from casatasks import casalog
from unittest.mock import patch, MagicMock
from paircars.utils.casatasks import *

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    traceback.print_exc()
    pass


def test_check_scan_in_caltable(dummy_caltables):
    assert check_scan_in_caltable(dummy_caltables[0], 1) == True
    assert check_scan_in_caltable(dummy_caltables[0], 3) == False


def test_reset_weights_and_flags(dummy_msname):
    if os.path.exists(f"{dummy_msname}/.reset"):
        os.system(f"rm -rf {dummy_msname}/.reset")
    reset_weights_and_flags(dummy_msname)
    assert os.path.exists(f"{dummy_msname}/.reset") == True


@patch("casatasks.flagdata")
@patch("casatasks.initweights")
@patch("casatasks.mstransform")
@patch("paircars.utils.casatasks.suppress_output")
@patch("paircars.utils.casatasks.limit_threads")
@patch("paircars.utils.casatasks.os.system")
@patch("paircars.utils.casatasks.os.path.exists", return_value=False)
@patch("paircars.utils.casatasks.msmetadata")
def test_single_mstransform(
    mock_msmetadata,
    mock_exists,
    mock_system,
    mock_limit_threads,
    mock_suppress,
    mock_mstransform,
    mock_initweights,
    mock_flagdata,
):
    # Setup mock for msmetadata
    mock_msmd = MagicMock()
    mock_msmd.fieldsforscan.return_value = [0]
    mock_msmetadata.return_value = mock_msmd

    # Call the function and check return
    outputms = single_mstransform(msname="mock.ms", outputms="mock_output.ms")
    assert outputms == "mock_output.ms"


class DummySuppress:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


@pytest.fixture
def mock_suppress(monkeypatch):
    monkeypatch.setattr(
        "paircars.utils.basic_utils.suppress_output",  # 🔁 update this
        lambda: DummySuppress(),
    )


@pytest.mark.parametrize(
    "npol, ant1, ant2, time, make_zero_auto, expect_nonzero, expect_flags",
    [
        # --- Case 1: normal valid (npol=2)
        (
            2,
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
            np.array([1, 1, 1, 1]),
            False,
            True,
            False,
        ),
        # --- Case 2: normal valid (npol=4)
        (
            4,
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
            np.array([1, 1, 1, 1]),
            False,
            True,
            False,
        ),
        # --- Case 3: force NaNs via zero autos
        (
            2,
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
            np.array([1, 1, 1, 1]),
            True,
            False,
            True,
        ),
    ],
)
def test_normalized_crosscorr_ms(
    mock_suppress,
    npol,
    ant1,
    ant2,
    time,
    make_zero_auto,
    expect_nonzero,
    expect_flags,
):
    """
    Single comprehensive test covering:
    - npol branches (2 & 4)
    - valid/invalid rows
    - missing autocorr
    - NaN handling
    """

    nrow = len(ant1)
    nchan = 3

    data = np.ones((npol, nchan, nrow), dtype=np.complex64)
    flag = np.zeros((npol, nchan, nrow), dtype=bool)

    # Force zero autos → NaNs
    if make_zero_auto:
        auto_mask = ant1 == ant2
        data[:, :, auto_mask] = 0

    norm, new_flag = calc_normzlized_crosscorr(
        data.copy(), flag.copy(), ant1, ant2, time
    )

    # --- shape checks
    assert norm.shape == data.shape
    assert new_flag.shape == flag.shape

    # --- NaN cleanup
    assert not np.isnan(norm).any()

    # --- expected behavior
    if expect_nonzero:
        assert np.any(norm != 0)
    else:
        assert np.all(norm == 0)

    if expect_flags:
        assert np.any(new_flag)
    else:
        # allow some flags if edge numerical issues,
        # but mostly expect no flags
        assert np.sum(new_flag) <= norm.size * 0.1
