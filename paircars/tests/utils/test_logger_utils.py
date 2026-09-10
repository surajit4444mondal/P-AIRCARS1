import pytest
import string
import tempfile
import logging
import argparse
import os
import urllib.error
from paircars.utils.logger_utils import *
from unittest.mock import MagicMock, mock_open, patch
from watchdog.events import FileModifiedEvent


def test_smart_defaults_help_formatter_suppresses_bool_defaults():
    parser = argparse.ArgumentParser(formatter_class=SmartDefaultsHelpFormatter)
    parser.add_argument("--flag", action="store_true", help="Enable feature")
    parser.add_argument("--value", type=int, default=42, help="Set value")

    help_output = parser.format_help()
    # Boolean default should not appear
    assert "Enable feature (default" not in help_output
    # Non-boolean default should appear
    assert "Set value (default: 42)" in help_output


def test_clean_shutdown_behavior():
    observer = MagicMock()
    clean_shutdown(observer)
    observer.stop.assert_called_once()
    observer.join.assert_called_once_with(timeout=5)


allowed_chars = string.ascii_letters + string.digits + "@#$&*"


@pytest.mark.parametrize("length", [6, 8, 12, 20, 50])
def test_generate_password_properties(length):
    pw1 = generate_password(length)
    pw2 = generate_password(length)
    assert len(pw1) == length
    assert len(pw2) == length
    assert all(c in allowed_chars for c in pw1)
    assert all(c in allowed_chars for c in pw2)
    assert pw1 != pw2  # Extremely unlikely to be the same


@pytest.mark.parametrize(
    "isfile_exists, file_data, urlopen_status, urlopen_exception, expected_output",
    [
        (True, "https://test-render.com\n", 200, None, "https://test-render.com"),
        (True, "https://test-render.com\n", None, urllib.error.URLError("Timeout"), ""),
        (False, "", None, None, ""),
    ],
)
@patch("os.getlogin", return_value="dummyuser")
@patch("paircars.utils.logger_utils.get_cachedir", return_value="/mock/cache")
def test_get_remote_logger_link(
    mock_get_cachedir,
    mock_getlogin,
    isfile_exists,
    file_data,
    urlopen_status,
    urlopen_exception,
    expected_output,
):
    with (
        patch("os.path.isfile", return_value=isfile_exists),
        patch("builtins.open", mock_open(read_data=file_data)) as mock_file,
        patch("urllib.request.urlopen") as mock_urlopen,
    ):

        if urlopen_exception:
            mock_urlopen.side_effect = urlopen_exception
        elif urlopen_status:
            mock_response = MagicMock()
            mock_response.__enter__.return_value.status = urlopen_status
            mock_urlopen.return_value = mock_response
        result = get_remote_logger_link()
        assert result == expected_output


@pytest.mark.parametrize(
    "file_content, file_exists, expected_result",
    [
        ("user@example.com\nanother@example.com\n", True, "user@example.com"),
        ("\n\n", True, ""),
        ("", False, ""),
    ],
)
@patch("paircars.utils.logger_utils.os.getlogin", return_value="testuser")
@patch(
    "paircars.utils.logger_utils.get_cachedir",
    return_value="/mock/.paircars",
)
def test_get_emails_all_cases(
    mock_cachedir, mock_getlogin, file_content, file_exists, expected_result
):
    if file_exists:
        m = mock_open(read_data=file_content)
        open_patch = patch("paircars.utils.logger_utils.open", m)
    else:
        open_patch = patch(
            "paircars.utils.logger_utils.open", side_effect=FileNotFoundError
        )

    with open_patch:
        result = get_emails()
        assert result == expected_result


@patch("paircars.utils.logger_utils.requests.post")
def test_remote_logger_emit_success(mock_post):
    logger = RemoteLogger(
        job_id="job123",
        log_id="log456",
        remote_link="https://mockserver.com",
        password="securepass",
    )

    # Create a dummy log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="This is a test message",
        args=(),
        exc_info=None,
    )

    logger.emit(record)


@patch(
    "paircars.utils.logger_utils.requests.post",
    side_effect=Exception("Connection error"),
)
def test_remote_logger_emit_failure(mock_post):
    logger = RemoteLogger(remote_link="https://mockserver.com")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=20,
        msg="Failing log",
        args=(),
        exc_info=None,
    )

    # Should not raise even though requests.post fails
    logger.emit(record)


def test_log_tail_handler_reads_new_lines():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write("Initial log line\n")
        tmp.flush()
        log_path = tmp.name
    mock_logger = MagicMock()
    handler = LogTailHandler(logfile=log_path, logger=mock_logger)
    with open(log_path, "a") as f:
        f.write("New log line 1\n")
        f.write("\n")
        f.write("New log line 2\n")
        f.flush()
    event = FileModifiedEvent(log_path)
    handler.on_modified(event)
    mock_logger.info.assert_any_call("New log line 1")
    mock_logger.info.assert_any_call("New log line 2")
    assert mock_logger.info.call_count == 2
    os.remove(log_path)


def test_create_logger():
    logfile = os.getcwd() + "/logfile"
    logname = "testlog"
    result_logger, result_logfile = create_logger(logname, logfile)
    assert result_logfile == logfile
    assert os.path.exists(result_logfile) == True
    result_logger.info("Testing")
    os.system(f"rm -rf {result_logfile}")
    assert os.path.exists(result_logfile) == False


@pytest.mark.parametrize(
    "logfile, expected",
    [
        (
            "apply_basiccal_target.log",
            "Applying basic calibration solutions on targets",
        ),
        ("selfcal_target.log", "All self-calibrations"),
        (
            "selfcal_1111474560_ch_103_int.log",
            "Intensity self-calibration, OBSID: 1111474560, coarse channel: 103",
        ),
        (
            "imaging_1111474560_ch_103.log",
            "Imaging, OBSID: 1111474560, coarse channel: 103",
        ),
        ("random_unknown.log", "random_unknown.log"),
        ("another_file.txt", "another_file.txt"),
    ],
)
def test_get_logid(logfile, expected):
    assert get_logid(logfile).split("time_")[-1] == expected


@patch("paircars.utils.logger_utils.Observer")
@patch("paircars.utils.logger_utils.requests.post")
@patch("paircars.utils.logger_utils.get_logid", return_value="MyLogID")
@patch(
    "paircars.utils.logger_utils.get_remote_logger_link",
    return_value="https://mock-logger.com",
)
@patch("paircars.utils.logger_utils.RemoteLogger")
def test_init_logger_remote_success(
    mock_remote_logger, mock_getlink, mock_logid, mock_post, mock_observer
):
    with tempfile.NamedTemporaryFile("w", delete=False) as tmpfile:
        logfile = tmpfile.name
    observer_mock = MagicMock()
    mock_observer.return_value = observer_mock
    result = init_logger("mylogger", logfile, jobname="JOB123", password="pw")
    assert isinstance(result, MagicMock)
    mock_remote_logger.assert_called_once()
    mock_post.assert_called_once_with(
        "https://mock-logger.com/api/log",
        json={
            "job_id": "JOB123",
            "log_id": "MyLogID",
            "message": "",
            "log_type": "task",
            "password": "pw",
            "first": True,
        },
        timeout=2,
    )
    mock_observer.return_value.schedule.assert_called()
    mock_observer.return_value.start.assert_called()
    os.remove(logfile)
