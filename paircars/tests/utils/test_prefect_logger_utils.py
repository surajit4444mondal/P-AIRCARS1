import pytest
import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from datetime import datetime, timezone
from uuid import uuid4


from paircars.utils.prefect_logger_utils import (
    save_logs_by_task_id,
    save_logs_by_flow_id,
    start_log_task_saver,
    start_flow_log_saver,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raise_error, final_logs",
    [
        (False, True),
        (True, False),
    ],
)
async def test_save_logs_by_task_id(raise_error, final_logs):
    task_id = str(uuid4())
    ts = datetime.now(timezone.utc)

    def make_log(log_id="log1", msg="Test message"):
        log = MagicMock()
        log.id = log_id
        log.message = msg
        log.task_run_id = task_id
        log.timestamp = ts  # use real datetime (no mocking headache)

        log.level = MagicMock()
        log.level.name = "INFO"
        return log

    first_log = make_log("log1", "Test message")
    second_log = make_log("log2", "Final message")
    mock_client = AsyncMock()
    if raise_error:
        mock_client.read_logs.side_effect = Exception("boom")
    else:
        if final_logs:
            mock_client.read_logs.side_effect = [
                [first_log],
                [second_log],
                [],
            ]
        else:
            mock_client.read_logs.return_value = [first_log]
    with (
        patch(
            "paircars.utils.prefect_logger_utils.get_client",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)),
        ),
        patch(
            "paircars.utils.prefect_logger_utils.local_tz",
            timezone.utc,
        ),
        patch("builtins.open", mock_open()) as m,
        patch("os.makedirs"),
    ):
        stop_event = threading.Event()
        task = asyncio.create_task(
            save_logs_by_task_id(
                task_id,
                "test-task",
                "logfile.log",
                poll_interval=1,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(0.2)
        stop_event.set()
        await task
        writes = [call.args[0] for call in m().write.call_args_list]

        if raise_error:
            assert any("Error fetching task logs" in w for w in writes)
        else:
            expected_ts = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            assert any(
                f"INFO | {expected_ts} | test-task | Test message\n" in w
                for w in writes
            )
            if final_logs:
                assert any(
                    f"INFO | {expected_ts} | test-task | Final message\n" in w
                    for w in writes
                )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raise_error, final_logs",
    [
        (False, True),
        (True, False),
    ],
)
async def test_save_logs_by_flow_id(raise_error, final_logs):
    flow_id = str(uuid4())
    other_flow_id = str(uuid4())
    ts = datetime.now(timezone.utc)

    def make_log(
        log_id,
        msg,
        flow_run_id,
        task_run_id=None,
    ):
        log = MagicMock()
        log.id = log_id
        log.message = msg
        log.flow_run_id = flow_run_id
        log.task_run_id = task_run_id
        log.timestamp = ts

        log.level = MagicMock()
        log.level.name = "INFO"
        return log

    flow_log = make_log("log1", "Flow message", flow_id, None)
    task_log = make_log("log2", "Task message", flow_id, "task123")
    wrong_flow_log = make_log("log3", "Wrong flow", other_flow_id, None)
    final_log = make_log("log4", "Final flow message", flow_id, None)
    mock_client = AsyncMock()
    if raise_error:
        mock_client.read_logs.side_effect = Exception("boom")
    else:
        if final_logs:
            mock_client.read_logs.side_effect = [
                [flow_log, task_log, wrong_flow_log],  # main loop
                [final_log],  # final drain
                [],
            ]
        else:
            mock_client.read_logs.return_value = [
                flow_log,
                task_log,
                wrong_flow_log,
            ]
    with (
        patch(
            "paircars.utils.prefect_logger_utils.get_client",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)),
        ),
        patch(
            "paircars.utils.prefect_logger_utils.local_tz",
            timezone.utc,
        ),
        patch("builtins.open", mock_open()) as m,
        patch("os.makedirs"),
    ):
        stop_event = threading.Event()
        task = asyncio.create_task(
            save_logs_by_flow_id(
                flow_id,
                "test-flow",
                "logfile.log",
                poll_interval=1,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(0.2)
        stop_event.set()
        await task
        writes = [call.args[0] for call in m().write.call_args_list]
        if raise_error:
            assert any("Error fetching flow logs" in w for w in writes)
        else:
            expected_ts = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            assert any(
                f"INFO | {expected_ts} | test-flow | Flow message\n" in w
                for w in writes
            )
            assert not any("Task message" in w for w in writes)
            assert not any("Wrong flow" in w for w in writes)
            if final_logs:
                assert any(
                    f"INFO | {expected_ts} | test-flow | Final flow message\n" in w
                    for w in writes
                )


def test_start_log_task_saver():
    stop_event = threading.Event()

    with patch(
        "paircars.utils.prefect_logger_utils.save_logs_by_task_id",
        new_callable=AsyncMock,
    ) as mock_async:
        thread = start_log_task_saver(
            "taskid",
            "test-task",
            "logfile.log",
            poll_interval=0.1,
            stop_event=stop_event,
        )
        assert isinstance(thread, threading.Thread)
        time.sleep(0.2)  # Let thread spin up
        stop_event.set()
        thread.join(timeout=1)
        mock_async.assert_called_once()


def test_start_flow_log_saver():
    stop_event = threading.Event()

    with patch(
        "paircars.utils.prefect_logger_utils.save_logs_by_flow_id",
        new_callable=AsyncMock,
    ) as mock_async:
        thread = start_flow_log_saver(
            "flowid",
            "test-flow",
            "flowfile.log",
            poll_interval=0.1,
            stop_event=stop_event,
        )
        assert isinstance(thread, threading.Thread)
        time.sleep(0.2)
        stop_event.set()
        thread.join(timeout=1)
        mock_async.assert_called_once()
