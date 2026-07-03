import os
import asyncio
import threading
from prefect.client.orchestration import get_client
from prefect.client.schemas.sorting import LogSort
from prefect.client.schemas.filters import LogFilter
from datetime import datetime, timezone, timedelta

local_tz = datetime.now().astimezone().tzinfo


async def save_logs_by_task_id(
    task_run_id, task_name, logfile, poll_interval=5, stop_event=None
):
    """
    Fetch and save prefect task logs to a file

    Parameters
    ----------
    tak_run_id : str
        The Prefect task run ID to monitor
    taks_name : str
        Task name
    logfile : str
        Output log file
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """
    poll_interval=int(poll_interval)
    logdir = os.path.dirname(os.path.abspath(logfile))
    os.makedirs(logdir, exist_ok=True)
    seen_ids = set()
    last_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with get_client() as client:
        while True:
            should_stop = stop_event and stop_event.is_set()
            try:
                log_filter = LogFilter(
                    task_run_id={"any_": [task_run_id]},
                    timestamp={"after_": last_timestamp},
                )
                logs = await client.read_logs(
                    log_filter=log_filter,
                    sort=LogSort.TIMESTAMP_ASC,
                )
                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id in seen_ids:
                            continue
                        seen_ids.add(log.id)
                        if str(log.task_run_id) == str(task_run_id):
                            ts = log.timestamp.astimezone(local_tz).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            level = (
                                log.level.name
                                if hasattr(log.level, "name")
                                else str(log.level)
                            )
                            f.write(
                                f"{level} | {ts} | {task_name} | {log.message}\n"
                            )
                        if log.timestamp > last_timestamp:
                            last_timestamp = log.timestamp
            except Exception as e:
                with open(logfile, "a") as f:
                    f.write(f"Error fetching task logs: {e}\n")
            if should_stop:
                break
            # interruptible sleep
            for _ in range(poll_interval):
                if stop_event and stop_event.is_set():
                    break
                await asyncio.sleep(1)
        try:
            for _ in range(3):  # retry a few times to catch delayed logs
                log_filter = LogFilter(
                    task_run_id={"any_": [task_run_id]},
                    timestamp={"after_": last_timestamp},
                )

                logs = await client.read_logs(
                    log_filter=log_filter,
                    sort=LogSort.TIMESTAMP_ASC,
                )

                if not logs:
                    break
                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id in seen_ids:
                            continue
                        seen_ids.add(log.id)
                        if str(log.task_run_id) == str(task_run_id):
                            ts = log.timestamp.astimezone(local_tz).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            level = (
                                log.level.name
                                if hasattr(log.level, "name")
                                else str(log.level)
                            )
                            f.write(
                                f"{level} | {ts} | {task_name} | {log.message}\n"
                            )
                        if log.timestamp > last_timestamp:
                            last_timestamp = log.timestamp
                await asyncio.sleep(1)
        except Exception as e:
            print("FINAL DRAIN ERROR:", e)


        
async def save_logs_by_flow_id(
    flow_run_id, flow_name, logfile, poll_interval=5, stop_event=None
):
    """
    Fetch and save prefect flow logs to a file

    Parameters
    ----------
    flow_run_id : str
        The Prefect flow run ID to monitor
    flow_name : str
        Flow name
    logfile : str
        Output log file
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """
    poll_interval=int(poll_interval)
    logdir = os.path.dirname(os.path.abspath(logfile))
    os.makedirs(logdir, exist_ok=True)
    seen_ids = set()
    last_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with get_client() as client:
        while True:
            should_stop = stop_event and stop_event.is_set()
            try:
                log_filter = LogFilter(
                    flow_run_id={"any_": [flow_run_id]},
                    timestamp={"after_": last_timestamp},
                )
                logs = await client.read_logs(
                    log_filter=log_filter,
                    sort=LogSort.TIMESTAMP_ASC,
                )
                with open(logfile, "a") as f:
                    for log in logs:
                        # avoid duplicates
                        if log.id in seen_ids:
                            continue
                        seen_ids.add(log.id)
                        # strict filtering (keep this)
                        if str(log.flow_run_id) != str(flow_run_id):
                            continue
                        if log.task_run_id is None:
                            ts = log.timestamp.astimezone(local_tz).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            level = (
                                log.level.name
                                if hasattr(log.level, "name")
                                else str(log.level)
                            )
                            f.write(
                                f"{level} | {ts} | {flow_name} | {log.message}\n"
                            )
                        if log.timestamp > last_timestamp:
                            last_timestamp = log.timestamp
            except Exception as e:
                print("LOG ERROR:", e)
                with open(logfile, "a") as f:
                    f.write(f"Error fetching flow logs: {e}\n")
            if should_stop:
                break
            # interruptible sleep
            for _ in range(poll_interval):
                if stop_event and stop_event.is_set():
                    break
                await asyncio.sleep(1)
        try:
            for _ in range(3):  # retry a few times to catch delayed logs
                log_filter = LogFilter(
                    flow_run_id={"any_": [flow_run_id]},
                    timestamp={"after_": last_timestamp},
                )

                logs = await client.read_logs(
                    log_filter=log_filter,
                    sort=LogSort.TIMESTAMP_ASC,
                )

                if not logs:
                    break
                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id in seen_ids:
                            continue
                        seen_ids.add(log.id)
                        if str(log.flow_run_id) != str(flow_run_id):
                            continue
                        if log.task_run_id is None:
                            ts = log.timestamp.astimezone(local_tz).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            level = (
                                log.level.name
                                if hasattr(log.level, "name")
                                else str(log.level)
                            )
                            f.write(
                                f"{level} | {ts} | {flow_name} | {log.message}\n"
                            )
                        if log.timestamp > last_timestamp:
                            last_timestamp = log.timestamp
                await asyncio.sleep(1)
        except Exception as e:
            print("FINAL DRAIN ERROR:", e)


def start_log_task_saver(
    task_run_id, task_name, logfile, poll_interval=5, stop_event=None
):
    """
    Start a background thread that saves Prefect task logs to a file continuously.

    Parameters
    ----------
    task_run_id : str
        The Prefect task run ID to monitor
    task_name : str
        Task name
    logfile : str
        Output log file.
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """

    def run_loop():
        asyncio.run(
            save_logs_by_task_id(
                task_run_id, task_name, logfile, poll_interval, stop_event
            )
        )

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return thread


def start_flow_log_saver(
    flow_run_id, flow_name, logfile, poll_interval=5, stop_event=None
):
    """
    Start a background thread that saves Prefect flow logs to a file continuously.

    Parameters
    ----------
    flow_run_id : str
        The Prefect flow run ID to monitor
    flow_name : str
        Flow name
    logfile : str
        Output log file
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """

    def run_loop():
        asyncio.run(
            save_logs_by_flow_id(
                flow_run_id, flow_name, logfile, poll_interval, stop_event
            )
        )

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return thread
