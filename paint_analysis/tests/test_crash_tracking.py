import json
import os
from pathlib import Path
import subprocess
import sys
import time

from crash_tracking import CrashTracker


def test_snapshot_retains_action_across_heartbeat_and_previous_run(tmp_path):
    tracker = CrashTracker(tmp_path)
    tracker.record('archive_array_started', node=23, bytes=1000)
    tracker.record('heartbeat')
    state = json.loads((tmp_path / 'latest.json').read_text())
    assert state['last_action']['details']['node'] == 23
    assert not state['clean_exit']
    again = CrashTracker(tmp_path)
    previous = json.loads((tmp_path / 'previous-run-latest.json').read_text())
    assert previous['run_id'] == state['run_id']
    assert previous['last_action']['event'] == 'archive_array_started'
    again.close()
    assert json.loads((tmp_path / 'latest.json').read_text())['clean_exit']


def test_logging_failure_does_not_interrupt_work(tmp_path):
    tracker = CrashTracker(tmp_path)
    tracker.snapshot = tmp_path / 'missing' / 'latest.json'
    tracker.record('still_running')


def test_event_history_is_bounded(tmp_path):
    tracker = CrashTracker(tmp_path)
    tracker.events.write_text('x' * (5 * 1024 * 1024 + 1))
    tracker.record('after_rotation')
    assert tracker.events.stat().st_size < 4096
    assert (tmp_path / 'events.previous.jsonl').exists()


def test_hard_kill_preserves_last_action(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
               PAINT_ANALYSIS_HOME=str(tmp_path))
    code = "from crash_tracking import start; import time; t=start(); t.record('loading_test_array', node=42); print('READY', flush=True); time.sleep(60)"
    child = subprocess.Popen([sys.executable, '-u', '-c', code], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        path = tmp_path / 'diagnostics' / 'latest.json'
        while time.monotonic() < deadline:
            if path.exists() and json.loads(path.read_text())['last_action']['event'] == 'loading_test_array':
                break
            assert child.poll() is None
            time.sleep(.05)
        else:
            raise AssertionError('Recorder did not start')
        child.kill()
        child.wait(timeout=5)
        state = json.loads(path.read_text())
        assert state['last_action']['details']['node'] == 42
        assert state['clean_exit'] is False
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def test_uncaught_exception_is_saved(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
               PAINT_ANALYSIS_HOME=str(tmp_path))
    result = subprocess.run([sys.executable, '-c',
        "from crash_tracking import start; start(); raise RuntimeError('diagnostic-test-failure')"],
        env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    state = json.loads((tmp_path / 'diagnostics/latest.json').read_text())
    assert state['last_action']['event'] == 'uncaught_exception'
    assert 'diagnostic-test-failure' in state['last_action']['details']['traceback']
    assert not state['clean_exit']


def test_macos_footprint_is_recorded():
    from crash_tracking import process_memory_metrics
    metrics = process_memory_metrics()
    if sys.platform == 'darwin':
        assert metrics['physical_footprint_bytes'] > 0
        assert metrics['resident_bytes'] > 0
