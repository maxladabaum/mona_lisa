"""Local, bounded flight recorder. Never serializes localization data."""
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
import faulthandler
import json
import os
import shutil
import sys
import threading
import traceback
import uuid


@lru_cache(maxsize=1)
def _macos_memory_query():
    import ctypes
    # struct rusage_info_v0, from the macOS SDK sys/resource.h.
    class Usage(ctypes.Structure):
        _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in (
                'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups',
                'pageins', 'wired_size', 'resident_size', 'phys_footprint',
                'proc_start_abstime', 'proc_exit_abstime')]
    library = ctypes.CDLL('/usr/lib/libproc.dylib')
    query = library.proc_pid_rusage
    query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    query.restype = ctypes.c_int
    return Usage, query


def process_memory_metrics():
    """Use macOS footprint accounting as well as resident memory."""
    metrics = {}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics['peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
        if sys.platform == 'darwin':
            import ctypes
            Usage, query = _macos_memory_query()
            usage = Usage()
            if query(os.getpid(), 0, ctypes.byref(usage)) == 0:
                metrics.update(resident_bytes=usage.resident_size,
                               physical_footprint_bytes=usage.phys_footprint)
    except (ImportError, OSError, AttributeError):
        pass
    return metrics

_tracker = None


def record(event, **details):
    if _tracker is not None:
        _tracker.record(event, **details)


class CrashTracker:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.state = dict(pid=os.getpid(), run_id=uuid.uuid4().hex, clean_exit=False)
        self.events = self.directory / 'events.jsonl'
        self.snapshot = self.directory / 'latest.json'
        # Preserve the previous run before writing any new startup evidence.
        for name in ('latest.json', 'events.jsonl', 'events.previous.jsonl', 'stacks.txt', 'fatal.txt'):
            path = self.directory / name
            if path.exists():
                shutil.copyfile(path, self.directory / ('previous-run-' + name))
        for name in ('events.previous.jsonl', 'stacks.txt', 'fatal.txt'):
            (self.directory / name).unlink(missing_ok=True)
        self.events.write_text('')
        self.record('startup', python=sys.version, executable=sys.executable)

    def record(self, event, **details):
        try:
            with self.lock:
                now = datetime.now(timezone.utc).isoformat()
                metrics = {'disk_free_bytes': shutil.disk_usage(self.directory).free}
                metrics.update(process_memory_metrics())
                entry = dict(time=now, event=event, thread=threading.current_thread().name,
                             details=details, **metrics)
                if event != 'heartbeat':
                    self.state['last_action'] = entry
                self.state.update(updated=now, **metrics)
                if self.events.exists() and self.events.stat().st_size > 5 * 1024 * 1024:
                    os.replace(self.events, self.directory / 'events.previous.jsonl')
                with self.events.open('a') as stream:
                    stream.write(json.dumps(entry, default=str) + '\n')
                    stream.flush()
                temporary = self.snapshot.with_suffix('.tmp')
                with temporary.open('w') as stream:
                    json.dump(self.state, stream, default=str, indent=2)
                    stream.flush()
                os.replace(temporary, self.snapshot)
        except Exception:
            # A full or disconnected disk must not bring down the application.
            pass

    def install(self):
        self.fatal = (self.directory / 'fatal.txt').open('w')
        faulthandler.enable(file=self.fatal, all_threads=True)
        original = sys.excepthook
        def exception(kind, value, tb):
            self.record('uncaught_exception', traceback=''.join(traceback.format_exception(kind, value, tb)))
            original(kind, value, tb)
        sys.excepthook = exception
        original_thread = threading.excepthook
        def thread_exception(args):
            self.record('thread_exception', traceback=''.join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback)))
            original_thread(args)
        threading.excepthook = thread_exception
        threading.Thread(target=self.watch, name='crash-recorder', daemon=True).start()

    def watch(self):
        while not self.stop.wait(10):
            self.record('heartbeat')
            try:
                temporary = self.directory / 'stacks.tmp'
                with temporary.open('w') as stream:
                    faulthandler.dump_traceback(file=stream, all_threads=True)
                os.replace(temporary, self.directory / 'stacks.txt')
            except Exception:
                pass

    def close(self):
        self.stop.set()
        self.state['clean_exit'] = True
        self.record('clean_exit')


def start():
    global _tracker
    default = (Path.home() / 'Library/Application Support/PaintAnalysis' if sys.platform == 'darwin'
               else Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'PaintAnalysis')
    try:
        _tracker = CrashTracker(Path(os.environ.get('PAINT_ANALYSIS_HOME', default)) / 'diagnostics')
        _tracker.install()
        print(f'Crash diagnostics: {_tracker.directory}', flush=True)
    except Exception as exc:
        print(f'Could not start crash diagnostics: {exc}', file=sys.stderr)
    return _tracker
