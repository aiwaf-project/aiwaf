"""
Shared CSV storage helpers (cross-platform).
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from threading import Lock
from pathlib import Path

# Cross-platform file locking imports
try:
    import fcntl  # Unix/Linux/macOS
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

try:
    import msvcrt  # Windows
    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False


logger = logging.getLogger(__name__)

_THREAD_LOCKS: dict[str, Lock] = {}


def _get_thread_lock(path: str) -> Lock:
    lock = _THREAD_LOCKS.get(path)
    if lock is None:
        lock = Lock()
        _THREAD_LOCKS[path] = lock
    return lock


@contextmanager
def file_lock(file_path, mode="r", max_retries=3, retry_delay=0.1):
    """Cross-platform file locking context manager."""
    file_obj = None
    lock_acquired = False
    thread_lock = _get_thread_lock(str(file_path))

    def _open_with_retries(path, open_mode):
        last_exc = None
        for attempt in range(max_retries):
            try:
                return open(path, open_mode, newline='' if 'b' not in open_mode else None)
            except (PermissionError, OSError) as e:
                last_exc = e
                time.sleep(retry_delay * (attempt + 1))
        if last_exc:
            raise last_exc

    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        thread_lock.acquire()
        file_obj = _open_with_retries(file_path, mode)

        if FCNTL_AVAILABLE and file_obj:
            try:
                if 'w' in mode or 'a' in mode:
                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                lock_acquired = True
            except (IOError, OSError):
                lock_acquired = False
        elif MSVCRT_AVAILABLE:
            # Windows locking is unreliable across threads/processes; rely on thread locks.
            lock_acquired = False

        yield file_obj
    finally:
        if file_obj:
            try:
                if lock_acquired and FCNTL_AVAILABLE:
                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
                file_obj.close()
            except Exception:
                pass
        try:
            thread_lock.release()
        except Exception:
            pass


def safe_csv_operation(operation, *args, max_retries=5, base_delay=0.01, **kwargs):
    """Safely perform CSV operation with retry logic and exponential backoff."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except (IOError, OSError, PermissionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.01)
                time.sleep(delay)
                logger.debug(
                    "CSV operation retry %s/%s after %.3fs delay",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                continue
            logger.error("CSV operation failed after %s attempts: %s", max_retries, e)
            break
        except Exception as e:
            logger.error("Unexpected error in CSV operation: %s", e)
            last_exception = e
            break
    if last_exception:
        raise last_exception
