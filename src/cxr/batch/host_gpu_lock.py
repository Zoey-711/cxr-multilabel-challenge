"""One host-wide GPU queue owner across Phase32 and E08-Tune.

The lock file is intentionally retained after release: unlinking a flock target
can let two processes lock different inodes under the same filename.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def host_gpu_lock(path=None):
    default_dir = Path(tempfile.gettempdir()) if os.name == "nt" else Path("/tmp")
    target = Path(path) if path is not None else default_dir / "cxr_friend_gpu_queue.lock"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as handle:
        if target.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(
                "Another Phase32 or E08-Tune queue owns this host's GPUs; run the batches sequentially"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
