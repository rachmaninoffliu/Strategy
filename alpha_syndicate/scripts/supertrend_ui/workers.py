"""Background work for the window: nothing that touches the network or simulates 70,000 bars runs on the GUI thread."""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


class FnWorker(QThread):
    """Runs fn(log) in a thread; emits log lines, progress, and finally the result or the error text. ``cancelled()`` lets fn stop early when asked."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object, object)           # result, error-text

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            self.done.emit(self.fn(self.log.emit), None)
        except Exception as e:                    # noqa: BLE001
            self.done.emit(None, f"{type(e).__name__}: {e}")


class WorkerBag:
    """Keeps a reference to running workers (a QThread that is garbage-collected while running crashes the process) and drops finished ones."""

    def __init__(self):
        self._w: list = []

    def start(self, w: FnWorker, on_done=None, on_log=None, on_progress=None) -> FnWorker:
        if on_done:
            w.done.connect(on_done)
        if on_log:
            w.log.connect(on_log)
        if on_progress:
            w.progress.connect(on_progress)
        w.finished.connect(lambda w=w: self._w.remove(w) if w in self._w else None)
        self._w.append(w)
        w.start()
        return w

    def busy(self) -> bool:
        return any(w.isRunning() for w in self._w)

    def stop_all(self, wait_ms: int = 3000) -> None:
        for w in list(self._w):
            w.cancel()
            w.wait(wait_ms)
