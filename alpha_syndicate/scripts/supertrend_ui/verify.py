"""Runs the live-Engine replay (supertrend_ui.replay) in a CHILD PROCESS and returns its result.

The replay swaps module-level state of supertrend_gui (clock, state file, config) - harmless in a throw-away process, destructive inside the app while a live session runs.
So the GUI never calls replay_live() itself; it calls run_replay_process()."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Callable

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_replay_process(values: dict, start: str, end: str, shortable: bool, lookahead: bool = False, log: Callable[[str], None] = lambda m: None,
                       progress: Callable[[int, int], None] = lambda i, n: None, cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Start ``python -m supertrend_ui.replay --json`` and stream its progress. Returns the parsed RESULT dict; raises RuntimeError with the child's last lines on failure."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="st_verify_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(values, fh)
    cmd = [sys.executable, "-m", "supertrend_ui.replay", "--json", "--start", start, "--end", end, "--settings", path]
    if not shortable:
        cmd.append("--no-shortable")
    if lookahead:
        cmd.append("--lookahead")
    tail: list = []
    result = None
    proc = subprocess.Popen(cmd, cwd=SCRIPTS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONUTF8": "1"})
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if cancelled():
                proc.terminate()
                raise RuntimeError("cancelled")
            if line.startswith("PROGRESS "):
                try:
                    _, i, n = line.split()
                    progress(int(i), int(n))
                except ValueError:
                    pass
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
            else:
                tail.append(line); del tail[:-25]
                log(line)
        code = proc.wait()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        if proc.poll() is None:
            proc.kill()
    if result is None:
        raise RuntimeError(f"the replay process ended without a result (exit code {code}):\n" + "\n".join(tail[-12:]))
    return result
