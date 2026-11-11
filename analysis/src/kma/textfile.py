"""Atomic writes into node-exporter's textfile directory.

The temp file sits in the same directory (a rename across filesystems is not
atomic) and does not end in `.prom`, which is the only suffix node-exporter's
textfile collector reads, so a half-written file is never scraped. Mode 0644
because `mkstemp` makes 0600 and the exporter often runs as another user
inside a container.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ENV = "KMA_TEXTFILE_DIR"


def gauges(samples: list[tuple[str, dict[str, str], float]]) -> str:
    """Exposition text for gauge samples `(metric, labels, value)`, one TYPE
    line per metric."""
    lines, typed = [], set()
    for metric, labels, value in samples:
        if metric not in typed:
            lines.append(f"# TYPE {metric} gauge")
            typed.add(metric)
        body = ",".join(f'{k}="{v}"' for k, v in labels.items())
        lines.append(f"{metric}{{{body}}} {value:g}" if body else f"{metric} {value:g}")
    return "\n".join(lines) + "\n"


def write(directory: Path, filename: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{filename}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target
