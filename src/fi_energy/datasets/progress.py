"""Terminal progress reporting for long, chunked downloads."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Self, TextIO

ChunkProgress = Callable[[str, int, int, int], None]
"""Reports ``(stage, completed_chunks, total_chunks, observations)`` after each chunk."""


class ProgressBar:
    """Single-line terminal progress bar with elapsed time, ETA, and detail."""

    def __init__(
        self,
        total: int,
        *,
        description: str,
        stream: TextIO | None = None,
        width: int = 24,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._total = max(total, 0)
        self._description = description
        self._stream = stream or sys.stderr
        self._width = width
        self._enabled = enabled
        self._clock = clock
        self._started = clock()
        self._closed = False
        self._last_width = 0

    def update(self, completed: int, *, detail: str = "") -> None:
        """Renders the bar for an absolute number of completed units."""
        if not self._enabled or self._closed:
            return

        completed = max(0, min(completed, self._total))
        fraction = completed / self._total if self._total else 1.0
        filled = round(fraction * self._width)
        bar = "█" * filled + "░" * (self._width - filled)
        elapsed = self._clock() - self._started

        parts = [
            f"{self._description:<12}",
            f"[{bar}]",
            f"{completed:>3}/{self._total:<3}",
            f"{fraction:>5.1%}",
            _format_seconds(elapsed),
        ]
        if completed >= self._total:
            parts.append("done")
        elif completed:
            remaining = elapsed / completed * (self._total - completed)
            parts.append(f"eta {_format_seconds(remaining)}")
        if detail:
            parts.append(detail)

        line = "\r" + "  ".join(parts)
        padding = max(0, self._last_width - len(line))
        self._stream.write(line + " " * padding)
        self._stream.flush()
        self._last_width = len(line)

    def close(self) -> None:
        if self._enabled and not self._closed:
            self._stream.write("\n")
            self._stream.flush()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DownloadTracker:
    """Status tracker that renders one progress bar per download stage."""

    def __init__(
        self,
        title: str = "",
        *,
        enabled: bool | None = None,
        stream: TextIO | None = None,
        width: int = 24,
    ) -> None:
        self._stream = stream or sys.stderr
        self._enabled = self._stream.isatty() if enabled is None else enabled
        self._title = title
        self._width = width
        self._bars: dict[str, ProgressBar] = {}
        self._title_written = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stage(
        self, name: str, completed: int, total: int, observations: int = 0
    ) -> None:
        """Updates (and lazily creates) the bar for ``name``."""
        if not self._enabled:
            return

        self._write_title()
        bar = self._bars.get(name)
        if bar is None:
            bar = self._bars[name] = ProgressBar(
                total,
                description=name,
                stream=self._stream,
                width=self._width,
            )
        detail = f"rows {observations:,}" if observations else ""
        bar.update(completed, detail=detail)

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _write_title(self) -> None:
        if self._title and not self._title_written:
            self._stream.write(self._title + "\n")
            self._stream.flush()
            self._title_written = True


def _format_seconds(seconds: float) -> str:
    total = int(seconds)
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"