from __future__ import annotations

from io import StringIO

from datasets.progress import DownloadTracker, ProgressBar


class FakeTty(StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_progress_bar_renders_counts_percent_and_completion():
    stream = FakeTty(tty=True)
    bar = ProgressBar(
        4,
        description="weather",
        stream=stream,
        enabled=True,
        clock=lambda: 10.0,
    )

    bar.update(1)
    bar.update(4, detail="rows 744")
    bar.close()

    output = stream.getvalue()
    assert "\rweather" in output
    assert " 1/4 " in output
    assert "25.0%" in output
    assert " 4/4 " in output
    assert "100.0%" in output
    assert "done" in output
    assert "rows 744" in output
    assert output.endswith("\n")


def test_progress_bar_shows_eta_after_the_first_chunk():
    stream = FakeTty(tty=True)
    clock = iter([0.0, 10.0, 20.0])
    bar = ProgressBar(
        4,
        description="weather",
        stream=stream,
        enabled=True,
        clock=lambda: next(clock),
    )

    bar.update(1)
    bar.update(2)

    assert "eta 00:30" in stream.getvalue()


def test_disabled_progress_bar_writes_nothing():
    stream = FakeTty(tty=True)
    bar = ProgressBar(3, description="weather", stream=stream, enabled=False)

    bar.update(1)
    bar.update(3)
    bar.close()

    assert stream.getvalue() == ""


def test_download_tracker_writes_title_once_and_one_bar_per_stage():
    stream = FakeTty(tty=True)
    tracker = DownloadTracker("Helsinki: FMI", enabled=True, stream=stream)

    tracker.stage("weather", 0, 2, 0)
    tracker.stage("weather", 2, 2, 1_488)
    tracker.stage("radiation", 1, 3, 168)
    tracker.close()

    output = stream.getvalue()
    assert output.count("Helsinki: FMI") == 1
    assert "weather" in output
    assert "radiation" in output
    assert "rows 1,488" in output
    assert output.endswith("\n")


def test_download_tracker_auto_disables_on_non_tty_streams():
    stream = FakeTty(tty=False)
    tracker = DownloadTracker("Helsinki: FMI", stream=stream)

    assert tracker.enabled is False
    tracker.stage("weather", 1, 2, 100)
    tracker.close()

    assert stream.getvalue() == ""


def test_download_tracker_auto_enables_on_tty_streams():
    stream = FakeTty(tty=True)
    tracker = DownloadTracker("Helsinki: FMI", stream=stream)

    assert tracker.enabled is True
    tracker.stage("weather", 1, 2, 100)
    tracker.close()

    assert "Helsinki: FMI" in stream.getvalue()