"""Small, hardware-independent helpers for bounded controller waits."""

import time
from typing import Callable


def wait_for_change(
    read_actual: Callable[[], float],
    previous_value: float,
    *,
    threshold: float,
    timeout: float = 5.0,
    poll_interval: float = 0.02,
) -> None:
    """Wait until a readback changes by ``threshold`` or raise on timeout."""

    deadline = time.monotonic() + timeout
    while abs(read_actual() - previous_value) < threshold:
        if time.monotonic() >= deadline:
            raise TimeoutError("Laser controller did not report the requested change")
        time.sleep(poll_interval)
