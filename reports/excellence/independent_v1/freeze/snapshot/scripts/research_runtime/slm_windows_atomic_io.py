"""Retry transient Windows sharing violations during atomic JSON publication.

This runtime layer leaves the frozen experiment sources and JSON bytes intact.
It applies only to .json.tmp replacements within this repository.
"""
from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_REPLACE = os.replace


@wraps(ORIGINAL_REPLACE)
def replace_with_retry(src, dst, *args, _replace=ORIGINAL_REPLACE, _root=ROOT, _timeout=10.0, **kwargs):
    source = Path(os.fsdecode(src))
    destination = Path(os.fsdecode(dst))
    scoped = (source.name.endswith('.json.tmp') and destination.suffix == '.json'
              and source.resolve().is_relative_to(_root)
              and destination.resolve().is_relative_to(_root))
    deadline = time.monotonic() + _timeout
    delay = 0.01
    while True:
        try:
            return _replace(src, dst, *args, **kwargs)
        except PermissionError as exc:
            if not scoped or getattr(exc, 'winerror', None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(0.25, delay * 2)


def install():
    if os.name == 'nt' and os.replace is not replace_with_retry:
        os.replace = replace_with_retry
