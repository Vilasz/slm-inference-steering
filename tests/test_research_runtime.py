from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading

import pytest

from scripts.research_runtime.slm_windows_atomic_io import replace_with_retry


def sharing_error():
    error = PermissionError('File held open without delete sharing')
    error.winerror = 32
    return error


def test_atomic_json_retry_preserves_bytes_after_transient_lock(tmp_path):
    source, target = tmp_path / 'manifest.json.tmp', tmp_path / 'manifest.json'
    source.write_bytes(b'{"complete":2433}\n')
    target.write_bytes(b'{"complete":2432}\n')
    calls = []

    def transient_replace(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise sharing_error()
        return os.replace(src, dst)

    replace_with_retry(source, target, _replace=transient_replace, _root=tmp_path)
    assert len(calls) == 2
    assert target.read_bytes() == b'{"complete":2433}\n'
    assert not source.exists()


def test_retry_is_bounded_and_keeps_previous_file_on_permanent_lock(tmp_path):
    source, target = tmp_path / 'manifest.json.tmp', tmp_path / 'manifest.json'
    source.write_bytes(b'new')
    target.write_bytes(b'previous')

    def locked(*args):
        raise sharing_error()

    with pytest.raises(PermissionError):
        replace_with_retry(source, target, _replace=locked, _root=tmp_path, _timeout=0)
    assert source.read_bytes() == b'new' and target.read_bytes() == b'previous'


def test_unrelated_files_are_not_retried(tmp_path):
    calls = []

    def locked(*args):
        calls.append(args)
        raise sharing_error()

    with pytest.raises(PermissionError):
        replace_with_retry(tmp_path / 'weights.tmp', tmp_path / 'weights.bin', _replace=locked, _root=tmp_path)
    assert len(calls) == 1


@pytest.mark.skipif(os.name != 'nt', reason='Uses real Windows file-sharing semantics')
def test_atomic_replace_waits_for_real_windows_reader(tmp_path):
    from ctypes import wintypes
    source, target = tmp_path / 'manifest.json.tmp', tmp_path / 'manifest.json'
    source.write_bytes(b'new')
    target.write_bytes(b'previous')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(Path(target).resolve()), 0x80000000, 3, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    try:
        with pytest.raises(PermissionError):
            os.replace(source, target)
    finally:
        timer = threading.Timer(0.15, lambda: kernel.CloseHandle(handle))
        timer.start()
    try:
        replace_with_retry(source, target, _root=tmp_path)
    finally:
        timer.join()
    assert target.read_bytes() == b'new' and not source.exists()
