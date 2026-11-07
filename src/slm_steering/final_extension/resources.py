"""Audited resource accounting; raw timings and scientific records stay immutable."""
from __future__ import annotations

from datetime import datetime, timedelta

from slm_steering.final_extension.common import REPORT, RUN, file_hash, read_json


def utc_time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Resource timestamps require an explicit timezone')
    return parsed


def merged_intervals(intervals):
    ordered = sorted((utc_time(v['start_utc']), utc_time(v['end_utc'])) for v in intervals)
    merged = []
    for start, end in ordered:
        if end <= start:
            raise ValueError('Invalid suspension interval')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def suspension_overlap(start, end, intervals):
    return sum(max(0., (min(end, right) - max(start, left)).total_seconds())
               for left, right in merged_intervals(intervals))


def accounting(sessions, intervals):
    raw, excluded = 0., 0.
    for session in sessions:
        seconds = session['wall_seconds']
        if seconds < 0:
            raise ValueError('Negative session duration')
        end = utc_time(session['ended_at'])
        start = end - timedelta(seconds=seconds)
        raw += seconds
        excluded += suspension_overlap(start, end, intervals)
    return {'raw_session_wall_hours': raw / 3600,
            'excluded_os_suspension_hours': excluded / 3600,
            'charged_active_session_hours': max(0., raw-excluded) / 3600,
            'scope': 'Active session wall time including loading/verification; not CUDA kernel profiling'}


def usage():
    correction = REPORT / 'resource_accounting_adjustments.json'
    intervals = []
    if correction.exists():
        value = read_json(correction)
        evidence = REPORT / value['evidence_file']
        if file_hash(evidence) != value['evidence_sha256']:
            raise ValueError('OS suspension evidence changed')
        intervals = value['intervals']
    sessions = [s for p in RUN.glob('*/*/manifest.json') for s in read_json(p).get('sessions', [])]
    return accounting(sessions, intervals)
