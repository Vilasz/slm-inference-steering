import pytest

from slm_steering.final_extension.resources import accounting


def test_suspension_intersections_do_not_double_count_or_credit_inactive_sessions():
    sessions = [{'ended_at': '2026-11-05T03:00:00+00:00', 'wall_seconds': 10800},
                {'ended_at': '2026-11-05T05:00:00+00:00', 'wall_seconds': 3600}]
    intervals = [{'start_utc': '2026-11-05T01:00:00Z', 'end_utc': '2026-11-05T02:00:00Z'},
                 {'start_utc': '2026-11-05T01:30:00Z', 'end_utc': '2026-11-05T02:30:00Z'},
                 {'start_utc': '2026-11-05T03:00:00Z', 'end_utc': '2026-11-05T04:30:00Z'}]
    result = accounting(sessions, intervals)
    assert result['raw_session_wall_hours'] == 4
    assert result['excluded_os_suspension_hours'] == 2
    assert result['charged_active_session_hours'] == 2


def test_uninterrupted_accounting_preserves_raw_budget():
    sessions = [{'ended_at': '2026-11-05T03:00:00Z', 'wall_seconds': 600}]
    assert accounting(sessions, [])['charged_active_session_hours'] == pytest.approx(1/6)


def test_invalid_or_unzoned_resource_intervals_are_rejected():
    sessions = [{'ended_at': '2026-11-05T03:00:00Z', 'wall_seconds': 600}]
    with pytest.raises(ValueError):
        accounting(sessions, [{'start_utc': '2026-11-05T02:00:00', 'end_utc': '2026-11-05T03:00:00Z'}])
    with pytest.raises(ValueError):
        accounting(sessions, [{'start_utc': '2026-11-05T03:00:00Z', 'end_utc': '2026-11-05T02:00:00Z'}])
