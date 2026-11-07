from __future__ import annotations

import json

import pytest

from slm_steering.excellence.approvals import approval_hashes, approved_design


def proposal(tmp_path):
    for relative in ['reports/excellence/proposal_design.json', 'docs/protocols/EXCELLENCE_EXTENSION_v1.md',
                     'src/slm_steering/excellence/independent.py', 'src/slm_steering/excellence/benchmark.py',
                     'configs/protocol_v0.2.yaml', 'runs/transfer/protocol_v0.2/models.joblib']:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}' if relative.endswith('.json') else 'fixture', encoding='utf-8')
    return tmp_path / 'docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json'


def test_draft_cannot_start_independent_collection(tmp_path):
    approval_path = proposal(tmp_path)
    assert approved_design(tmp_path, require_approval=False) == {}
    with pytest.raises(PermissionError, match='student approval'):
        approved_design(tmp_path)
    approval_path.write_text(json.dumps({'student_approved': True, 'advisor_review_completed': False}))
    with pytest.raises(PermissionError, match='advisor review'):
        approved_design(tmp_path)


def test_approved_version_is_bound_to_design_code_bundle_and_configuration(tmp_path):
    approval_path = proposal(tmp_path)
    approval_path.write_text(json.dumps({'student_approved': True, 'advisor_review_completed': True,
                                          **approval_hashes(tmp_path)}))
    assert approved_design(tmp_path) == {}
    (tmp_path / 'docs/protocols/EXCELLENCE_EXTENSION_v1.md').write_text('changed scientific criterion')
    with pytest.raises(ValueError, match='differs'):
        approved_design(tmp_path)


def test_new_endpoint_uses_paired_common_counts_and_handles_no_positives():
    from slm_steering.excellence.independent import paired_retention_result
    result = paired_retention_result([[0, 3, 3], [0, 2, 2]], replications=100)
    assert result['estimate'] == -1 and result['ci95'] == [-1, -1]
    assert result['directional_hypothesis_supported']
    missing = paired_retention_result([[0, 0, 0]], replications=100)
    assert not missing['identified'] and missing['estimate'] is None
    assert not missing['directional_hypothesis_supported']
