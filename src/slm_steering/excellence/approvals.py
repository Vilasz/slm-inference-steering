"""Standard-library-only approval check, safe before loading CUDA or scientific DLLs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def approval_hashes(root: Path):
    files = {'protocol_sha256': 'docs/protocols/EXCELLENCE_EXTENSION_v1.md',
             'design_sha256': 'reports/excellence/proposal_design.json',
             'collector_sha256': 'src/slm_steering/excellence/independent.py',
             'capture_sha256': 'src/slm_steering/excellence/benchmark.py',
             'inference_config_sha256': 'configs/protocol_v0.2.yaml',
             'bundle_sha256': 'runs/transfer/protocol_v0.2/models.joblib'}
    return {key: hashlib.sha256((root / path).read_bytes()).hexdigest() for key, path in files.items()}


def approved_design(root: Path, *, require_approval=True):
    design = json.loads((root / 'reports/excellence/proposal_design.json').read_text(encoding='utf-8'))
    if require_approval:
        path = root / 'docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json'
        if not path.exists():
            raise PermissionError('Independent collection requires recorded student approval and advisor review of the written protocol')
        approval = json.loads(path.read_text(encoding='utf-8'))
        if approval.get('student_approved') is not True or approval.get('advisor_review_completed') is not True:
            raise PermissionError('Student approval and advisor review remain required before scientific freeze')
        if any(approval.get(key) != value for key, value in approval_hashes(root).items()):
            raise ValueError('Approved protocol/design/collector differs; amendment and review required')
    return design
