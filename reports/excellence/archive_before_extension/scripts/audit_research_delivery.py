"""Verify publication inputs and executed notebook after rendering."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.protocol import file_hash, now, write_json


def audit_supplemental(evidence: dict) -> dict:
    extra = evidence.get('supplemental')
    if extra is None:
        return {'available': False}
    if extra['status'] != 'post_hoc_exploratory_frozen_predictions':
        raise ValueError('Supplemental results must remain explicitly post-hoc exploratory')
    for metric, primary_key in [('within_task_auc', 'delta_hb_minus_b'),
                                ('log_loss', 'delta_log_loss_hb_minus_b')]:
        check = extra['primary_reproduction'][metric]
        expected = evidence['main']['signal']['H2'][primary_key]
        effect = check['effect']
        if not check['matches'] or (effect is None) != (expected is None):
            raise ValueError('Supplemental primary reproduction failed')
        if effect is not None:
            values = [effect['estimate'], *effect['ci95']]
            original = [expected['estimate'], *expected['ci95']]
            if any(abs(a - b) > 1e-12 for a, b in zip(values, original)):
                raise ValueError('Supplemental primary effect differs from the frozen analysis')
    ledger = {row['criterion']: row['satisfied'] for row in extra['decision_ledger']}
    if ledger['H2 joint criterion'] != evidence['main']['signal']['H2']['supported']:
        raise ValueError('Supplemental ledger changed the H2 decision')
    if ledger['Joint operational benefit'] != evidence['main']['policies']['H3']['operational_benefit_claim_supported']:
        raise ValueError('Supplemental ledger changed the operational decision')
    if any(evidence['inputs'].get(path) != digest for path, digest in extra['artifacts'].items()):
        raise ValueError('Supplemental artifacts are missing from publication provenance')
    return {'available': True, 'status': extra['status'], 'primary_contrasts_reproduced': True,
            'original_decisions_preserved': True, 'artifacts_verified': len(extra['artifacts'])}


def main():
    from update_execution_summary import update
    update(ROOT)
    report = ROOT / 'reports/final_review'
    evidence = json.loads((report / 'evidence.json').read_text(encoding='utf-8'))
    for relative, expected in evidence['inputs'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f'Publication input changed: {relative}')
    supplemental = audit_supplemental(evidence)
    notebook = nbformat.read(ROOT / 'notebooks/final_research_summary.ipynb', as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    errors = [output for cell in cells for output in cell.outputs if output.output_type == 'error']
    if errors or any(cell.execution_count is None for cell in cells):
        raise ValueError('Publication notebook has errors or unexecuted cells')
    pdf = ROOT / 'tcc/build/manuscrito.pdf'
    if not pdf.read_bytes().startswith(b'%PDF-'):
        raise ValueError('Compiled manuscript is not a PDF')
    pages = None
    try:
        import pymupdf
        with pymupdf.open(pdf) as document:
            pages = document.page_count
            if '@@' in ''.join(page.get_text() for page in document):
                raise ValueError('Unresolved manuscript placeholders')
    except ImportError:
        pass
    test_path = ROOT / 'reports/research/test_results_v0.2.xml'
    tests = [suite.attrib for suite in ET.parse(test_path).getroot().iter('testsuite')] if test_path.exists() else []
    artifacts = [ROOT / 'notebooks/final_research_summary.ipynb', report / 'notebook.html',
                 report / 'evidence.json', pdf, ROOT / 'tcc/referencias.bib', ROOT / 'RESULTS.md']
    result = {'created_at': now(), 'as_of': evidence['as_of'], 'evidence_status': evidence['status'],
              'notebook_code_cells_executed': len(cells), 'notebook_errors': len(errors), 'pdf_pages': pages,
              'checks': 'All publication input hashes match; notebook cells executed; PDF compiled',
              'supplemental': supplemental,
              'tests': tests, 'packages': {name: importlib.metadata.version(name)
                        for name in ('numpy', 'scikit-learn', 'nbformat', 'nbclient', 'nbconvert', 'ipykernel', 'typst')},
              'artifacts': {path.relative_to(ROOT).as_posix(): file_hash(path) for path in artifacts}}
    write_json(report / 'delivery_manifest.json', result)
    print(json.dumps({'passed': True, 'notebook_cells': len(cells), 'pdf_pages': pages,
                      'evidence_status': evidence['status']}, indent=2))


if __name__ == '__main__':
    main()
