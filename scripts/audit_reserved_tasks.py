"""Audit the approved reservation without labels, exclusions, or any model generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.excellence.approvals import approved_design
from slm_steering.research.evaluation import wsl_path
from slm_steering.research.protocol import file_hash, now, read_bank, write_json


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wsl-python', default='/home/joaof/.venvs/slm-evalplus-v031/bin/python')
    args = parser.parse_args()
    design = approved_design(ROOT)
    output = ROOT / 'reports/excellence'
    command = ['wsl', '-d', 'Ubuntu', '--', args.wsl_python, '-u',
               wsl_path(ROOT / 'scripts/audit_reserved_tasks_worker.py')]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', timeout=90)
    contents = json.loads(result.stdout)
    if contents['metadata']['dataset_sha256'] != design['benchmark_sha256']:
        raise ValueError('Public task contents differ from the approved dataset')
    write_json(output / 'reserved_task_contents.json', contents)
    exposure = json.loads((output / 'target_exposure_audit.json').read_text(encoding='utf-8'))
    observed = set(exposure['observed_task_ids'])
    technical, main_ids = set(design['technical_pilot_task_ids']), set(design['main_task_ids'])
    if technical & main_ids or (technical | main_ids) & observed:
        raise ValueError('Reservation overlaps old tasks or the technical pilot')
    rows = []
    for row in contents['tasks']:
        group = 'main' if row['task_id'] in main_ids else 'technical' if row['task_id'] in technical else 'previous_mbpp' if row['task_id'] in observed else 'unused_mbpp'
        rows.append({**row, 'group': group})
    seen = set()
    for phase in ('pilot', 'definitive'):
        for row in read_bank(ROOT / f'runs/candidate_bank/{phase}_v0.2/trajectories.jsonl'):
            if row['task_id'] in seen:
                continue
            seen.add(row['task_id'])
            rows.append({'task_id': row['task_id'], 'prompt': row['prompt'], 'group': 'previous_humaneval'})
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True)
    similarities = cosine_similarity(vectorizer.fit_transform([r['prompt'] for r in rows]))
    pairs, duplicates = [], []
    for i, left in enumerate(rows):
        if left['group'] not in ('main', 'technical'):
            continue
        for j in similarities[i].argsort()[::-1]:
            if i == j or rows[j]['group'] == 'unused_mbpp':
                continue
            right = rows[j]
            score = float(similarities[i, j])
            same_prompt = left.get('prompt_sha256') == right.get('prompt_sha256') if right.get('prompt_sha256') else left['prompt'] == right['prompt']
            same_structure = bool(left.get('canonical_structure_sha256') and left.get('canonical_structure_sha256') == right.get('canonical_structure_sha256'))
            row = {'task_id': left['task_id'], 'group': left['group'], 'neighbor': right['task_id'],
                   'neighbor_group': right['group'], 'character_tfidf_cosine': score,
                   'exact_prompt': same_prompt, 'same_normalized_canonical_structure': same_structure}
            if score >= .70 or same_prompt or same_structure:
                pairs.append(row)
            if same_prompt or same_structure:
                duplicates.append(row)
    # The signature audit covers all relevant pairs, including those whose prose is dissimilar.
    for i, left in enumerate(rows):
        if left['group'] not in ('main', 'technical'):
            continue
        for j, right in enumerate(rows):
            if i == j or right['group'] == 'unused_mbpp':
                continue
            structural = left.get('canonical_structure_sha256')
            if structural and structural == right.get('canonical_structure_sha256'):
                key = (left['task_id'], right['task_id'])
                if not any((r['task_id'], r['neighbor']) == key for r in duplicates):
                    duplicates.append({'task_id': left['task_id'], 'group': left['group'], 'neighbor': right['task_id'],
                        'neighbor_group': right['group'], 'character_tfidf_cosine': float(similarities[i, j]),
                        'exact_prompt': left['prompt_sha256'] == right.get('prompt_sha256'),
                        'same_normalized_canonical_structure': True})
    current = json.loads((output / 'independent_v1/execution/status.json').read_text(encoding='utf-8'))
    report = {'created_at': now(), 'status': 'completed_label_blind_post_start_diagnostic',
              'timing': 'Performed after collection began under the already approved reservation; no exclusions or changes to the endpoint',
              'collection_phase_at_audit': current['phase'], 'dataset': contents['metadata'],
              'technical_tasks': len(technical), 'main_tasks': len(main_ids), 'id_sets_disjoint': True,
              'contents_versioned_by_sha256': True, 'similar_pairs_ge_070': pairs, 'prompt_or_structure_duplicates': duplicates,
              'method': 'Character-within-word TF-IDF 3–5 grams over public prompts; normalized canonical AST is a syntax heuristic, not semantic equivalence',
              'limitations': ['Cannot certify absence of pretraining exposure or all semantic overlap',
                             'Shared templates and simple algorithms can inflate resemblance',
                             'No task was removed; the predeclared task-bootstrap result remains the primary analysis'],
              'inputs': {p: file_hash(ROOT / p) for p in ['reports/excellence/proposal_design.json',
                  'reports/excellence/target_exposure_audit.json', 'reports/excellence/reserved_task_contents.json',
                  'scripts/audit_reserved_tasks.py', 'scripts/audit_reserved_tasks_worker.py']}}
    write_json(output / 'reserved_task_audit.json', report)
    print(json.dumps({'main_tasks': len(main_ids), 'technical_tasks': len(technical), 'disjoint': True,
                      'similar_pairs': len(pairs), 'prompt_or_structure_matches': len(duplicates),
                      'matches': duplicates[:8]}), flush=True)


if __name__ == '__main__':
    main()
