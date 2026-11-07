"""Recover pre-approval parser bytes from the preserved initial Git state and patch."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def reconstruct(relative, head, patch_text):
    original = subprocess.check_output(['git', 'show', head + ':' + relative], cwd=ROOT).decode('utf-8')
    marker = f'diff --git a/{relative} b/{relative}\n'
    if marker not in patch_text:
        return original.encode('utf-8')
    section = patch_text.split(marker, 1)[1].split('\ndiff --git ', 1)[0]
    if not section.endswith('\n'):
        section += '\n'
    source = original.splitlines(keepends=True)
    output, cursor, in_hunk = [], 0, False
    for line in section.splitlines(keepends=True):
        match = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', line)
        if match:
            start = int(match[1]) - 1
            output.extend(source[cursor:start])
            cursor, in_hunk = start, True
        elif in_hunk and line.startswith((' ', '-')):
            if source[cursor] != line[1:]:
                raise ValueError(f'Archived patch context does not match {relative} line {cursor + 1}: {source[cursor]!r} versus {line[1:]!r}')
            if line.startswith(' '):
                output.append(line[1:])
            cursor += 1
        elif in_hunk and line.startswith('+'):
            output.append(line[1:])
        elif in_hunk and line.startswith('\\ No newline'):
            if output:
                output[-1] = output[-1].rstrip('\n')
    output.extend(source[cursor:])
    return ''.join(output).encode('utf-8')


if __name__ == '__main__':
    baseline = json.loads((ROOT / 'reports/excellence/baseline.json').read_text(encoding='utf-8'))
    patch_path = ROOT / 'reports/excellence/initial_worktree.patch'
    text = patch_path.read_text(encoding='utf-8')
    directory = ROOT / 'reports/excellence/recovered_preapproval_sources'
    directory.mkdir(exist_ok=True)
    rows = []
    for relative in ('src/slm_steering/verifier.py', 'src/slm_steering/datasets.py'):
        recovered = reconstruct(relative, baseline['git_head'], text)
        current = (ROOT / relative).read_bytes()
        equal = recovered == current
        if not equal:
            raise ValueError(f'Current imported source differs from the archived pre-approval state: {relative}')
        target = directory / (Path(relative).name + '.txt')
        target.write_bytes(recovered)
        rows.append({'path': relative, 'sha256': hashlib.sha256(recovered).hexdigest(),
                     'matches_current_bytes': equal, 'recovered_copy': target.relative_to(ROOT).as_posix()})
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'preapproval_source_recovered_and_matches',
               'baseline_created_at': baseline['created_at'], 'git_head': baseline['git_head'],
               'archived_patch_sha256': hashlib.sha256(patch_path.read_bytes()).hexdigest(), 'sources': rows,
               'scope': 'These imported helpers were not separate entries in the 46-file extension snapshot. Their pre-approval bytes are recoverable from the initial Git state and saved patch; no prospective snapshot entry is fabricated.'}
    (ROOT / 'reports/excellence/parser_provenance_reconstruction.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)
