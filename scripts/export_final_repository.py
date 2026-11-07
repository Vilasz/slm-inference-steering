"""Build a curated local publication copy, preserving the research Git state.

Selection follows the active manuscript and its inputs, both scientific hash
manifests, local Python import dependencies, and an explicit final-delivery list.
No commit, push, model operation, or recursive deletion is performed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'final-tcc-2026-11-07'
MANIFEST = 'PUBLICATION_MANIFEST.json'
EXPLICIT = (
    'README.md', 'LICENSE', 'requirements-final.txt', 'RESEARCH_PROTOCOL.md',
    'tcc/README.md', 'tcc/build/manuscrito.pdf',
    'scripts/build_manuscript.py', 'scripts/render_integrated_manuscript.py',
    'scripts/audit_integrated_manuscript.py', 'scripts/finalization_audit.py',
    'scripts/export_final_repository.py', 'scripts/verify_publication_roundtrip.py',
    'docs/FINAL_REPRODUCTION.md', 'docs/FINAL_PUBLICATION.md',
    'docs/execution_amendment_v0.2.md', 'docs/protocols/RESEARCH_PROTOCOL_v0.1.md',
    'docs/protocols/EXCELLENCE_EXTENSION_v1.md', 'docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json',
    'reports/final_extension/freeze.json', 'reports/final_extension/freeze.sha256',
    'reports/final_extension/historical_hashes.json', 'reports/final_extension/results.json',
    'reports/final_extension/external_predictions.csv', 'reports/final_extension/replay_per_task.csv',
    'reports/final_extension/power_simulation.csv', 'reports/final_extension/power_training_sensitivity.csv',
    'reports/final_extension/qwen_training_sensitivity.csv', 'reports/final_extension/qwen_training_sensitivity.json',
    'reports/final_extension/historical_task_structure.csv', 'reports/final_extension/historical_full_task_counts.csv',
    'reports/final_extension/model_inspection.md', 'reports/final_extension/power_analysis.md',
    'reports/final_extension/scientific_audit.md', 'reports/final_extension/historical_audit.json',
    'reports/final_extension/external_bank_audit.json', 'reports/final_extension/model_weight_integrity.json',
    'reports/final_extension/sandbox_preflight.json',
    'reports/manuscript_revision_20261107/manifest.json',
    'reports/manuscript_revision_20261107/validation.json',
    'reports/finalization/CRITICAL_ISSUES.md', 'reports/finalization/STATISTICAL_AUDIT.md',
    'reports/finalization/MATHEMATICAL_AND_BIBLIOGRAPHIC_AUDIT.md',
    'reports/finalization/FINAL_REVIEW.md', 'reports/finalization/statistical_verification.json',
    'reports/finalization/tests.xml', 'reports/finalization/run.json',
)
GENERATED = {
    '.gitignore': ('__pycache__/\n*.py[cod]\n*.egg-info/\n.venv/\nvenv/\n.pytest_cache/\n.ruff_cache/\n'
                   '.mypy_cache/\n.tmp/\ndist/\ntcc/build/preview/\nreports/finalization/preview/\n'
                   'reports/finalization/before/\nreports/manuscript_revision_20261107/preview/\n'),
    '.gitattributes': '# Preserve the bytes covered by scientific SHA256 manifests on every platform.\n* -text\n*.pdf binary\n*.png binary\n*.joblib binary\n',
    'pytest.ini': '[pytest]\npythonpath = src .\naddopts = -p no:cacheprovider\ntestpaths = tests/test_final_extension_resources.py\n',
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select():
    selected = {}
    def add(path, reason):
        path = Path(path)
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing or out-of-workspace publication input: {path}')
        relative = path.relative_to(ROOT).as_posix()
        selected.setdefault(relative, set()).add(reason)
        return path
    for name in EXPLICIT:
        add(name, 'final delivery')
    for name in ('dependency_verification.json', 'publication_verification.json', 'git_preservation.json'):
        path=ROOT/'reports/finalization'/name
        if path.is_file():
            add(path, 'final delivery verification')
    for name in ('freeze.json', 'historical_hashes.json'):
        for path, expected in read(ROOT / 'reports/final_extension' / name)['files'].items():
            source = add(path, 'required by ' + name)
            if sha(source) != expected:
                raise ValueError('Scientific checksum mismatch: ' + path)
    editorial = read(ROOT / 'reports/manuscript_revision_20261107/manifest.json')
    for group in ('inputs', 'generated', 'editorial_sources'):
        for path, expected in editorial[group].items():
            source = add(path, 'active manuscript ' + group)
            if sha(source) != expected:
                raise ValueError('Editorial checksum mismatch; rebuild first: ' + path)
    if sha(ROOT / 'tcc/build/manuscrito.pdf') != editorial['pdf_sha256']:
        raise ValueError('PDF differs from editorial manifest')
    for path in (ROOT / 'reports/finalization/reproduction').glob('*'):
        if path.suffix in ('.csv', '.json'):
            add(path, 'compact statistical reconstruction')
    # Images and .typ imports actually reachable from the manuscript entry point.
    visited = set()
    def manuscript(path):
        path = add(path, 'active manuscript dependency')
        if path in visited:
            return
        visited.add(path)
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'#(?:include|import)\s+"([^"]+\.typ)"', text):
            manuscript(path.parent / target)
        for target in re.findall(r'image\("([^"]+)"', text):
            add(path.parent / target, 'figure used by final PDF')
    manuscript(ROOT / 'tcc/main.typ')
    # Resolve only local slm_steering modules, including relative imports.
    scanned = set()
    def module_path(name):
        base = ROOT / 'src' / Path(*name.split('.'))
        for candidate in (base.with_suffix('.py'), base / '__init__.py'):
            if candidate.is_file():
                return candidate
        return None
    while True:
        pending = [ROOT / p for p in selected if p.endswith('.py') and p not in scanned]
        if not pending:
            break
        for path in pending:
            relative = path.relative_to(ROOT).as_posix()
            scanned.add(relative)
            package = []
            if relative.startswith('src/'):
                parts = path.relative_to(ROOT / 'src').with_suffix('').parts
                package = list(parts[:-1])
                for index in range(1, len(parts)):
                    initializer = ROOT / 'src' / Path(*parts[:index]) / '__init__.py'
                    if initializer.is_file():
                        add(initializer, 'local package initializer')
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    prefix = package[:len(package)-node.level+1] if node.level else []
                    components = prefix + (node.module.split('.') if node.module else [])
                    base = '.'.join(components)
                    names.append(base)
                    names.extend(base + '.' + alias.name for alias in node.names if alias.name != '*')
            for name in names:
                if name == 'slm_steering' or name.startswith('slm_steering.'):
                    target = module_path(name)
                    if target:
                        add(target, 'local Python import dependency')
    return selected


def safe_destination(destination):
    distribution = (ROOT / 'dist').resolve()
    destination = destination.resolve()
    if not distribution.is_relative_to(ROOT) or not destination.is_relative_to(distribution) or destination == distribution:
        raise ValueError('Publication output must be a named directory inside this workspace/dist')
    if (destination / '.git').exists():
        raise ValueError('Refusing to overwrite a separate Git repository')
    return destination


def verify(destination):
    manifest = read(destination / MANIFEST)
    if manifest['profile'] != PROFILE:
        raise ValueError('Unknown publication profile')
    total = 0
    for name, row in manifest['files'].items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or not path.is_file():
            raise ValueError('Missing or unsafe publication path: ' + name)
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Publication content differs: ' + name)
        if row['bytes'] >= 95*1024*1024:
            raise ValueError('File exceeds the curated per-file size limit: ' + name)
        total += row['bytes']
    # Core readme links must resolve within the public copy.
    missing = []
    for filename in ('README.md', 'docs/FINAL_PUBLICATION.md', 'docs/FINAL_REPRODUCTION.md'):
        for link in re.findall(r'\]\(([^)]+)\)', (destination / filename).read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#'):
                if not (destination / filename).parent.joinpath(link.split('#')[0]).exists():
                    missing.append((filename, link))
    if missing:
        raise ValueError('Broken public guide links: ' + str(missing))
    return {'files': len(manifest['files']), 'bytes': total, 'mib': round(total/1024**2, 2),
            'pdfs': [p for p in manifest['files'] if p.endswith('.pdf')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dist/github_final'))
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    destination = safe_destination(ROOT / args.output)
    if args.verify:
        print(json.dumps(verify(destination), ensure_ascii=False))
        return
    previous = None
    if destination.exists():
        if not args.refresh or not (destination / MANIFEST).is_file():
            raise ValueError('Output exists; use --refresh only for a managed publication copy')
        previous = read(destination / MANIFEST)
        if previous['profile'] != PROFILE:
            raise ValueError('Output belongs to another publication profile')
    selected = select()
    destination.mkdir(parents=True, exist_ok=True)
    all_names = set(selected) | set(GENERATED)
    if previous:
        # Only remove stale files explicitly listed in our own prior manifest.
        for name in set(previous['files']) - all_names:
            stale = (destination / name).resolve()
            if not stale.is_relative_to(destination):
                raise ValueError('Unsafe stale publication path')
            if stale.exists():
                if sha(stale) != previous['files'][name]['sha256']:
                    raise ValueError('A stale managed file was edited; preserve it separately: ' + name)
                stale.unlink()
    files = {}
    for name, reasons in sorted(selected.items()):
        source, target = ROOT / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = {'bytes': target.stat().st_size, 'sha256': sha(target), 'reasons': sorted(reasons)}
        if files[name]['sha256'] != sha(source):
            raise ValueError('Source changed while exporting: ' + name)
    for name, content in GENERATED.items():
        target = destination / name
        target.write_bytes(content.encode('utf-8'))
        files[name] = {'bytes': target.stat().st_size, 'sha256': sha(target), 'reasons': ['publication-specific configuration']}
    manifest = {'profile': PROFILE, 'exported_at_utc': datetime.now(timezone.utc).isoformat(),
                'selection': 'Active manuscript, exact hash prerequisites, compact scored-data reconstruction, local source dependencies and explicit final documents',
                'scope': 'No LLM weights, activation banks, old manuscript PDFs, editorial backups or full execution logs',
                'files': files}
    (destination / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = verify(destination)
    if result['pdfs'] != ['tcc/build/manuscrito.pdf']:
        raise ValueError('Publication copy must contain exactly one manuscript PDF')
    print(json.dumps({'destination': str(destination), **result}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
