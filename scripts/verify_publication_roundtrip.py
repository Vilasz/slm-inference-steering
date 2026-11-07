"""Check a curated copy through an isolated Git tree/archive and optional CPU smoke runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid

from export_final_repository import ROOT, MANIFEST, safe_destination, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, default=Path('dist/github_final'))
    parser.add_argument('--smoke', action='store_true', help='Run numerical audit, manuscript build/audit and light tests in the extracted copy')
    parser.add_argument('--python', type=Path, help='Python interpreter for smoke runs, for example a separate minimal environment')
    args = parser.parse_args()
    interpreter = str(args.python.resolve()) if args.python else sys.executable
    if not Path(interpreter).is_file():
        raise ValueError('Smoke Python interpreter not found')
    research_index = ROOT / '.git/index'
    index_before = sha(research_index) if research_index.is_file() else None
    publication = safe_destination(ROOT / args.publication)
    inventory = verify(publication)
    record = json.loads((publication / MANIFEST).read_text(encoding='utf-8'))
    checkdir = ROOT / 'dist/publication_checks' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    checkdir.mkdir(parents=True)
    gitdir = checkdir / 'repository.git'
    subprocess.run(['git', 'init', '--bare', str(gitdir)], check=True, capture_output=True)
    prefix = ['git', '-c', 'core.autocrlf=true', '-c', 'core.safecrlf=false',
              '--git-dir=' + str(gitdir), '--work-tree=' + str(publication)]
    subprocess.run([*prefix, 'add', '--all'], cwd=publication, check=True, capture_output=True)
    staged = set(subprocess.check_output([*prefix, 'ls-files'], cwd=publication, text=True).splitlines())
    if staged != set(record['files']) | {MANIFEST}:
        raise ValueError('Staged publication set differs from the curated manifest')
    tree = subprocess.check_output([*prefix, 'write-tree'], cwd=publication, text=True).strip()
    archive = checkdir / 'roundtrip.tar'
    with archive.open('wb') as stream:
        subprocess.run([*prefix, 'archive', '--format=tar', tree], cwd=publication, check=True, stdout=stream, stderr=subprocess.PIPE)
    extracted = checkdir / 'extracted'
    extracted.mkdir()
    with tarfile.open(archive) as bundle:
        for entry in bundle.getmembers():
            if not (extracted / entry.name).resolve().is_relative_to(extracted.resolve()) or entry.issym() or entry.islnk():
                raise ValueError('Unsafe archive member')
        bundle.extractall(extracted, filter='data')
    roundtrip = verify(extracted)
    if sha(extracted / MANIFEST) != sha(publication / MANIFEST):
        raise ValueError('Manifest bytes changed through Git')
    commands = []
    if args.smoke:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        # Verify that the public execution path does not rely on installed model packages.
        wrapper = """import importlib.abc, pathlib, runpy, sys
class ModelPackageBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','datasets','accelerate','huggingface_hub'}:
            raise ImportError('Model packages intentionally unavailable in the CPU publication check: '+fullname)
sys.meta_path.insert(0,ModelPackageBlocker())
sys.argv=sys.argv[1:]
sys.path.insert(0,str(pathlib.Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        for script, arguments in [('scripts/finalization_audit.py', []),
                                  ('scripts/build_manuscript.py', ['--compile']),
                                  ('scripts/audit_integrated_manuscript.py', [])]:
            start = time.monotonic()
            process = subprocess.run([interpreter, '-X', 'utf8', '-u', '-c', wrapper, script, *arguments],
                                     cwd=extracted, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=300)
            row = {'command': [script, *arguments], 'seconds': time.monotonic()-start,
                   'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
            commands.append(row)
            print(script + ': ' + str(process.returncode), flush=True)
            if process.returncode:
                (checkdir / 'smoke_failure.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
                raise RuntimeError(process.stderr or process.stdout)
        process = subprocess.run([interpreter, '-m', 'pytest', '-q', 'tests/test_final_extension_resources.py'],
                                 cwd=extracted, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=120)
        commands.append({'command': ['pytest', 'tests/test_final_extension_resources.py'],
                         'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        if process.returncode:
            raise RuntimeError(process.stderr or process.stdout)
    result = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'publication': 'dist/github_final',
              'inventory': inventory, 'git_tree': tree, 'git_autocrlf_tested': True,
              'git_roundtrip_hashes_match': roundtrip == inventory,
              'original_git_index_untouched': True, 'smoke_commands': commands,
              'model_packages_blocked_for_script_smoke': args.smoke,
              'smoke_python': interpreter,
              'script_hashes': {p: sha(publication / p) for p in record['files'] if p.startswith('scripts/') and p.endswith('.py')},
              'check_directory': checkdir.relative_to(ROOT).as_posix(),
              'scope': 'CPU execution from a Git-extracted copy with model packages blocked; interpreter and dependency installation are recorded separately; no fresh model inference'}
    if index_before is not None and sha(research_index) != index_before:
        raise ValueError('Research Git index changed during the publication check')
    (ROOT / 'reports/finalization/publication_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'git_roundtrip': 'passed', 'smoke': bool(commands), **inventory}, ensure_ascii=False))


if __name__ == '__main__':
    main()
