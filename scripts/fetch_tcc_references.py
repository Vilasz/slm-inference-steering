"""Rebuild bibliographic metadata from the primary pages listed in the review.

No generated scientific text is downloaded or inserted into the manuscript.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[1]
ARXIV = {
    'chen2021codex': '2107.03374', 'brown2024': '2407.21787', 'liu2023evalplus': '2305.01210',
    'kadavath2022': '2207.05221', 'azaria2023': '2304.13734', 'zou2023': '2310.01405',
    'hui2024': '2409.12186', 'vaswani2017': '1706.03762', 'dicicco2026': '2606.14530v3',
    'klassert2026': '2609.13975v1', 'ruan2026': '2607.06503v2', 'zhu2025': '2509.12886v1',
    'stbon2025': '2503.01422v3',
    'rathnasuriya2026': '2610.11300v1',
}


def fetch(pair):
    key, identifier = pair
    url = f'https://arxiv.org/abs/{identifier}'
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    def fields(name):
        return [tag['content'] for tag in soup.select(f'meta[name="{name}"]')]
    title, authors, dates = fields('citation_title'), fields('citation_author'), fields('citation_date')
    if not title or not authors or not dates:
        raise ValueError(f'Missing primary bibliographic metadata: {url}')
    entry = {'key': key, 'title': title[0], 'authors': authors, 'year': dates[0][:4], 'url': url,
             'eprint': identifier, 'source': 'arXiv citation metadata', 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    body = '\n'.join(f'  {name} = {{{value}}},' for name, value in {
        'title': '{' + entry['title'] + '}', 'author': ' and '.join(authors), 'year': entry['year'],
        'eprint': identifier, 'archivePrefix': 'arXiv', 'url': url,
        'urldate': entry['retrieved_at'][:10],
        'note': 'Preprint; versão consultada registrada no endereço quando explicitamente fixada',
    }.items())
    return entry, f'@unpublished{{{key},\n{body}\n}}\n'


def preserve_additional_entries(existing: str, generated: str) -> str:
    generated_keys = set(re.findall(r'(?m)^@\w+\{([^,]+),', generated))
    extra = []
    for block in re.split(r'(?m)(?=^@)', existing):
        match = re.match(r'@\w+\{([^,]+),', block)
        if match and match[1] not in generated_keys:
            extra.append(block.rstrip())
    return generated.rstrip() + '\n\n' + '\n\n'.join(extra) + '\n'


def preserve_source_metadata(existing: list[dict], generated: list[dict]) -> list[dict]:
    keys = {entry['key'] for entry in generated}
    return generated + [entry for entry in existing if entry['key'] not in keys]


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(fetch, ARXIV.items()))
    official = r'''
@article{cawley2010,
  title={On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  author={Cawley, Gavin C. and Talbot, Nicola L. C.},
  journal={Journal of Machine Learning Research}, year={2010}, volume={11}, pages={2079--2107},
  url={https://jmlr.org/papers/v11/cawley10a.html}
}
@article{ojala2010,
  title={Permutation Tests for Studying Classifier Performance},
  author={Ojala, Markus and Garriga, Gemma C.},
  journal={Journal of Machine Learning Research}, year={2010}, volume={11}, pages={1833--1863},
  url={https://jmlr.org/papers/v11/ojala10a.html}
}
@inproceedings{guo2017,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  year={2017}, volume={70}, pages={1321--1330}, publisher={PMLR},
  url={https://proceedings.mlr.press/v70/guo17a.html}
}
'''
    target = ROOT / 'tcc'
    target.mkdir(exist_ok=True)
    bibliography = target / 'referencias.bib'
    existing = bibliography.read_text(encoding='utf-8') if bibliography.exists() else ''
    text = preserve_additional_entries(existing, '\n'.join(body for _, body in entries) + official)
    bibliography.write_text(text, encoding='utf-8')
    metadata_path = target / 'reference_sources.json'
    existing_metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else []
    metadata = preserve_source_metadata(existing_metadata, [entry for entry, _ in entries])
    metadata_path.write_text(json.dumps(metadata,
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{len(re.findall(r'(?m)^@', text))} bibliography entries; additional references preserved")


if __name__ == '__main__':
    main()
