"""Render every final PDF page and record explicit visual inspection separately.

Rendering and geometry checks alone never count as visual approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.protocol import file_hash, now, write_json

PDFS = {'manuscript': 'tcc/build/manuscrito.pdf',
        'summary': 'reports/excellence/materials/research_summary_en.pdf',
        'slides': 'reports/excellence/materials/defesa.pdf'}
OUTPUT = ROOT / 'reports/excellence/visual_review'


def prepare():
    import pymupdf
    from PIL import Image, ImageDraw
    OUTPUT.mkdir(exist_ok=True)
    manifest = {'created_at': now(), 'status': 'awaiting_explicit_visual_inspection', 'pdfs': {}, 'groups': {}}
    for name, relative in PDFS.items():
        with pymupdf.open(ROOT / relative) as document:
            bounds = []
            small = []
            for page_number, page in enumerate(document, 1):
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines', []):
                        for span in line['spans']:
                            x0, y0, x1, y1 = span['bbox']
                            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                                bounds.append({'page': page_number, 'text': span['text'], 'bbox': span['bbox']})
                            if span['size'] < 8 and span['text'].strip():
                                small.append({'page': page_number, 'size': span['size'], 'text': span['text'][:80]})
            manifest['pdfs'][name] = {'path': relative, 'sha256': file_hash(ROOT / relative), 'pages': len(document),
                                      'text_spans_outside_page': bounds, 'text_spans_below_8pt': small,
                                      'geometry_scope': 'Text spans only; equations subscripts may be small; neither check certifies diagram legibility'}
            for start in range(0, len(document), 4):
                group = f'{name}_{start + 1:03d}_{min(start + 4, len(document)):03d}'
                slide = name == 'slides'
                tile_width, tile_height = (820, 500) if slide else (450, 655)
                canvas = Image.new('RGB', (tile_width * 2, tile_height * 2), '#d5d5d5')
                draw = ImageDraw.Draw(canvas)
                pages = []
                for position in range(4):
                    if start + position >= len(document):
                        break
                    page = document[start + position]
                    scale = min((tile_width - 20) / page.rect.width, (tile_height - 35) / page.rect.height)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    bitmap = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                    x, y = position % 2 * tile_width + 10, position // 2 * tile_height + 25
                    canvas.paste(bitmap, (x, y))
                    draw.text((x, y - 18), f'{name} PDF page {start + position + 1}', fill='black')
                    pages.append(start + position + 1)
                path = OUTPUT / (group + '.jpg')
                canvas.save(path, quality=94)
                manifest['groups'][group] = {'pdf': name, 'pages': pages,
                    'image': path.relative_to(ROOT).as_posix(), 'image_sha256': file_hash(path), 'reviewed': False}
    write_json(OUTPUT / 'manifest.json', manifest)
    print(f"Prepared {len(manifest['groups'])} groups; none marked visually reviewed")


def record(groups, notes):
    path = OUTPUT / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for group in groups:
        item = manifest['groups'][group]
        pdf = manifest['pdfs'][item['pdf']]
        if file_hash(ROOT / pdf['path']) != pdf['sha256'] or file_hash(ROOT / item['image']) != item['image_sha256']:
            raise ValueError('PDF or rendered group changed; prepare and inspect again')
        item.update(reviewed=True, reviewed_at=now(), reviewer='Codex visual inspection of rendered pages', notes=notes)
    complete = all(item['reviewed'] for item in manifest['groups'].values())
    manifest['status'] = 'all_pages_explicitly_visually_inspected' if complete else 'partially_visually_inspected'
    manifest['updated_at'] = now()
    write_json(path, manifest)
    print(manifest['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'record'])
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--notes')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        if not args.groups or not args.notes:
            parser.error('Explicit groups and inspection notes are required')
        record(args.groups, args.notes)


if __name__ == '__main__':
    main()
