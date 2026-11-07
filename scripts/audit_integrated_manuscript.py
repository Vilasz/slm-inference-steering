"""Check the integrated edition against its sources and the frozen evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    import typst
    import pymupdf
    from slm_steering.final_extension.common import assert_history, validate_freeze
    from render_integrated_manuscript import sha

    freeze=validate_freeze()
    assert_history()
    directory=ROOT/'reports/manuscript_revision_20261107'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for key in ('inputs','generated','editorial_sources'):
        for path, expected in manifest[key].items():
            if sha(ROOT/path)!=expected:
                raise ValueError(f'Editorial provenance differs: {path}')
    if sha(ROOT/'tcc/build/manuscrito.pdf')!=manifest['pdf_sha256']:
        raise ValueError('PDF differs from the recorded build')
    _,warnings=typst.Compiler(str(ROOT/'tcc/main.typ'),root=str(ROOT)).compile_with_warnings()
    if warnings:
        raise ValueError(f'Compiler warnings require review: {warnings}')
    active=set()
    def visit(path):
        path=path.resolve()
        if path in active:
            return
        if not path.is_relative_to(ROOT):
            raise ValueError('Manuscript source outside workspace')
        active.add(path)
        text=path.read_text(encoding='utf-8')
        for name in re.findall(r'#(?:include|import)\s+"([^"]+\.typ)"',text):
            visit(path.parent/name)
    visit(ROOT/'tcc/main.typ')
    text='\n'.join(p.read_text(encoding='utf-8') for p in active)
    forbidden=['Um único modelo pequeno e dois benchmarks','Expansões para classificadores mais complexos precisam',
               'A avaliação externa desta extensão ainda não foi concluída',
               'uma análise de repetição agrupada de todo o ajuste exigiria dimensionamento próprio e não foi realizada']
    for phrase in forbidden:
        if phrase.casefold() in text.casefold():
            raise ValueError(f'Obsolete scope statement remains active: {phrase}')
    if re.search(r'@@[A-Z0-9_]+@@',text):
        raise ValueError('Unresolved template marker')
    for path in ('generated/resumo.typ','chapters/01_introducao.typ','chapters/04_metodologia.typ',
                 'chapters/07_discussao_integrada.typ','generated/conclusao.typ'):
        section=(ROOT/'tcc'/path).read_text(encoding='utf-8')
        if not all(term in section for term in ('Qwen','DeepSeek','MLP')):
            raise ValueError(f'Incomplete integrated scope in {path}')
    doc=pymupdf.open(ROOT/'tcc/build/manuscrito.pdf')
    body=' '.join(' '.join(p.get_text().split()) for p in doc)
    if re.search(r'\b(?:In|Em):\s*\.',body):
        raise ValueError('Empty conference prefix in the rendered bibliography')
    final=json.loads((ROOT/'reports/final_extension/results.json').read_text(encoding='utf-8'))
    for model in ('deepseek','qwen'):
        for family in ('linear','MLP'):
            row=final['contrasts'][model][family]
            for number in (row['estimate'],*row['ci95']):
                needle=f'{number:.4f}'.replace('-','−').replace('.',',')
                if needle not in body:
                    raise ValueError(f'External contrast not found in compiled PDF: {needle}')
    for needle in ('−0,047','−0,099','0,006','82,72','99,08','−16,36'):
        if needle not in body:
            raise ValueError(f'Historical result missing: {needle}')
    overflow=[]
    vertical_overflow=[]
    for i,page in enumerate(doc):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                x0,y0,x1,y1=line['bbox']
                if x0<65 or x1>page.rect.width-35:
                    overflow.append({'page':i+1,'bbox':line['bbox']})
                if y0<40 or y1>page.rect.height-40:
                    vertical_overflow.append({'page':i+1,'bbox':line['bbox']})
    if overflow:
        raise ValueError(f'Content outside width requires inspection: {overflow[:5]}')
    if vertical_overflow:
        raise ValueError(f'Content outside vertical bounds requires inspection: {vertical_overflow[:5]}')
    result={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'pages':len(doc),
            'frozen_at_utc':freeze['frozen_at_utc'],'frozen_sources_and_history_unchanged':True,
            'input_and_output_hashes_verified':True,'compiler_warnings':[],
            'obsolete_scope_phrases_in_active_text':0,'numeric_contrasts_found_in_pdf':True,
            'width_overflows':0,'vertical_overflows':0,'empty_conference_prefixes':0,'active_sources':sorted(p.relative_to(ROOT).as_posix() for p in active),
            'chapters':doc.get_toc(),
            'scope':'Local editorial consistency checks; not an independent scientific or institutional approval'}
    (directory/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Integrated edition: {len(doc)} pages, zero compiler warnings, frozen results intact')


if __name__=='__main__':
    main()
