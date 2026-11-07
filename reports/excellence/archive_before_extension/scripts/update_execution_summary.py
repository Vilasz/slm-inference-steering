"""Refresh the human-readable evidence index from actual publication artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def update(root: Path = ROOT):
    evidence = json.loads((root / 'reports/final_review/evidence.json').read_text(encoding='utf-8'))
    lines = ['# Estado da evidência', '', f"Snapshot: {evidence['built_at']}.", '',
             'A execução integral foi autorizada pelo estudante. O desenho e os fontes foram',
             'copiados antes do banco principal. Aprovação do orientador pendente;',
             'não se reivindica pré-registro independente. Consulte a',
             '[emenda administrativa](docs/execution_amendment_v0.2.md).', '',
             'O status atual, sem carregar a GPU, é mostrado por:', '',
             '```powershell', '.venv/Scripts/python.exe scripts/research_status.py', '```', '',
             '## Coleta', '', '| Banco | Trajetórias | Tarefas observadas | Estado |', '|---|---:|---:|---|']
    for pilot in evidence['pilot']:
        lines.append(f"| Piloto v{pilot['version']} | {pilot['attempts']}/{pilot['planned']} | {pilot['tasks']} | {pilot['status']} |")
    for phase, planned in [('definitive', 2640), ('transfer', 640)]:
        path = root / f'runs/candidate_bank/{phase}_v0.2/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        lines.append(f"| {phase} v0.2 | {manifest.get('completed_attempts', 0)}/{planned} | {len(manifest.get('task_ids', [])) if manifest else 0} | {manifest.get('status', 'aguardando')} |")
    lines.extend(['', 'O piloto escolhe temperatura somente após as 960 trajetórias.',
                  'O banco principal mantém todas as tentativas, mesmo após um sucesso.', '',
                  '## Resultados principais', ''])
    if evidence.get('main'):
        main = evidence['main']
        lines.extend(['| Preditor em 32 tokens | AUC | WT-AUC | Log loss |', '|---|---:|---:|---:|'])
        for row in main['signal']['metrics']:
            if row['checkpoint'] == '32' and row['predictor'] in ('B', 'H', 'HB'):
                def display(key):
                    return f"{row[key]:.4f}" if row[key] is not None else 'não identificável'
                lines.append(f"| {row['predictor']} | {display('roc_auc')} | {display('within_task_auc')} | {display('log_loss')} |")
        lines.extend(['', 'Decisão de Q2 e intervalos pareados:', '', '```json',
                      json.dumps(main['signal']['H2'], ensure_ascii=False, indent=2), '```', '',
                      'Decisão operacional:', '', '```json',
                      json.dumps(main['policies']['H3'], ensure_ascii=False, indent=2), '```', '',
                      'Bootstrap por tarefa condicional aos modelos OOF ajustados; não refaz o treino.',
                      'Tempos online são descritivos. Economia de tokens não implica menor latência.'])
    else:
        lines.extend(['Q1 precoce, Q2 em 32 tokens e Q3 ainda não foram estimadas no banco principal.',
                      'A execução está ativa; taxas do piloto não substituem os testes principais.'])
    if evidence.get('supplemental'):
        extra = evidence['supplemental']
        lines.extend(['', '## Diagnósticos complementares posteriores ao resultado', '',
                      'Robustez por fold e exclusão de tarefa, coorte temporal constante, sensibilidade ao',
                      'histórico, calibração e retenção externa foram acrescentadas como análises exploratórias.',
                      'Modelos, calibradores, limiares e critérios principais permanecem congelados.', '',
                      '[Relatório completo dos diagnósticos](reports/final_review/supplemental/README.md).', '',
                      '| Avaliação | Política | Corretos retidos/ativos | Retenção [IC 95%] |',
                      '|---|---|---:|---|'])
        for dataset, item in extra['correct_retention'].items():
            for row in item['aggregate']:
                ci = row['ci95']
                value = f"{row['recall']:.4f} [{ci[0]:.4f}; {ci[1]:.4f}]" if ci else 'não identificável'
                lines.append(f"| {dataset} | {row['policy']} | {row['retained_correct']}/{row['active_correct']} | {value} |")
        lines.extend(['', 'O denominador usa todos os corretos ativos no gate congelado, inclusive candidatos',
                      'não iniciados no replay. As políticas podem usar checkpoints diferentes.',
                      'O alvo empírico de 97% no desenvolvimento não constitui garantia externa.'])
        if all(row['sign_preserved_all_deletions'] for row in extra['influence_summary']):
            lines.extend(['A retirada de qualquer tarefa isolada preserva o sinal das estimativas principais',
                          '(com os escores fixos); isso não garante significância em cada exclusão.'])
    lines.extend(['', '## Histórico exploratório separado', '',
                  '| Preditor tardio | AUC | WT-AUC | Log loss |', '|---|---:|---:|---:|'])
    legacy = evidence['legacy_reanalysis']
    for row in legacy['metrics']:
        lines.append(f"| {row['family']} | {row['roc_auc']:.3f} | {row['within_task_auc']:.3f} | {row['log_loss']:.3f} |")
    lines.extend(['', 'Estados finais, 150 tentativas/30 tarefas, verificador histórico local.',
                  'Não são evidência de predição precoce ou de EvalPlus.',
                  'O incremento de WT-AUC permanece inconclusivo; a perda melhora nesta reanálise.', '',
                  '## Verificação e apresentação', ''])
    report = root / 'reports/research/test_results_v0.2.xml'
    if report.exists():
        suites = list(ET.parse(report).getroot().iter('testsuite'))
        total = sum(int(s.get('tests', 0)) for s in suites)
        skipped = sum(int(s.get('skipped', 0)) for s in suites)
        failed = sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites)
        lines.append(f'Suíte: {total - skipped - failed} testes passaram, {skipped} ignorados, {failed} falhas/erros.')
    lines.extend(['', '[Notebook executado](notebooks/final_research_summary.ipynb),',
                  '[HTML](reports/final_review/notebook.html), [PDF](tcc/build/manuscrito.pdf),',
                  '[evidência e hashes](reports/final_review/evidence.json),',
                  '[auditoria de publicação](reports/final_review/delivery_manifest.json).', '',
                  'O runner gera figuras temporais, controles, ROC/PR, calibração, deltas por tarefa,',
                  'efeitos pareados das políticas, risco de aborto e transferência congelada.',
                  'Resultados negativos e intervalos que incluem zero são preservados.', ''])
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    update()
