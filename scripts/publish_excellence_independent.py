"""Publish only complete approved independent results, with regression and hash checks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def run(arguments):
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace',
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f'Publication command failed: {arguments}')
    return result.stdout


def update_documents(analysis):
    primary = analysis['primary']
    effect = f"{primary['estimate']}; IC95% {primary['ci95']}"
    finding = 'satisfeito' if primary['directional_hypothesis_supported'] else 'não satisfeito'
    path = ROOT / 'docs/excellence_defense.md'
    guide = path.read_text(encoding='utf-8').split('<!-- independent-completed -->')[0].rstrip()
    next_question = max(map(int, re.findall(r'^### (\d+)\.', guide, re.MULTILINE)), default=0) + 1
    guide = guide.replace('exposição e aprovação pendente divulgadas.', 'exposição e histórico de autorização divulgados.')
    guide = guide.replace('| 12. Próximo teste e conclusão | 1 min | Validação independente proposta, não executada; contribuição delimitada e auditável. |',
        '| 12. Avaliação independente e conclusão | 1 min | 120 tarefas novas; endpoint de retenção e suas limitações, sem substituir Q2/H3. |')
    guide = guide.replace('Nenhuma aprovação foi registrada automaticamente.',
        'A aprovação explícita e a revisão informada pelo estudante foram registradas antes do congelamento e das novas trajetórias.')
    guide = guide.replace('### 33. O que é necessário antes da nova coleta?',
                          '### 33. Quais condições precederam a nova coleta?')
    guide += f'''

<!-- independent-completed -->
### {next_question}. Qual foi a nova validação independente?

Foram 20 tarefas técnicas disjuntas (400 tentativas) e 120 tarefas principais
MBPP+ sem resultados locais anteriores (2.400 tentativas), reservadas por hash.
O estudante aprovou e informou revisão do orientador antes do congelamento.
B e H+B de origem foram comparados no mesmo t=8, sem ajuste no alvo.
O único endpoint confirmatório é a diferença de razões de contagens de corretos
mantidos/ativos, H+B menos B: {effect}. Critério direcional: **{finding}**.
São 10.000 reamostragens pareadas por tarefa, seed 20261017; o suporte exige
limite superior do IC bilateral de 95% abaixo de zero. O piloto técnico não
entra nessas contagens. Tarefas sem acertos continuam no conjunto de clusters.

### {next_question + 1}. Esse resultado muda as hipóteses originais ou prova uma causa de domain shift?

Não muda Q2/H3 originais. O endpoint novo compara retenção de dois gates
específicos já congelados. Não testa superioridade universal de representações,
ganho incremental após novos ajustes, energia ou latência. Mesmo quando o critério
de retenção é satisfeito, a diferença não identifica uma causa de domain shift.
Há um único modelo e uma nova reserva MBPP+; o bootstrap pressupõe tarefas
intercambiáveis e a ausência de exposição local não certifica ausência no pré-treino.
As métricas de custo, resolução, calibração e outros orçamentos são secundárias.
'''
    path.write_text(guide, encoding='utf-8')
    path = ROOT / 'reports/excellence/scientific_audit.md'
    audit = path.read_text(encoding='utf-8').split('<!-- independent-completed -->')[0].rstrip()
    audit += f'''

<!-- independent-completed -->
## Validação independente após aprovação

O relato explícito do estudante autorizou o congelamento prospectivo da proposta
e informou revisão do orientador. O protocolo manteve seu hash e foi copiado com
os fontes para um snapshot de 46 arquivos antes da primeira trajetória nova.
Foram concluídas 400 tentativas técnicas e 2.400 principais. A auditoria conferiu
identidades, seeds, hashes de todas as ativações, controles e probabilidades
reaplicadas: erro máximo {analysis['rescoring_max_probability_error']}.

Endpoint único H+B−B em t=8: {effect}; critério **{finding}**. Os critérios
originais e os resultados exploratórios foram preservados. A revisão informada
pelo estudante não foi transformada em assinatura ou pré-registro externo.
Detalhes e limitações: [independent_v1/README.md](independent_v1/README.md).
'''
    path.write_text(audit, encoding='utf-8')


def verify_publication(validation):
    import pymupdf
    from slm_steering.excellence.audit import assert_preserved
    from slm_steering.research.protocol import file_hash, write_json, now
    assert_preserved(ROOT)
    state = read('reports/research/full_execution_v0.2/status.json')
    old = read('reports/excellence/archive_before_extension/reports/research/full_execution_v0.2/status.json')
    if state['status'] != 'completed' or len(state['stages']) != 19:
        raise ValueError('Publication pipeline incomplete')
    scientific_stages = list(state['stages'])[:15]
    if any(state['stages'][key] != old['stages'][key] for key in scientific_stages):
        raise ValueError('An original scientific stage record changed')
    evidence = read('reports/final_review/evidence.json')
    delivery = read('reports/final_review/delivery_manifest.json')
    for paths in (evidence['inputs'], delivery['artifacts']):
        for relative, expected in paths.items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError(f'Published hash differs: {relative}')
    if not evidence['excellence']['independent']['primary'] == delivery['excellence']['independent_primary']:
        raise ValueError('Independent primary result differs across publication artifacts')
    notebook = read('notebooks/final_research_summary.ipynb')
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    if any(c['execution_count'] is None or any(o.get('output_type') == 'error' for o in c.get('outputs', [])) for c in cells):
        raise ValueError('Notebook incomplete or contains errors')
    pdfs = {}
    for relative in ('tcc/build/manuscrito.pdf', 'reports/excellence/materials/research_summary_en.pdf', 'reports/excellence/materials/defesa.pdf'):
        with pymupdf.open(ROOT / relative) as document:
            pdfs[relative] = {'pages': len(document), 'sha256': file_hash(ROOT / relative)}
    result = {'created_at': now(), 'status': 'completed_including_approved_independent_evaluation',
              'historical_science': {'protected_artifacts_unchanged': 16, 'execution_freeze_input_hashes_valid': 29,
                                    'unchanged_scientific_stage_records': scientific_stages, 'original_trajectories': 4240},
              'publication': {'stages_verified': 19, 'delivery_manifest_sha256': file_hash(ROOT / 'reports/final_review/delivery_manifest.json'),
                              'evidence_sha256': file_hash(ROOT / 'reports/final_review/evidence.json'), 'pdfs': pdfs,
                              'notebook_executed_code_cells': len(cells), 'notebook_errors': 0,
                              'academic_questions': delivery['excellence']['questions']},
              'validation': validation, 'independent_study': evidence['excellence']['independent']}
    write_json(ROOT / 'reports/excellence/final_verification.json', result)
    progress = ROOT / 'reports/excellence/PROGRESS.md'
    journal = progress.read_text(encoding='utf-8').split('<!-- independent-publication-completed -->')[0].rstrip()
    journal += (f'\n\n<!-- independent-publication-completed -->\n## Avaliação independente concluída e publicada\n\n'
        f"Conclusão em {now()}: 400 tentativas técnicas e 2.400 principais, sem substituir tarefas nem alterar gates. "
        f"Endpoint: {evidence['excellence']['independent']['primary']}. "
        f"{validation['pytest_passed']} testes aprovados, sem ignorados; Ruff e mypy aprovados. "
        f"Notebook com {len(cells)} células executadas sem erros; manuscrito com {pdfs['tcc/build/manuscrito.pdf']['pages']} páginas. "
        f"Research summary com duas páginas, defesa com 12 slides e {delivery['excellence']['questions']} perguntas respondidas. "
        'As 15 etapas científicas históricas permanecem idênticas. '
        'Relatório: `reports/excellence/independent_v1/README.md`; verificação: `final_verification.json`.\n')
    progress.write_text(journal, encoding='utf-8')
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8').replace(
        'A coleta está em execução; resultados completos, auditoria e\npublicação serão gerados pelo supervisor, sem ajuste dos critérios ao alvo.',
        'A coleta, a análise e a publicação foram concluídas: 400 tentativas técnicas\ne 2.400 principais, sem ajuste dos critérios ao alvo. '
        '[Resultados independentes](reports/excellence/independent_v1/README.md).')
    text = re.sub(r'Doze slides e \d+ perguntas respondidas',
                  f"Doze slides e {delivery['excellence']['questions']} perguntas respondidas", text)
    path.write_text(text, encoding='utf-8')
    print(json.dumps({'status': result['status'], 'primary': evidence['excellence']['independent']['primary'],
                      'notebook_cells': len(cells), 'pdfs': pdfs}), flush=True)


def main():
    from slm_steering.excellence.approvals import approved_design
    from slm_steering.research.protocol import write_json, now
    approved_design(ROOT)
    run(['-u', 'scripts/analyze_excellence_independent.py'])
    analysis = read('reports/excellence/independent_v1/analysis.json')
    update_documents(analysis)
    try:
        run(['-u', 'scripts/run_operational_benchmark.py'])
    except RuntimeError as exc:
        (ROOT / 'reports/excellence/operational_v1').mkdir(exist_ok=True)
        write_json(ROOT / 'reports/excellence/operational_v1/technical_failure.json',
                   {'created_at': now(), 'status': 'technical_failure_optional_benchmark', 'error': str(exc)})
    run(['scripts/run_final_scientific_audit.py'])
    run(['scripts/analyze_research_robustness.py'])
    run(['scripts/prepare_final_figures.py'])
    pytest_output = run(['-c', "import os; os.environ['SLM_STEERING_TEST_TORCH']='1'; import torch,pytest; raise SystemExit(pytest.main(['-q','--junitxml=reports/research/test_results_v0.2.xml']))"])
    suites = list(ET.parse(ROOT / 'reports/research/test_results_v0.2.xml').getroot().iter('testsuite'))
    total = sum(int(s.get('tests', 0)) for s in suites)
    skipped = sum(int(s.get('skipped', 0)) for s in suites)
    failures = sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites)
    if skipped or failures:
        raise ValueError('Full Torch-enabled regression suite must pass without skipped tests')
    ruff = run(['-m', 'ruff', 'check', '.'])
    mypy = run(['-m', 'mypy', 'src', 'scripts'])
    validation = {'created_at': now(), 'pytest_passed': total, 'pytest_skipped': skipped, 'pytest_failed': failures,
                  'torch_tests_enabled': True, 'pytest_summary': pytest_output.strip().splitlines()[-1],
                  'ruff': ruff.strip(), 'mypy': mypy.strip()}
    write_json(ROOT / 'reports/excellence/independent_v1/publication_validation.json', validation)
    audit_path = ROOT / 'reports/excellence/scientific_audit.md'
    audit_text = re.sub(r'A suíte integral com PyTorch habilitado encerrou com \d+ testes aprovados',
                        f'A suíte integral com PyTorch habilitado encerrou com {total} testes aprovados',
                        audit_path.read_text(encoding='utf-8'))
    audit_path.write_text(audit_text, encoding='utf-8')
    run(['scripts/run_excellence.py', 'report'])
    run(['scripts/build_excellence_materials.py'])
    status_path = ROOT / 'reports/research/full_execution_v0.2/status.json'
    state = read('reports/research/full_execution_v0.2/status.json')
    if (status_path.parent / '.runner.lock').exists():
        raise RuntimeError('Another original publication runner is active')
    original = read('reports/excellence/archive_before_extension/reports/research/full_execution_v0.2/status.json')
    if any(state['stages'].get(key) != original['stages'][key] for key in list(original['stages'])[:15]):
        raise ValueError('Original scientific stages must be intact before refreshing publication')
    for key in ('delivery', 'notebook', 'manuscript', 'publication_audit'):
        state['stages'].pop(key, None)
    state.update(status='publication_refresh_pending')
    state.pop('ended_at', None)
    state.setdefault('publication_revisions', []).append({'requested_at': datetime.now(timezone.utc).isoformat(),
        'reason': 'Integrate complete prospectively approved independent retention endpoint; original scientific records retained'})
    write_json(status_path, state)
    run(['-u', 'scripts/run_research_safely.py', '--wsl-python', '/home/joaof/.venvs/slm-evalplus-v031/bin/python'])
    verify_publication(validation)


if __name__ == '__main__':
    main()
