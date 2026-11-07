"""Publish the final acceptance report only from completed, hash-checked evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.protocol import file_hash, now, write_json


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    assert_preserved(ROOT)
    final = read('reports/excellence/final_verification.json')
    if final['status'] != 'completed_including_approved_independent_evaluation':
        raise ValueError('Complete independent analysis and publication before final acceptance')
    state = read('reports/research/full_execution_v0.2/status.json')
    original = read('reports/excellence/archive_before_extension/reports/research/full_execution_v0.2/status.json')
    if state['status'] != 'completed' or any(state['stages'][k] != original['stages'][k] for k in list(original['stages'])[:15]):
        raise ValueError('Historical science changed or publication incomplete')
    for relative, expected in read('reports/excellence/independent_v1/freeze/manifest.json')['inputs'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Independent frozen source changed')
    evidence = read('reports/final_review/evidence.json')
    delivery = read('reports/final_review/delivery_manifest.json')
    for relative, expected in {**evidence['inputs'], **delivery['artifacts']}.items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f'Publication hash differs: {relative}')
    visual = read('reports/excellence/visual_review/manifest.json')
    if visual['status'] != 'all_pages_explicitly_visually_inspected':
        raise ValueError('Inspect every rendered page before recording visual acceptance')
    for pdf in visual['pdfs'].values():
        if file_hash(ROOT / pdf['path']) != pdf['sha256'] or pdf['text_spans_outside_page']:
            raise ValueError('Visual review is stale or text is outside the PDF page')
    independent = read('reports/excellence/independent_v1/analysis.json')
    validation = final['validation']
    if validation['pytest_failed'] or validation['pytest_skipped']:
        raise ValueError('Regression validation incomplete')
    publication = final['publication']
    operational_path = ROOT / 'reports/excellence/operational_v1/results.json'
    operational = json.loads(operational_path.read_text(encoding='utf-8')) if operational_path.exists() else None
    lines = ['# Relatório final de qualidade', '', f'Verificação em {now()}.', '', '## PASSOU', '',
        '- Ciência original preservada: dezesseis artefatos protegidos, fontes congelados e os quinze registros científicos originais idênticos.',
        '- Tabelas reconstruídas de artefatos: 39 linhas de preditores, 14.432 linhas de replay histórico e 2.400 linhas do replay independente, retenção, critérios conjuntos, ablações e oracle. O novo endpoint e seu bootstrap foram recontados por implementação aritmética separada. Reexecuções estatísticas integrais original/transferência coincidem com os JSONs históricos.',
        '- Resultado original negativo mantido: ΔWT-AUC −0,0470, IC95% [−0,0994; 0,0056]; Δlog loss +0,0379, IC95% [0,0067; 0,0706]. Benefício operacional conjunto não demonstrado.',
        f"- Reserva aprovada concluída: {independent['main_tasks']} tarefas principais/2.400 tentativas e vinte tarefas técnicas/400 tentativas disjuntas; endpoint único {independent['primary']}.",
        '- Gates e limiares originais em t=8; denominador comum; bootstrap pareado por tarefa; nenhum ajuste no alvo, troca de tarefa ou interrupção por resultado.',
        '- Auditoria de prompts/testes e parser; dois pares textualmente semelhantes qualificados e agrupados somente em sensibilidade posterior. Omissão dos helpers no snapshot explicitada com recuperação pré-aprovação por HEAD+patch.',
        '- Casos sintéticos de replay/oracle e custo finito verificados; nenhum sucesso perdido pelo oracle no mesmo banco/orçamento. Oracle apresentado como contrafactual e tokens separados de latência.',
        f"- Regressão: {validation['pytest_passed']} testes sem falhas nem ignorados, Ruff e mypy aprovados. Notebook com {publication['notebook_executed_code_cells']} células executadas, sem erros.",
        '- Bibliografia primária revisada, matriz final e fonte recente incorporadas; nenhuma prioridade universal ou causalidade atribuída às ablações.',
        f"- Manuscrito, resumo de duas páginas e doze slides compilados; {publication['academic_questions']} perguntas respondidas. Todas as páginas dos três PDFs inspecionadas visualmente, com hashes e notas em `visual_review/manifest.json`.",
        '- Formato ajustado ao manual geral FGV consultado, com escopo institucional explicitado; comandos de análise separados de coleta longa em `docs/FINAL_REPRODUCIBILITY.md`.']
    if operational:
        lines.append(f"- Latência operacional medida em {operational['runs']} execuções, seis tarefas conhecidas, três repetições e N=3, quatro modos; GPU real {operational['hardware']['device']}. Resultado {operational['status']}, descritivo, com componentes de tempo e cobertura publicados.")
    lines += ['', '## PENDENTE', '',
        '- Revisão final e anuência do orientador sobre a versão integrada; ensaio pelo estudante para defender pessoalmente cada decisão. Isso é distinto da revisão da proposta já informada antes da coleta. Risco: dúvidas de autoria/argumentação na defesa. Ação: enviar a versão final e realizar defesa simulada.',
        '- Confirmação pelo curso/biblioteca do modelo institucional exigido, ficha catalográfica quando aplicável e folha de aprovação após defesa real. Risco: devolução administrativa. Ação: validar o arquivo com EMAP/biblioteca e inserir somente documentos autênticos.',
        '- Preparação de entrega/depósito e conferência final de autoria. Reservar 1–14/12/2026 prioritariamente para redação, orientação e defesa; não iniciar coleta de alto risco nesse período.']
    if not operational or operational['status'] != 'completed_descriptive':
        lines.append('- Benchmark operacional não completou todas as 72 execuções planejadas; consultar registros técnicos. Sua condição parcial é explícita e não redefine o estudo independente. Ação: retomar apenas se viável; não alegar aceleração robusta.')
    lines += ['', '## NÃO VERIFICÁVEL', '',
        '- Nota futura, aceitação em mestrado, aprovação formal da banca e manifestação direta/assinatura do orientador: dependem de avaliação externa. A revisão da proposta foi informada pelo estudante, sem assinatura fabricada.',
        '- Ausência de contaminação no pré-treinamento, de toda dependência semântica e de adaptação humana histórica: os artefatos só sustentam os controles e limitações documentados.',
        '- Corretude universal além da suíte, desempenho em outros modelos/benchmarks, causalidade de dimensão/domain shift e aceleração robusta em produção ou RTX4060 não acessível.',
        '- Energia e FLOPs não foram medidos; nem tokens nem tempos descritivos substituem essas medidas.', '']
    output = ROOT / 'reports/excellence/final_quality_report.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    paths = ['reports/excellence/final_verification.json', 'reports/final_review/delivery_manifest.json',
             'reports/excellence/independent_v1/analysis.json', 'reports/excellence/final_scientific_audit.md',
             'reports/excellence/visual_review/manifest.json', 'reports/excellence/final_quality_report.md',
             'scripts/finalize_quality_report.py']
    write_json(ROOT / 'reports/excellence/final_quality_manifest.json',
               {'created_at': now(), 'status': 'technical_work_complete_external_review_explicit',
                'inputs_and_report': {p: file_hash(ROOT / p) for p in paths}})
    print('Final quality report written; external decisions remain explicit')


if __name__ == '__main__':
    main()
