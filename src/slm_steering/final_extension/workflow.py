from __future__ import annotations

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from slm_steering.final_extension.common import (ROOT, REPORT, RUN, PROTOCOL, atomic_json, file_hash, now, read_json, immutable_json, validate_freeze)


def prepare():
    from slm_steering.final_extension.preparation import metadata, historical_audit, inventory
    from slm_steering.final_extension.sandbox import SandboxedEvalPlus, probe_sandbox
    from slm_steering.final_extension.power import run as power
    REPORT.mkdir(parents=True, exist_ok=True)
    historical_audit()
    if not (REPORT / 'model_metadata.json').exists():
        metadata()
    atomic_json(REPORT / 'sandbox_preflight.json', {'measured_at': now(), 'checks': probe_sandbox()})
    if not (REPORT / 'task_inventory.json').exists():
        with SandboxedEvalPlus('mbpp') as verifier:
            inventory(verifier.dataset['tasks'], verifier.dataset['metadata'])
    design = read_json(REPORT / 'task_inventory.json')
    final_path = REPORT / 'final_design.json'
    final = read_json(final_path) if final_path.exists() else None
    power(len(final['external_ids']) if final and final['feasible'] else design['external_tasks'])
    if not (REPORT / 'pilot_plan.json').exists():
        paths = ['docs/protocols/FINAL_CROSS_MODEL_EXTENSION_v1.md', 'reports/final_extension/task_inventory.json',
                 'reports/final_extension/model_metadata.json', 'src/slm_steering/final_extension/generation.py',
                 'src/slm_steering/final_extension/collection.py', 'src/slm_steering/final_extension/sandbox.py',
                 'scripts/final_extension_worker.py', 'scripts/evalplus_worker.py']
        atomic_json(REPORT / 'pilot_plan.json', {'frozen_at': now(), 'authorization': 'Student request 2026-11-04',
                                                'development_ids': design['pilot_ids'], 'attempts': 20,
                                                'files': {p: file_hash(ROOT / p) for p in paths},
                                                'external_labels_inspected': False})
    report()


def charged_hours():
    from slm_steering.final_extension.resources import usage
    return usage()['charged_active_session_hours']


def finalize_design():
    if (REPORT / 'final_design.json').exists():
        return read_json(REPORT / 'final_design.json')
    from slm_steering.final_extension.collection import summarize_pilot
    from slm_steering.final_extension.power import run as power
    pilot = summarize_pilot()
    inventory = read_json(REPORT / 'task_inventory.json')
    source_n = len(inventory['source_ids']) * 20
    spent = charged_hours()
    options = []
    for n in (150, 140, 120, 100, 80):
        if n > len(inventory['eligible_ids']):
            continue
        # All attempts: measured per-model pilot times plus 50% contingency,
        # including verifier/model-load time through the conservative wall-rate.
        ds_manifest = read_json(RUN / 'deepseek/pilot/manifest.json')
        qw_manifest = read_json(RUN / 'qwen/technical/manifest.json')
        ds_rate = max(pilot['mean_seconds_deepseek'], sum(s['wall_seconds'] for s in ds_manifest['sessions']) / 240)
        qw_rate = max(pilot['mean_seconds_qwen'], sum(s['wall_seconds'] for s in qw_manifest['sessions']) / 12)
        projected = spent + 1.5 * ((source_n + n * 20) * ds_rate + n * 20 * qw_rate) / 3600
        options.append({'tasks': n, 'projected_hours_with_contingency': projected,
                        'measured_deepseek_seconds_per_attempt': ds_rate, 'measured_qwen_seconds_per_attempt': qw_rate})
    choice = next((o for o in options if o['projected_hours_with_contingency'] <= 60), None)
    reasons = []
    if not pilot['feasible']:
        reasons.append('Pilot failed one or more predeclared outcome/formatting gates')
    if choice is None:
        reasons.append('Even 80 external tasks exceed the 60h projection including contingency')
    result = {'created_at': now(), 'viable_pilot': pilot['feasible'], 'budget_hours': 60,
              'hours_charged': spent, 'options': options,
              'feasible': bool(pilot['feasible'] and choice),
              'infeasibility_reasons': reasons,
              'minimal_prospective_amendment_if_infeasible': (
                  'If formatting/performance fails, audit prompt/extraction on development only and repeat a versioned pilot. '
                  'If only budget fails, review a smaller external sample below 80 with a new precision analysis before any external generation; '
                  'retain both models, matched classifier comparisons and 20 attempts. No automatic scientific change is authorized by this rule.'),
              'external_ids': inventory['eligible_ids'][:choice['tasks']] if choice and pilot['feasible'] else [],
              'selected_option': choice, 'outcome_selection': False}
    immutable_json(REPORT / 'final_design.json', result) if not (REPORT / 'final_design.json').exists() else None
    if result['feasible']:
        power(choice['tasks'])
    pilot_table = '\n'.join(f'| {o["tasks"]} | {o["projected_hours_with_contingency"]:.2f} |' for o in options)
    (REPORT / 'pilot_report.md').write_text(f'''# Piloto DeepSeek e viabilidade

Doze tarefas de desenvolvimento fixadas antes da coleta; vinte tentativas por
tarefa, sem seleção pelo resultado. A amostra final permanece sem rótulos novos.

```json
{json.dumps(pilot, ensure_ascii=False, indent=2)}
```

O benchmark Qwen usa uma tentativa nas mesmas doze tarefas. Tempo por tentativa
para dimensionamento usa o maior entre geração média e parede média da sessão,
incluindo carga/verificação. A contingência multiplica o custo futuro por 1,5.
Os manifests registram pico de memória e o benchmark pareado de captura.

| Tarefas externas por modelo | Horas previstas, incluindo contingência |
|---:|---:|
{pilot_table}

Decisão viável: {result['feasible']}. Motivos de inviabilidade: {reasons}.
O tamanho escolhido usa custo e inventário, sem examinar labels externos.
Em caso de inviabilidade: {result['minimal_prospective_amendment_if_infeasible']}.
Não se altera automaticamente temperatura, modelo, tarefas ou limiares.
''', encoding='utf-8')
    report()
    return result


def freeze():
    from slm_steering.final_extension.probes import FAMILIES
    design = read_json(REPORT / 'final_design.json')
    if not design['feasible']:
        raise ValueError('Pilot/budget infeasible; do not freeze an external run')
    if (REPORT / 'freeze.json').exists():
        return validate_freeze()
    if not read_json(REPORT / 'source_label_audit.json')['all_match']:
        raise ValueError('Reused Qwen source labels need matching sandbox audit')
    from slm_steering.final_extension.bank_audit import audit
    audit('development')
    tests = REPORT / 'tests.xml'
    xml = ET.parse(tests).getroot()
    if any(int(s.get('failures', 0)) or int(s.get('errors', 0)) for s in xml.iter('testsuite')):
        raise ValueError('Required tests failed')
    paths = [PROTOCOL, REPORT / 'task_inventory.json', REPORT / 'model_metadata.json', REPORT / 'final_design.json',
             REPORT / 'power_results.json', REPORT / 'pilot_results.json', tests, ROOT / 'scripts/final_extension_worker.py',
             ROOT / 'scripts/evalplus_worker.py', ROOT / 'scripts/run_final_extension.py',
             ROOT / 'tests/test_final_extension.py', ROOT / 'tests/test_final_extension_resources.py', REPORT / 'hardware.json',
             ROOT / 'docs/final_extension_decision.md', REPORT / 'source_label_audit.json',
             REPORT / 'development_bank_audit.json', REPORT / 'environment_packages.json']
    paths += list((ROOT / 'src/slm_steering/final_extension').glob('*.py'))
    paths += [ROOT / 'src/slm_steering/research' / f for f in ('generation.py', 'features.py', 'statistics.py', 'protocol.py')]
    paths += [ROOT / 'src/slm_steering' / f for f in ('generation.py', 'verifier.py')]
    paths += [ROOT / 'src/slm_steering/excellence' / f for f in ('economics.py', 'audit.py')]
    paths += [ROOT / 'src/slm_steering/activations/hooks.py']
    for name in ('resource_accounting_adjustments.json', 'windows_power_events_20261105.json', 'resource_accounting_tests.xml'):
        if (REPORT / name).exists():
            paths.append(REPORT / name)
    amendment = ROOT / 'docs/protocols/FINAL_CROSS_MODEL_EXTENSION_v1_RESOURCE_ACCOUNTING.md'
    if amendment.exists():
        paths.append(amendment)
    for name in ('deepseek', 'qwen'):
        for family in FAMILIES:
            paths += [RUN / name / 'probes' / f'{family}.joblib', RUN / name / 'probes' / f'{family}.json']
    for p in RUN.glob('*/external/records/*.json'):
        raise ValueError(f'External outcome already exists before final freeze: {p}')
    value = {'frozen_at_utc': now(), 'authorization': 'Student request 2026-11-04; no advisor approval inferred',
             'external_outcomes_before_freeze': 0,
             'files': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}}
    atomic_json(REPORT / 'freeze.json', value)
    (REPORT / 'freeze.sha256').write_text(file_hash(REPORT / 'freeze.json') + '\n', encoding='utf-8')
    return value


def report():
    REPORT.mkdir(parents=True, exist_ok=True)
    from slm_steering.final_extension.resources import usage
    resource_usage = usage()
    audit = read_json(REPORT / 'historical_audit.json') if (REPORT / 'historical_audit.json').exists() else {}
    inventory = read_json(REPORT / 'task_inventory.json') if (REPORT / 'task_inventory.json').exists() else {}
    pilot = read_json(REPORT / 'pilot_results.json') if (REPORT / 'pilot_results.json').exists() else None
    results = read_json(REPORT / 'results.json') if (REPORT / 'results.json').exists() else None
    label_audit = read_json(REPORT / 'source_label_audit.json') if (REPORT / 'source_label_audit.json').exists() else None
    training = read_json(REPORT / 'qwen_training_sensitivity.json') if (REPORT / 'qwen_training_sensitivity.json').exists() else None
    label_text = (f"A reavaliação adicional dos {label_audit['candidates']} programas Qwen de origem no sandbox, "
                  f"com limite de 3 GiB em vez do padrão histórico de 4 GiB, encontrou {len(label_audit['mismatches'])} "
                  "discrepâncias. O artefato source_label_audit.json registra a comparação; labels anteriores permanecem intactos.") if label_audit else "A reavaliação adicional dos labels Qwen no sandbox ainda não foi concluída."
    training_text = (f"Foram executadas {training['repetitions']} repetições completas de seleção, refit e calibração "
                     f"em subamostras de tarefas de origem, com desvio de ΔWT-AUC "
                     f"{training['training_sensitivity_sd']['linear']:.4f} (linear) e "
                     f"{training['training_sensitivity_sd']['MLP']:.4f} (MLP) no conjunto fixo de auditoria. "
                     "São sensibilidades em subamostras sobrepostas, não estimativas sem viés da variância do procedimento final.") if training else "A sensibilidade por reajustes completos ainda não foi concluída."
    tests = REPORT / 'tests.xml'
    checks = None
    if tests.exists():
        suites = list(ET.parse(tests).getroot().iter('testsuite'))
        checks = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'errors', 'failures', 'skipped')}
    status = {'updated_at': now(), 'historical_audit': bool(audit), 'tests': checks,
              'pilot': pilot, 'final_frozen': (REPORT / 'freeze.json').exists(),
              'external_results_complete': results is not None, 'charged_gpu_session_hours': charged_hours(),
              'resource_accounting': resource_usage,
              'collections': [{'path': str(p.relative_to(ROOT)), 'status': read_json(p)['status'],
                              'completed': read_json(p).get('completed_attempts', 0),
                              'expected': read_json(p).get('expected_attempts', 0)} for p in RUN.glob('*/*/manifest.json')]}
    atomic_json(REPORT / 'status.json', status)
    text = f'''# Auditoria científica da extensão final

Atualizado: {status['updated_at']}. Estado e contagens reproduzíveis em `status.json`.

Contabilidade de recursos: {json.dumps(resource_usage, ensure_ascii=False)}.
Suspensões comprovadas pelo log do sistema são excluídas do orçamento de GPU,
mantendo manifests e tempos brutos intactos. A emenda de recursos e os eventos
originais ficam versionados antes da avaliação externa. Esse valor continua
sendo tempo de parede ativo da sessão, incluindo carga e verificação, não um
perfil de tempo de kernels CUDA. Os IDs, seeds, labels e a amostra não mudam.

## Evidência histórica preservada

O banco original contém {audit.get('attempts', '?')} gerações, {audit.get('tasks', '?')}
tarefas e {audit.get('mixed_tasks_t32', '?')} tarefas mistas em t=32. O contraste
original foi reconstruído exatamente a partir dos scores por candidato, com
empates conferidos por contagem independente de pares. Seeds, labels gravados
versus status base+plus, identidades e agrupamento externo foram verificados.
Hashes em `historical_hashes.json` protegem os experimentos anteriores, inclusive
a validação independente de 120 tarefas. Seus resultados e conclusões não mudam.
A retenção 82,72% versus 99,08% (−16,36 pp) continua uma conclusão sobre os gates
congelados anteriores, não uma impossibilidade universal de sinal interno.

O inventário local encontra {len(inventory.get('exposed_ids', []))} IDs MBPP expostos,
{len(inventory.get('eligible_ids', []))} elegíveis após a triagem lexical, entre
{inventory.get('total_tasks', '?')} tarefas. IDs, hashes de fontes e pares
semelhantes ficam em `task_inventory.json`; não há garantia de independência
semântica completa ou ausência de contaminação no pré-treinamento.

## Correções delimitadas à nova extensão

O worker histórico não fornecia isolamento de rede/arquivos pelo sistema
operacional. A extensão executa EvalPlus em bubblewrap/WSL, com namespaces,
ambiente limpo e mounts mínimos read-only. `sandbox_preflight.json` documenta
rede inacessível, host oculto e limites de recursos. Não se modificaram labels
antigos. {label_text}

O template DeepSeek já inclui BOS; o novo adapter usa add_special_tokens=False.
Estados são capturados depois de consumir o token do checkpoint, somente em
posições necessárias. A validação com o modelo compara cache, prefixo isolado,
sequência completa e probabilidades teacher-forced. Esses checks só contam
como executados quando aparecem no manifest da coleta; testes unitários não
substituem validação do modelo real.

PCA64, imputação e escalas são ajustados dentro dos treinos; GroupKFold separa
tarefas, early stopping usa tarefas internas e calibração/limiar usam reserva
da origem. Quatro famílias com grids correspondentes impedem confundir H com
uma mudança isolada de classe do classificador. Três seeds são retidas sem
escolher a melhor. Intervalos finais permanecem condicionais aos probes;
cenários de variabilidade de treinamento no planejamento são aproximações.
{training_text}

## Verificações e execução

Testes específicos: {checks}. Conferem observáveis sem futuro, PCA sem dados
retidos, seeds de MLP, probabilidades, AUC/empates/homogêneas, pareamento,
caso T=32, replay/custos e Bellman contra enumeração exata de políticas iid.
Estado do piloto: {json.dumps(pilot, ensure_ascii=False) if pilot else 'não concluído'}.
Protocolo final travado: {status['final_frozen']}. Resultados externos completos:
{status['external_results_complete']}. Fontes e bundles somente ficam travados
por `freeze.json` após treino, piloto e dimensionamento técnico.

Não publicar inferência externa a partir de um banco parcial. Tabelas finais
são reconstruídas dos JSON por candidato/NPZ e escritas por `evaluation.py`.
Tarefas homogêneas entram nos custos, Brier/log loss e denominadores gerais;
AUC intratarefa exige as mistas. As trajetórias dos dois modelos na mesma
tarefa compartilham o cluster. Tokens economizados não medem latência.

## Pendências explicitamente delimitadas

Enquanto `results.json` não existir, não há conclusão empírica nova sobre a
hipótese DeepSeek/MLP nem justificativa para reinterpretar os achados anteriores.
A extensão matemática em `docs/final_extension_decision.md` é conceitual,
com antecedentes identificados e testes sintéticos; não é teorema inédito.
'''
    (REPORT / 'scientific_audit.md').write_text(text, encoding='utf-8')
    (REPORT / 'README.md').write_text('''# Extensão científica final

- [Estado executado](status.json)
- [Protocolo](../../docs/protocols/FINAL_CROSS_MODEL_EXTENSION_v1.md)
- [Auditoria científica](scientific_audit.md)
- [Poder e precisão](power_analysis.md)
- [Inventário e IDs](task_inventory.json)
- [Decisão sequencial e derivação](../../docs/final_extension_decision.md)
- [Reprodução e segurança](../../docs/FINAL_EXTENSION_REPRODUCIBILITY.md)
- [Manuscrito atualizado](../../tcc/build/manuscrito.pdf)

Comandos: `.venv/Scripts/python.exe scripts/run_final_extension.py --help`.
`run` percorre as fases com orçamento local máximo aproximado de 60h;
as coletas retomam por identidade. Consulte status.json antes de interpretar
resultados; existência de código não significa experimento concluído.
''', encoding='utf-8')
    return status


def run_child(command):
    process = subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/run_final_extension.py'), command], cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f'Phase failed: {command} ({process.returncode}); no automatic scientific workaround')


def wait_for_existing_session(name, phase):
    """A refreshed controller lets an already active atomic session finish."""
    import psutil
    lock = RUN / name / phase / '.writer.lock'
    deadline = time.monotonic() + 900
    announced = False
    while lock.exists():
        owner = read_json(lock)
        if not psutil.pid_exists(owner['pid']):
            raise RuntimeError(f'Stale collection lock requires audited recovery: {lock}')
        if not announced:
            print(f'Waiting for existing {name}/{phase} writer PID {owner["pid"]} to finish its atomic session', flush=True)
            announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError('Existing collection session exceeds controller handoff window; writer was not interrupted')
        time.sleep(5)


def run_all():
    prepare()
    for name, phase in (('deepseek', 'pilot'), ('qwen', 'technical')):
        while True:
            wait_for_existing_session(name, phase)
            run_child(f'{name}-{phase}')
            manifest = read_json(RUN / name / phase / 'manifest.json')
            if manifest['status'] == 'completed':
                break
            if charged_hours() >= 60:
                atomic_json(REPORT / 'execution_outcome.json', {'status': 'budget_exhausted', 'at': now(), 'stage': 'pilot'})
                report()
                return
    design = read_json(REPORT / 'final_design.json') if (REPORT / 'final_design.json').exists() else finalize_design()
    if not design['feasible']:
        atomic_json(REPORT / 'execution_outcome.json', {'status': 'infeasible_at_pilot_or_budget', 'at': now(), 'design': design})
        report()
        from slm_steering.final_extension.publication import integrate
        integrate()
        return
    while True:
        wait_for_existing_session('deepseek', 'source')
        run_child('deepseek-source')
        if read_json(RUN / 'deepseek/source/manifest.json')['status'] == 'completed':
            break
        if charged_hours() >= 60:
            atomic_json(REPORT / 'execution_outcome.json', {'status': 'budget_exhausted', 'at': now(), 'stage': 'source'})
            report()
            return
    run_child('train-qwen')
    run_child('train-deepseek')
    run_child('audit-qwen-labels')
    freeze()
    for name in ('deepseek', 'qwen'):
        while True:
            wait_for_existing_session(name, 'external')
            run_child(f'{name}-external')
            if read_json(RUN / name / 'external/manifest.json')['status'] == 'completed':
                break
            if charged_hours() >= 60:
                atomic_json(REPORT / 'execution_outcome.json', {'status': 'budget_exhausted', 'at': now(), 'stage': name+'-external'})
                report()
                return
    run_child('evaluate')
    report()
    from slm_steering.final_extension.publication import integrate
    integrate()
    atomic_json(REPORT / 'execution_outcome.json', {'status': 'completed', 'at': now(),
                'charged_gpu_session_hours': charged_hours(), 'results': 'reports/final_extension/results.json'})
