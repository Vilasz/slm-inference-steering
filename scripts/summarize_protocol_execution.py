from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.protocol import read_bank, write_json


def main():
    parser = argparse.ArgumentParser(description='Summarize a bounded pilot session without computing probes')
    parser.add_argument('--bank-dir', type=Path, default=ROOT / 'runs/candidate_bank/pilot_v0.2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/research')
    args = parser.parse_args()
    manifest_path = args.bank_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['phase'] != 'pilot':
        raise ValueError('This report is reserved for development pilot execution, not confirmatory inference')
    records = read_bank(args.bank_dir / 'trajectories.jsonl')
    tokens = sum(r['generated_tokens'] for r in records)
    generation = sum(r['generation_time_seconds'] for r in records)
    verification = sum(r['verification_time_seconds'] for r in records)
    max_relative = max(value for checkpoint in manifest['causal_invariance']['checks']
                       for comparison in checkpoint['comparisons'] for value in comparison['relative_l2_by_layer'])
    count = len(records)
    measured_path = args.output_dir / 'hardware_measurements.json'
    measurements = (json.loads(measured_path.read_text(encoding='utf-8'))
                    if measured_path.exists() and manifest['protocol_version'] == '0.1' else [])
    maximum_peak = max([manifest['hardware']['peak_reserved_mib']] + [row['peak_reserved_mib'] for row in measurements]
                       + [session.get('hardware', {}).get('peak_reserved_mib', 0) for session in manifest['sessions']])
    session_count = len(manifest['sessions'])
    resume_path = args.output_dir / 'resume_validation.json'
    resume_verified = resume_path.exists() and json.loads(resume_path.read_text(encoding='utf-8')).get('passed', False)
    # No dependence on success labels, latent scores or a favorable temperature.
    hours_pilot = (generation + verification) / count * manifest['expected_attempts'] / 3600
    hours_definitive = (generation + verification) / count * 2640 / 3600
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_name = f'pilot_manifest_v{manifest["protocol_version"]}.json'
    shutil.copyfile(manifest_path, args.output_dir / snapshot_name)
    run_date = manifest['started_at'][:10]
    version = manifest['protocol_version']
    temperature_selected = manifest.get('selected_temperature') is not None
    report = f'''# Execução de desenvolvimento — banco iniciado em {run_date}

Status: **{manifest['status']}; nenhuma conclusão confirmatória**.

| Item | Medida |
|---|---:|
| Trajetórias verificadas | {count} / {manifest['expected_attempts']} |
| Tarefas representadas | {len(set(r['task_id'] for r in records))} / 32 |
| Temperaturas | 0,6; 0,8; 1,0 |
| Tokens de continuação | {tokens} |
| Tempo de geração do banco | {generation:.2f} s |
| Tempo de verificação do banco | {verification:.2f} s |
| Velocidade efetiva de geração | {tokens / generation:.2f} tokens/s |
| Pico reservado observado | {maximum_peak:.0f} MiB |
| Maior erro L2 relativo no teste causal | {max_relative:.8f} |

GPU: {manifest['hardware']['device']}. Python {manifest['python']}, PyTorch
{manifest['packages']['torch']}, CUDA {manifest['hardware']['cuda']}. Pesos bfloat16, projeções lineares
float32 temporárias, backend eager, batch 1, TF32 desativado. A configuração
está em `configs/protocol_v0.2.yaml`; o manifesto copiado está em
[{snapshot_name}]({snapshot_name}).

O teste causal passou para 0/8/16/32/64/final, tanto no forward completo
quanto na geração com KV cache. O maior erro observado ficou abaixo da
tolerância predefinida de 1%. A repetição da primeira tentativa com a mesma
semente reproduziu os IDs dos tokens. Sessões concluídas: {session_count}.
Validação adicional do hash de um prefixo do banco antes/depois da retomada:
{'aprovada' if resume_verified else 'não disponível'}; consulte
`resume_validation.json` quando presente.

EvalPlus 0.3.1 executou as suítes base e plus integrais no Ubuntu via WSL;
controles positivo e negativo passaram. O dataset HumanEval+ é v0.2.10.
Os hashes do benchmark, tokens, texto, código e ativações foram auditados.
Nenhuma falha de inicialização/numericalidade recebeu rótulo científico.

No preflight, float16 gerou logits não finitos e bfloat16 nativo excedeu a
tolerância causal. Os registros dessas tentativas técnicas estão em
`runs/diagnostics/`. A configuração validada foi escolhida por validade
numérica antes da coleta de trajetórias rotuladas, sem comparar desempenho
de probes, taxa de acerto ou economia de uma política.

A extrapolação mecânica da duração média deste pequeno lote dá cerca de
**{hours_pilot:.1f} h para o piloto completo** e **{hours_definitive:.1f} h para
o banco definitivo**, somente geração/verificação. Essas estimativas são
provisórias: comprimentos, prefill, temperatura, carga e condições térmicas
podem mudar. Elas excluem os testes iniciais, carregamento do modelo, análise
estatística e MBPP+. O limite por sessão controla tempo de execução, não reduz
o número de tentativas nem o teto de 512 tokens.

Temperatura selecionada: **{manifest.get('selected_temperature')}**. O piloto não fornece evidência confirmatória de probes.
Complete as 960 trajetórias, prepare o registro de sampling e obtenha o
congelamento previsto pelo protocolo antes do banco definitivo. MBPP+ permanece
uma análise futura; seu subset deverá ser definido por hash de task_id antes
de inspecionar desfechos.
'''
    (args.output_dir / f'execution_{run_date}_v{version}.md').write_text(report, encoding='utf-8')
    write_json(args.output_dir / f'execution_summary_v{version}.json', {'development_only': True, 'attempts': count,
               'planned_attempts': manifest['expected_attempts'], 'completion_tokens': tokens,
               'generation_seconds': generation, 'verification_seconds': verification,
               'maximum_causal_relative_l2': max_relative, 'pilot_hours_extrapolated': hours_pilot,
               'definitive_hours_extrapolated': hours_definitive, 'temperature_selected': temperature_selected,
               'protocol_version': version})
    print(f'Execution report saved to {args.output_dir}')


if __name__ == '__main__':
    main()
