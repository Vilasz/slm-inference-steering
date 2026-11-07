"""Build a traceable synthesis and manuscript chapter from executed extensions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from slm_steering.excellence.audit import assert_preserved
from slm_steering.excellence.models import loss_vector
from slm_steering.research.protocol import file_hash, now, write_json
from slm_steering.research.statistics import paired_task_bootstrap, task_auc


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence'
    paths = {name: f'reports/excellence/{name}/results.json' for name in ('models', 'transfer', 'economics', 'geometry', 'hardware', 'prefixes')}
    results = {name: json.loads((root / path).read_text(encoding='utf-8')) for name, path in paths.items()}
    predictions = pd.read_csv(output / 'models/predictions.csv')
    prior = pd.read_csv(output / 'archive_models_first_pass/predictions.csv')
    key = ['family', 'task_id', 'attempt_id']
    comparison = predictions.merge(prior, on=key, suffixes=('', '_prior'), validate='one_to_one')
    repeat_error = float(np.max(np.abs(comparison.probability - comparison.probability_prior)))
    if repeat_error > 1e-12 or len(comparison) != len(predictions):
        raise ValueError('Independent exploratory rerun differs from the archived first pass')
    recovery = []
    baseline = predictions[predictions.family == 'HB_original'].set_index(['task_id', 'attempt_id']).sort_index()
    for family in ('HB_stronger', 'HB_pca16', 'HB_pca64', 'H_residual_B'):
        part = predictions[predictions.family == family].set_index(['task_id', 'attempt_id']).reindex(baseline.index)
        groups, y, p = part.reset_index().task_id.to_numpy(), part.passed.to_numpy(), part.probability.to_numpy()
        auc, original = task_auc(y, p, groups), task_auc(y, baseline.probability.to_numpy(), groups)
        losses = loss_vector(y, p) - loss_vector(y, baseline.probability.to_numpy())
        tasks = pd.DataFrame({'task': groups, 'loss': losses}).groupby('task', sort=True).loss.mean()
        recovery.append({'family': family,
                         'delta_wt_auc_vs_original_HB': paired_task_bootstrap([auc[t] - original[t] for t in auc], replications=2000, seed=20260921),
                         'delta_log_loss_vs_original_HB': paired_task_bootstrap(tasks.to_numpy(), replications=2000, seed=20260921)})
    costs = pd.read_csv(output / 'economics/per_task.csv')
    concentration = []
    for dataset in ('HumanEval+', 'MBPP+'):
        group = costs[(costs.dataset == dataset) & (costs.checkpoint == 8) & (costs.budget == 20)]
        v = group[group.policy == 'verifier'].set_index('task_id').sort_index()
        oracle = group[group.policy == 'oracle'].set_index('task_id').reindex(v.index)
        savings = v.tokens - oracle.tokens
        concentration.append({'dataset': dataset, 'share_oracle_saving_from_unresolved_tasks': float(savings[v.resolved == 0].sum() / savings.sum()),
                              'share_verifier_tokens_in_unresolved_tasks': float(v.loc[v.resolved == 0, 'tokens'].sum() / v.tokens.sum())})
    models = results['models']
    result = {'created_at': now(), 'status': 'completed_authorized_exploratory_extension',
              'independent_status': 'proposal_pending_student_approval_and_advisor_review; no new outcomes',
              'models': models['summaries'], 'model_units': models['units'], 'repeat_max_probability_error': repeat_error,
              'recovery_vs_original_HB': recovery, 'transfer': results['transfer'],
              'economics': results['economics'], 'geometry': results['geometry'], 'hardware': results['hardware'],
              'oracle_concentration': concentration, 'prefix_diagnostics': results['prefixes'],
              'interpretation': ['Observable mean NLL and margin retain much of the point-estimate within-task ranking; probability loss tells a different story',
                                 'PCA and an anchored residual formulation reduce the original H+B deficit; neither establishes improvement over B',
                                 'Stronger joint regularization alone did not resolve the deficit in the tested family',
                                 'Weak score correlation is not evidence of useful incremental correctness information or conditional independence',
                                 'Transferred high-retention thresholds lose correct candidates on a concentrated subset of tasks',
                                 'The oracle measures conditional economic opportunity, not an implementable predictor or universal lower bound'],
              'limitations': ['Post-hoc comparisons; pointwise conditional bootstrap intervals without simultaneous coverage',
                             'No causal attribution to dimension, redundancy, or benchmark shift',
                             'No new independent confirmation before the requested approval/review gate',
                             'No wall-clock policy savings, energy savings or FLOPs claims from tokens']}
    independent_path = output / 'independent_v1/analysis.json'
    if independent_path.exists():
        independent = json.loads(independent_path.read_text(encoding='utf-8'))
        if independent['status'] != 'completed_approved_independent_analysis':
            raise ValueError('Incomplete independent analysis')
        for path, expected in independent['inputs'].items():
            if file_hash(root / path) != expected:
                raise ValueError(f'Independent analysis input changed: {path}')
        result.update(independent=independent, independent_status='completed_approved_independent_evaluation')
        result['limitations'][2] = 'The independent retention endpoint does not confirm the exploratory readout comparisons'
    operational_path = output / 'operational_v1/results.json'
    if operational_path.exists():
        operational = json.loads(operational_path.read_text(encoding='utf-8'))
        if file_hash(root / 'src/slm_steering/excellence/operational.py') != operational['source_sha256']:
            raise ValueError('Operational timing source differs from the measured implementation')
        for inputs in (operational['inputs'], operational['artifacts']):
            for path, expected in inputs.items():
                if file_hash(root / path) != expected:
                    raise ValueError(f'Operational timing artifact changed: {path}')
        result['operational'] = operational
    plot_models(output, models['summaries'])
    write_readme(output, result)
    input_paths = list(paths.values()) + ['reports/excellence/reproduction.json', 'reports/excellence/models/predictions.csv',
        'reports/excellence/models/plan.json', 'reports/excellence/transfer/retention_threshold_curves.csv',
        'reports/excellence/transfer/false_aborts_by_task.csv', 'reports/excellence/transfer/target_task_coverage.csv',
        'reports/excellence/economics/per_task.csv', 'reports/excellence/hardware/plan.json',
        'reports/excellence/hardware/pairs.jsonl', 'reports/excellence/hardware/component_timings.json',
        'reports/excellence/resource_budget.json', 'reports/excellence/resource_budget.md',
        'reports/excellence/proposal_design.json', 'reports/excellence/proposed_freeze_manifest.json',
        'reports/excellence/target_exposure_audit.json', 'reports/excellence/scientific_audit.md',
        'reports/excellence/README.md', 'docs/protocols/EXCELLENCE_EXTENSION_v1.md']
    input_paths += ['requirements-excellence.txt', 'scripts/run_excellence.py', 'scripts/run_excellence_extension.py',
                    'scripts/finalize_excellence_proposal.py', 'scripts/fetch_tcc_references.py', 'tcc/reference_sources.json']
    if result.get('independent'):
        input_paths += list(result['independent']['inputs']) + [
            'reports/excellence/independent_v1/analysis.json', 'reports/excellence/independent_v1/README.md',
            'reports/excellence/independent_v1/publication_validation.json',
            'scripts/run_approved_excellence.py', 'scripts/analyze_excellence_independent.py',
            'scripts/publish_excellence_independent.py']
    if result.get('operational'):
        input_paths += ['reports/excellence/operational_v1/results.json',
                        'reports/excellence/operational_v1/plan.json',
                        'reports/excellence/operational_v1/README.md']
        input_paths += list(result['operational']['artifacts'])
    input_paths += [p for p in ('reports/excellence/final_scientific_audit.md',
        'reports/excellence/final_table_reconstruction.json', 'reports/excellence/reserved_task_audit.json',
        'reports/excellence/reserved_similarity_notes.md',
        'reports/excellence/parser_provenance_reconstruction.json', 'reports/excellence/final_related_work_matrix.md',
        'reports/excellence/primary_sources/research_source_manifest.json',
        'reports/excellence/formatting_review.md', 'docs/FINAL_REPRODUCIBILITY.md',
        'docs/FINAL_HANDOFF.md',
        'reports/excellence/print_ready/manifest.json') if (root / p).exists()]
    if (output / 'print_ready/manifest.json').exists():
        input_paths += list(json.loads((output / 'print_ready/manifest.json').read_text(encoding='utf-8'))['artifacts'])
    for source in json.loads((output / 'primary_sources/research_source_manifest.json').read_text(encoding='utf-8')):
        relative = 'reports/excellence/primary_sources/' + source['file']
        if file_hash(root / relative) != source['sha256']:
            raise ValueError('Consulted primary-source PDF changed')
        input_paths.append(relative)
    input_paths += [p.relative_to(root).as_posix() for p in (output / 'figures').glob('*')]
    for directory in ('transfer', 'economics', 'geometry'):
        input_paths += [p.relative_to(root).as_posix() for p in (output / directory).glob('*') if p.suffix in ('.svg', '.png', '.pdf')]
    input_paths += [p.relative_to(root).as_posix() for p in (root / 'src/slm_steering/excellence').glob('*.py')]
    for unit in (root / 'runs/excellence/models_v1').glob('*.json'):
        metadata = json.loads(unit.read_text(encoding='utf-8'))
        for extension, key_name in (('.joblib', 'model_sha256'), ('.csv', 'predictions_sha256')):
            if file_hash(unit.with_suffix(extension)) != metadata[key_name]:
                raise ValueError('Exploratory model artifact changed')
        input_paths.append(unit.relative_to(root).as_posix())
    result['inputs'] = {path: file_hash(root / path) for path in sorted(set(input_paths))}
    write_json(output / 'results.json', result)
    print(f'Excellence synthesis: {len(result["inputs"])} auditable inputs; exploratory rerun error={repeat_error:g}', flush=True)


def plot_models(output, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    frame = pd.DataFrame(summaries)
    order = ['prompt_only', 'last_nll', 'max_entropy', 'mean_entropy', 'mean_nll', 'logit_margin',
             'nll_pair', 'entropy_pair', 'confidence_only', 'B_original', 'HB_original', 'HB_stronger', 'HB_pca16', 'HB_pca64', 'H_residual_B']
    frame = frame.set_index('family').loc[order]
    labels = {'prompt_only': 'Prompt', 'last_nll': 'NLL último token', 'max_entropy': 'Entropia máxima',
              'mean_entropy': 'Entropia média', 'mean_nll': 'NLL média', 'logit_margin': 'Margem',
              'nll_pair': 'NLL média + última', 'entropy_pair': 'Entropias', 'confidence_only': 'Confiança',
              'B_original': 'B original', 'HB_original': 'H+B original', 'HB_stronger': 'H+B, L2 forte',
              'HB_pca16': 'H+B, PCA16', 'HB_pca64': 'H+B, PCA64', 'H_residual_B': 'Residual ancorado em B'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    positions = np.arange(len(frame))
    axes[0].barh(positions, frame.within_task_auc, color='#3274A1')
    axes[0].set(yticks=positions, yticklabels=[labels[name] for name in order], xlabel='WT-AUC externa, 32 tokens', xlim=(.45, .68))
    axes[0].axvline(.628155136147495, color='black', ls='--', lw=.8)
    for j, (_, row) in enumerate(frame.iterrows()):
        effect = row.delta_log_loss_vs_B
        axes[1].plot(effect['ci95'], [j, j], color='#C44E52')
        axes[1].scatter(effect['estimate'], j, color='#C44E52', s=20)
    axes[1].axvline(0, color='gray', lw=.8)
    axes[1].set(yticks=positions, yticklabels=[], xlabel='Δlog loss contra B, IC pontual 95%')
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(figures / f'model_ablation.{ext}', dpi=180)
    plt.close(fig)


def write_readme(output, result):
    def effect(e):
        return f"{e['estimate']:.5f} [{e['ci95'][0]:.5f}; {e['ci95'][1]:.5f}]"
    lines = ['# Aprofundamento científico executado', '',
             'Os diagnósticos abaixo utilizam dados já examinados e são exploratórios. A avaliação independente tem seção própria. Resultados e protocolo v0.2 foram preservados.', '',
             '## Por que H+B perdeu para B?', '',
             '| Família | WT-AUC | Log loss | ΔWT-AUC contra B [IC95%] | Δlog loss contra B [IC95%] |', '|---|---:|---:|---|---|']
    for row in result['models']:
        lines.append(f"| {row['family']} | {row['within_task_auc']:.5f} | {row['log_loss']:.5f} | {effect(row['delta_wt_auc_vs_B'])} | {effect(row['delta_log_loss_vs_B'])} |")
    lines += ['', 'Média de NLL e margem preservam grande parte do ranking na estimativa pontual, mas têm perda maior que B.',
              'PCA e a correção residual reduzem a desvantagem de H+B; seus intervalos contra B continuam incluindo zero.',
              'Penalização conjunta ainda mais forte, isoladamente, não resolveu a desvantagem nesta grade.',
              'Isso sustenta sensibilidade à parametrização, sem identificar dimensionalidade como causa única ou provar redundância completa.', '',
              'PCA é ajustado apenas nos treinos de cada split; a correção residual usa offsets de B cuja seleção e calibração excluem o rótulo pontuado.',
              'O residual tem ligação logística nativa, sem um segundo Platt. C, camada e todos os pré-processamentos são escolhidos dentro do treino externo.',
              'As 70 unidades foram executadas duas vezes após uma limpeza de nomes de variáveis; o erro máximo das probabilidades foi ' + f"{result['repeat_max_probability_error']:.3g}.", '',
              'Prefixos compartilhados foram verificados em 8/16/32/64 tokens. Em 32, nenhuma das 92 tarefas mistas principais tem um único prefixo;',
              'portanto o colapso integral dos prefixos por tarefa não explica o contraste principal. Em 8 tokens, são 15/100 mistas na origem e 3/24 no alvo.',
              'Scores de prompt/prefixo idênticos conferem dentro da tolerância numérica; contagens e colisões parciais estão em `prefixes/`.', '',
              '## Transporte do gate', '',
              'O gate H+B da origem em t=8 descartou 90/366 corretos potenciais em MBPP+, distribuídos por 12 tarefas.',
              'Cinco tarefas concentram 60 desses descartes. As tarefas Mbpp/166 e Mbpp/259 perderam todos os seus sucessos.',
              'Entre corretos ativos efetivamente iniciados, 23/48 foram abortados; esse denominador difere dos 366 corretos potenciais.',
              'O ranking intratarefa H+B em t=8 no alvo é aproximadamente 0,494. Curvas mostram a distribuição dos corretos em torno do limiar congelado.',
              'A comparação inclui origem OOF, desenvolvimento da origem usado na seleção e modelo de origem completo aplicado ao alvo.',
              'As diferenças não isolam causalmente domain shift. Nenhum limiar foi ajustado com os rótulos MBPP+.', '',
              '## Limite econômico retrospectivo', '',
              '| Banco | t | Economia oracle contra verificador [IC95%] | Resolução |', '|---|---:|---|---:|']
    for row in result['economics']['summaries']:
        if row['policy'] == 'oracle' and row['budget'] == 20:
            lines.append(f"| {row['dataset']} | {row['checkpoint']} | {effect(row['saving_vs_verifier'])} | {row['resolution']:.5f} |")
    lines += ['', 'O oracle conhece rótulos finais e preserva todos os corretos. É limite de custo somente na classe com essa restrição e checkpoint fixo.',
              'Políticas que perdem sucessos podem custar menos em certos caminhos; por isso a fronteira sempre acompanha resolução.', '',
              '| Banco, t=8, N=20 | Parcela da economia oracle nas tarefas não resolvidas |', '|---|---:|']
    for row in result['oracle_concentration']:
        lines.append(f"| {row['dataset']} | {row['share_oracle_saving_from_unresolved_tasks']:.4f} |")
    lines += ['', 'A derivação econômica condiciona na tarefa, admite dependência entre custo e sucesso da mesma tentativa e exige independência entre tentativas.',
              'No limite ilimitado, a condição é E[D]/E[T] < r para p>0 e r>0. Para N finito, custo e cobertura devem ser comparados conjuntamente.',
              'Retenção global inclui os sucessos anteriores ao gate; não é sempre igual à retenção ativa.', '',
              '## Recursos, validação e continuidade', '',
              'Benchmark técnico: 50 pares / 100 gerações em 26 tarefas conhecidas, tokens e estados exatamente iguais entre implementações.',
              'GPU detectada: RTX 3050 Laptop 6 GB. A RTX 4060 mencionada na missão não está disponível neste ambiente.',
              'Tempos descritivos e componentes separados estão em `hardware/`; não se reivindica aceleração da política a partir de tokens.',
              'A nova avaliação independente está proposta com 20 tarefas técnicas e 120 principais, 20 tentativas por tarefa.',
              'Ela aguarda aprovação do estudante e revisão do orientador, conforme solicitado. Não houve coleta independente.', '',
              'Artefatos: [auditoria](scientific_audit.md), [reprodução](reproduction.json), [orçamento](resource_budget.md),',
              '[progresso](PROGRESS.md), [protocolo proposto](../../docs/protocols/EXCELLENCE_EXTENSION_v1.md).', '',
              '## Reprodução estatística sem geração na GPU', '', '```powershell',
              '.venv/Scripts/python.exe scripts/run_final_scientific_audit.py',
              '.venv/Scripts/python.exe scripts/run_excellence.py report', '```', '',
              '[Guia completo de reprodução](../../docs/FINAL_REPRODUCIBILITY.md), com reexecução estatística em pastas separadas e publicação.',
              'A busca exploratória já foi executada duas vezes e tem 70 unidades arquivadas; a auditoria compara as probabilidades sem reabrir uma seleção de vencedores.', '',
              '### Medições na GPU, separadas da análise', '', '```powershell',
              '.venv/Scripts/python.exe scripts/run_operational_benchmark.py', '```', '',
              'Esse comando usa tarefas antigas, exige a coleta independente encerrada e conserva resultados concluídos com o mesmo fonte.',
              'A coleta científica tem executor próprio com aprovação vinculada a hashes e não é acionada pelos comandos estatísticos.', '']
    if result.get('operational'):
        lines += ['## Latência operacional descritiva', '',
                  f"{result['operational']['runs']} execuções; seis tarefas conhecidas; três repetições; N=3; quatro modos.",
                  '[Plano, dados e componentes de tempo](operational_v1/README.md). Não se reivindica aceleração robusta.', '']
    if result.get('independent'):
        independent = result['independent']
        lines = [line for line in lines if line not in (
            'A nova avaliação independente está proposta com 20 tarefas técnicas e 120 principais, 20 tentativas por tarefa.',
            'Ela aguarda aprovação do estudante e revisão do orientador, conforme solicitado. Não houve coleta independente.')]
        p = independent['primary']
        lines += ['## Avaliação independente aprovada e concluída', '',
                  '20 tarefas técnicas disjuntas e 120 principais, 20 tentativas por tarefa; gates de origem mantidos em t=8.',
                  f"Retenção H+B−B: {p['estimate']}; IC95% {p['ci95']}; suporte pelo critério aprovado: {p['directional_hypothesis_supported']}.",
                  'Aprovação explícita do estudante; revisão do orientador informada por ele. Snapshot e hashes precedem a primeira trajetória.',
                  'Único endpoint confirmatório novo; não modifica Q2/H3 originais. Resultados secundários permanecem descritivos.', '',
                  '[Relatório, auditoria e dados completos](independent_v1/README.md).', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def render_typst(root, result, number, interval, table):
    def effect(e):
        return f"{number(e['estimate'], 4)} {interval(e['ci95'])}"
    text = '''#import "../lib.typ": *
= Aprofundamento exploratório e limite econômico

== Perguntas explicativas e desenho das ablações

Esta extensão foi definida em 17/10/2026, depois da conclusão e inspeção dos
resultados anteriores. Ela não substitui o endpoint original nem constitui nova
confirmação. Seu objetivo é distinguir explicações compatíveis com a desvantagem
de H+B, examinar decisões do gate transferido e quantificar a oportunidade
econômica de uma interrupção idealizada.

Mantiveram-se os cinco folds externos originais, com tarefas inteiras. Cada
ablação de observáveis seleciona C nos três folds internos. A penalização
conjunta forte usa C em 0,00001, 0,0001 e 0,001. PCA compara 16 e 64 componentes,
com camada e C escolhidos no treino; imputation, padronização e PCA são refeitos
dentro de cada split. A padronização e o classificador ponderam igualmente as
tarefas; PCA utiliza as observações de treino, sem rótulos, e whitening.

A calibração Platt recebe scores produzidos por cross-fitting de toda a seleção.
O residual usa uma probabilidade de B cujo ajuste, seleção e calibrador excluem
o rótulo do candidato pontuado. Dentro da seleção residual, esse primeiro estágio
é novamente refeito apenas no respectivo treino. Seu preditor é

$ hat(p)_"res"=sigma("logit"(hat(p)_B)+w_H^top tilde(H)). $

O offset fica fixo durante o ajuste L2 de w; não há segundo Platt. Esse método
ancora B e restringe a correção, mas a comparação não identifica separadamente
todos os efeitos de parametrização e calibração. Todos os intervalos adicionais
são bootstrap pontual de 2.000 tarefas com seed 20260921, condicionais aos modelos
ajustados; não há cobertura simultânea nem promoção de vencedores à hipótese principal.

== Componentes do baseline e leituras latentes alternativas

'''
    text += table(['Família', 'WT-AUC', 'Log loss', 'Δlog loss contra B e IC95%'],
                  [[r['family'], number(r['within_task_auc'], 4), number(r['log_loss'], 4), effect(r['delta_log_loss_vs_B'])]
                   for r in result['models'] if r['family'] != 'B_reproduced'], 'tab-excellence-models',
                  'Predições em tarefas externas no checkpoint de 32 tokens. Comparações exploratórias; sinais positivos de perda representam piora.')
    text += '''
O ranking obtido com NLL média ou margem isoladas é próximo ao de B na
estimativa pontual, enquanto sua perda probabilística é maior. Portanto,
identificar componentes com bom ranking não basta para justificar um limiar
absoluto de interrupção. A combinação das estatísticas beneficia a qualidade
probabilística neste regime. Prompt-only tem WT-AUC igual a 0,5 por construção.

PCA e o residual reduzem a desvantagem do H+B original; os intervalos contra B
continuam incluindo zero. Penalização conjunta mais forte, sozinha, não resolve
o problema na família testada. Esses resultados são compatíveis com sensibilidade
à parametrização e ao custo de estimação, sem demonstrar uma causa única ou
ausência universal de informação latente incremental.

#figure(image("../../reports/excellence/figures/model_ablation.svg", width: 100%),
 caption: [Ablações e leituras alternativas com seleção agrupada. À direita, diferenças de perda e intervalos pontuais; todas as famílias são mostradas.]) <fig-excellence-models>

== Geometria e sensibilidade ao número de tarefas

Os espectros descritivos usam vetores brutos, distintos dos padronizados na
ablação preditiva, e são ajustados exclusivamente no treino externo. Fração de variância
explicada não mede informação sobre corretude. Direções só são comparadas entre
folds que escolheram a mesma camada, depois de desfazer a escala das coordenadas.
Sobreposição dos treinos impede tratar seus ângulos como replicações independentes.

'''
    geometry = result['geometry']
    cosines = geometry['same_layer_direction_cosines']
    if cosines:
        values = [r['cosine_raw_coordinates'] for r in cosines]
        text += f"Na camada comum, os cossenos das direções variam de {number(min(values))} a {number(max(values))}. "
    text += f"A correlação entre escores H e B centrados por tarefa é {number(geometry['within_task_score_correlation_B_H'])}. "
    text += '''Correlação pequena não demonstra independência condicional, e direções
distintas não garantem informação útil adicional para o rótulo. A predição
linear do componente H+B latente a partir de B apresenta R² baixo ou negativo
em alguns folds, sem estabelecer redundância completa.

A curva de tarefas usa algoritmos explicitamente fixos: C=0,01, camada 13 e
PCA16 para a combinação. Subconjuntos de tarefas são definidos por hash dentro
do treino externo, com aproximadamente 32, 63 e 105 tarefas. Ela não representa
a curva do procedimento original completo nem estima um tamanho amostral ótimo.
WT-AUC não cresce monotonamente neste diagnóstico; aumentar a amostra não é
demonstrado como solução suficiente para a desvantagem original.

#figure(image("../../reports/excellence/geometry/geometry_learning.svg", width: 100%),
 caption: [Variância de treino por camada e sensibilidade de algoritmos fixos à quantidade de tarefas. Não são efeitos causais da dimensão ou do tamanho amostral.]) <fig-excellence-geometry>

=== Prefixos compartilhados e divergência

O teste de prefixos reagrupa as tentativas pelo prompt e IDs de tokens
observados, e confirma constância dos scores em cada grupo idêntico. Em t=8,
15 das 100 tarefas mistas HumanEval+ e três das 24 mistas MBPP+ têm um único
prefixo; nelas, leitores determinísticos não distinguem os resultados.
Em t=32, nenhuma das 92 tarefas mistas principais está nessa condição.
Portanto, o colapso integral de cada tarefa em um único prefixo não explica
o contraste negativo principal. Colisões parciais limitam alguns pares, sem
identificar sozinhas a causa da deterioração no alvo. As contagens por tarefa
e checkpoint estão exportadas, sem escolher casos pelos resultados do probe.

== Anatomia da deterioração do gate em MBPP+

A análise distingue origem OOF, desenvolvimento de origem utilizado na escolha
do gate e aplicação do bundle completo ao alvo. Reconstituir o desenvolvimento
reproduz sua retenção aparente de 0,9715 para H+B em t=8; o calibrador e o limiar
foram ajustados nesses rótulos, portanto esse valor não é validação externa.

No alvo, H+B em t=8 mantém 276 dos 366 corretos potenciais e descarta 90,
distribuídos por 12 tarefas; cinco concentram 60 descartes. Os corretos têm
mediana de score próxima de 0,373 e quartil inferior próximo de 0,286, com limiar
0,2695. O ranking intratarefa nesse ponto é 0,494, com intervalo amplo. Mudança
da prevalência e dos comprimentos acompanha a diferença de scores, mas os
procedimentos de ajuste também diferem entre origem e alvo: nenhuma comparação
isola por si só uma causa de domínio.

Mbpp/166 perde seus oito sucessos e Mbpp/259 perde seu único sucesso no banco,
explicando duas tarefas a menos resolvidas. Entre corretos ativos iniciados,
23 de 48 são descartados; entre todos os corretos potenciais, são 90 de 366.
Essa seleção pelo caminho de execução explica por que as duas taxas respondem
perguntas diferentes. As curvas no alvo são diagnósticos retrospectivos; não
fornecem um limiar reajustado ou uma nova política congelada.

#figure(image("../../reports/excellence/transfer/retention_curves.svg", width: 100%),
 caption: [Retenção versus limiar nas três avaliações distintas. Linhas verticais indicam gates escolhidos na origem, sem seleção sobre MBPP+.]) <fig-excellence-retention>

#figure(image("../../reports/excellence/transfer/conditional_scores.svg", width: 100%),
 caption: [Distribuições por resultado funcional no gate H+B de 8 tokens. São densidades por candidato; o desenvolvimento não é avaliação externa.]) <fig-excellence-score>

== Potencial econômico e distância ao oracle

O oracle conserva a mesma ordem, N e checkpoint e conhece os rótulos finais.
Preserva todos os corretos, interrompe os incorretos ativos e para no primeiro
sucesso. As condições matemáticas e a derivação estão na fundamentação. A
implementação independente do replay também reproduziu os custos, resolução,
tentativas iniciadas e descartes das políticas originais, tarefa por tarefa.

'''
    text += table(['Banco', 't', 'Tokens oracle/tarefa', 'Economia contra verificador e IC95%'],
                  [[r['dataset'], r['checkpoint'], number(r['tokens_per_task'], 1), effect(r['saving_vs_verifier'])]
                   for r in result['economics']['summaries'] if r['policy'] == 'oracle' and r['budget'] == 20],
                  'tab-excellence-oracle', 'Oracle retrospectivo, N=20. Sua resolução coincide exatamente com a do verificador; economia é razão de médias por tarefa.')
    for row in result['oracle_concentration']:
        text += f"Em {row['dataset']}, {number(100 * row['share_oracle_saving_from_unresolved_tasks'], 1)}% da economia oracle em t=8 vem de tarefas não resolvidas no banco. "
    text += '''Isso delimita onde está grande parte da oportunidade: evitar prolongar
tentativas que, retrospectivamente, não produzem solução. Acesso a esse rótulo
não existe online. Uma política que descarta sucessos pode até custar menos que
este oracle em um caminho, ao custo de perder cobertura; ele não é limite global
de todas as políticas possíveis.

#figure(image("../../reports/excellence/economics/oracle_frontier.svg", width: 100%),
 caption: [Custo e resolução para N=1,5,10,20 no checkpoint 8. B e H+B usam limiares específicos desse checkpoint escolhidos somente na origem; o oracle é idealizado.]) <fig-excellence-oracle>

== Medições técnicas e avaliação independente

'''
    hardware = result['hardware']
    text += f"O benchmark contém {hardware['pairs']} pares, {hardware['trajectories_generated']} gerações e {hardware['tasks']} tarefas conhecidas. "
    text += f"Tokens e estados coincidiram exatamente. A GPU conectada é {hardware['hardware']['device']}; pico reservado {number(hardware['hardware']['peak_reserved_mib'], 0)} MiB. "
    text += f"Os tempos médios de geração instrumentada foram {number(hardware['mean_original_seconds'])} s na captura original e {number(hardware['mean_selective_seconds'])} s na seletiva. "
    text += '''São tempos descritivos, parcialmente concorrentes com análises CPU;
não constituem inferência de aceleração da política. Os microbenchmarks separados
de gate, prefill e verificador estão nos artefatos. A RTX 4060 mencionada na
solicitação não está disponível; seu desempenho não é extrapolado.

A proposta independente compara os gates de origem em t=8 em 120 novas tarefas
MBPP+, com vinte tentativas cada e piloto técnico disjunto de vinte tarefas.
A escolha investiga transportabilidade de decisões e não seleciona o melhor
classificador exploratório. O endpoint é diferença de retenções sobre a mesma
população ativa, com bootstrap pareado por tarefa. IDs, precisão, orçamento e
gates estão em EXCELLENCE_EXTENSION_v1. A proposta aguarda aprovação do estudante
e revisão do orientador; nenhuma coleta independente é reivindicada.
'''
    if result.get('independent'):
        start = text.index('A proposta independente compara os gates de origem em t=8')
        text = text[:start]
        text = text.replace('= Aprofundamento exploratório e limite econômico',
                            '= Diagnósticos, limite econômico e validação independente')
        text = text.replace('Ela não substitui o endpoint original nem constitui nova\nconfirmação.',
                            'Os diagnósticos não substituem o endpoint original. A avaliação\nindependente subsequente tem hipótese e congelamento próprios.')
        if result.get('operational'):
            operational = result['operational']
            accumulated_seconds = sum(r['runs'] * r['total_seconds']['mean'] for r in operational['summaries'])
            text += '\n=== Latência completa em tarefas conhecidas\n\n'
            text += (f"Foram medidas {operational['runs']} execuções em seis tarefas do piloto já examinado, "
                     'três repetições e até três tentativas por tarefa, com a ordem dos quatro modos alternada. '
                     'O plano precedeu as medições; não usa a nova reserva. Warmup dos caminhos e dos oracles '
                     'do verificador e carregamento dos pesos foram excluídos. Tokens, estados capturados e '
                     'rótulos funcionais foram comparados ao banco conhecido.\n\n')
            text += (f'O somatório dos tempos completos de tarefa foi {number(accumulated_seconds / 60, 1)} minutos. '
                     'O orçamento temporal de 30 minutos é verificado antes de cada bloco de quatro modos; '
                     'o bloco final é concluído mesmo quando ultrapassa esse valor. Não se trata de um limite rígido por processo.\n\n')
            text += table(['Modo', 'Tempo total médio (s)', 'Mín–máx (s)', 'Verificador RPC (s)', 'Resoluções/execuções'],
                [[r['mode'].replace('_', ' '), number(r['total_seconds']['mean']),
                  f"{number(r['total_seconds']['min'])}–{number(r['total_seconds']['max'])}",
                  number(r['verification_rpc_seconds']['mean']), f"{r['resolved_runs']}/{r['runs']}"]
                 for r in operational['summaries']], 'tab-operational-latency',
                'Tempos de ponta a ponta na GPU efetivamente conectada, N=3. Variabilidade descritiva, sem teste confirmatório de aceleração.')
            text += ('Os arquivos por tarefa decompõem geração, prefill, cópia seletiva, classificação, parsing e '
                     'RPC do verificador. Prefill, cópia e classificação são partes da geração medida; '
                     'não se somam novamente a ela. Mesmo o modo sem probe utiliza o sampler instrumentado '
                     'da pesquisa. Orçamento, tarefas conhecidas e cache aquecido diferem do replay N=20: '
                     'não se extrapola uma economia de latência, energia ou desempenho na RTX 4060.\n\n')
        from slm_steering.excellence.independent_report import render_typst as render_independent
        text += render_independent(result['independent'], number, interval, table)
    (root / 'tcc/generated/excellence.typ').write_text(text, encoding='utf-8')
