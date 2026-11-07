"""Render the integrated manuscript from completed, unchanged empirical artifacts.

Editorial rendering only: no generation, probe fitting, threshold selection,
new confidence intervals, or changes to the frozen scientific procedure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


INPUTS = {
    'original': 'reports/protocol_v0.2/results.json',
    'transfer': 'reports/transfer_v0.2/results.json',
    'exploratory': 'reports/excellence/models/results.json',
    'evidence': 'reports/final_review/evidence.json',
    'independent': 'reports/excellence/independent_v1/results.json',
    'final': 'reports/final_extension/results.json',
    'power': 'reports/final_extension/power_results.json',
    'design': 'reports/final_extension/final_design.json',
    'status': 'reports/final_extension/status.json',
    'freeze': 'reports/final_extension/freeze.json',
    'deepseek_probes': 'reports/final_extension/deepseek_probe_manifest.json',
    'qwen_probes': 'reports/final_extension/qwen_probe_manifest.json',
    'finalization': 'reports/finalization/statistical_verification.json',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value, digits=3):
    if value is None:
        return 'não identificável'
    return f'{value:.{digits}f}'.replace('-', '−').replace('.', ',')


def interval(value, digits=3, scale=1):
    return '[' + '; '.join(number(v*scale, digits) for v in value) + ']'


def effect(row, digits=3, scale=1):
    return number(row['estimate']*scale, digits) + ', IC95% ' + interval(row['ci95'], digits, scale)


def model_name(name):
    return {'deepseek': 'DeepSeek', 'qwen': 'Qwen'}[name]


def table(headers, rows, label, caption, widths=None):
    def cell(value):
        return '[' + str(value).replace('_', r'\_') + ']'
    columns = str(len(headers)) if widths is None else '(' + ', '.join(f'{x}fr' for x in widths) + ')'
    cells = ',\n'.join(', '.join(cell(v) for v in row) for row in rows)
    return ('#figure(\n { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);\n'
            f' table(columns: {columns}, align: left, inset: 5pt,\n table.header(' +
            ', '.join(cell('*' + h + '*') for h in headers) + '),\n' + cells + '\n ) },\n'
            ' kind: table, caption: [' + caption + ']\n) <' + label + '>\n')


def load(root):
    data = {k: json.loads((root/p).read_text(encoding='utf-8')) for k, p in INPUTS.items()}
    if data['final']['status'] != 'completed' or not data['status']['external_results_complete']:
        raise ValueError('Integrated edition requires completed external evidence')
    if len(data['design']['external_ids']) != 100:
        raise ValueError('The prose describes the frozen 100-task cohort; revise explicitly for another dataset')
    for path, expected in data['freeze']['files'].items():
        if sha(root/path) != expected:
            raise ValueError(f'Frozen scientific source differs: {path}')
    metrics = {(r['model'], r['family']): r for r in data['final']['metrics']}
    for model in ('deepseek', 'qwen'):
        for kind in ('linear', 'MLP'):
            difference = metrics[model, f'HB_{kind}']['within_task_auc'] - metrics[model, f'B_{kind}']['within_task_auc']
            if abs(difference-data['final']['contrasts'][model][kind]['estimate']) > 1e-12:
                raise ValueError('Contrast and metric table disagree')
    return data, metrics


def figures(root, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    destination = root/'reports/manuscript_revision_20261107/figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ordering = [('deepseek', 'MLP'), ('deepseek', 'linear'), ('qwen', 'MLP'), ('qwen', 'linear')]
    for y, (model, kind) in enumerate(ordering):
        row = data['final']['contrasts'][model][kind]
        value, (lo, hi) = row['estimate'], row['ci95']
        color = '#174c72' if y == 0 else '#657482'
        ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt='o', color=color, capsize=4,
                    markersize=7 if y == 0 else 5, linewidth=1.5)
    ax.set_yticks(range(4), [f'{model_name(m)} · {"logística" if k=="linear" else k} (m={data["final"]["contrasts"][m][k]["tasks"]})'
                           for m, k in ordering])
    ax.invert_yaxis()
    ax.axvline(0, color='#888888', ls='--', lw=1)
    ax.set_xlabel('Diferença de WT-AUC: H+B − B')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'.replace('.', ',')))
    ax.grid(axis='x', alpha=.15)
    fig.tight_layout()
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(destination/f'contrastes.{suffix}', dpi=180)
    plt.close(fig)
    import csv
    with (root/'reports/final_extension/calibration.csv').open(encoding='utf-8') as stream:
        bins=list(csv.DictReader(stream))
    fig, axes=plt.subplots(1,2,figsize=(6.5,3.6),sharex=True,sharey=True)
    for ax,model in zip(axes,('deepseek','qwen')):
        ax.plot([0,1],[0,1],color='#999999',ls=':',lw=1)
        for family,label,color,style in [('B_linear','B / logística','#c16a2b','-'),('HB_linear','H+B / logística','#174c72','-'),
                                         ('B_MLP','B / MLP','#c16a2b','--'),('HB_MLP','H+B / MLP','#174c72','--')]:
            rows=[r for r in bins if r['model']==model and r['family']==family and int(r['count'])>0]
            ax.plot([float(r['mean_probability']) for r in rows],[float(r['fraction_correct']) for r in rows],
                    marker='o',ms=3,color=color,ls=style,label=label)
        ax.set_title(model_name(model))
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Probabilidade prevista')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Frequência de corretos')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,.15,1,1))
    for suffix in ('svg','pdf','png'):
        fig.savefig(destination/f'calibracao.{suffix}',dpi=180)
    plt.close(fig)


def render(root: Path, *, compile_pdf=False):
    data, metrics = load(root)
    input_paths=list(INPUTS.values())
    original, final = data['original'], data['final']
    h2 = original['signal']['H2']
    old, old_loss = h2['delta_hb_minus_b'], h2['delta_log_loss_hb_minus_b']
    primary = final['primary']
    independent = data['evidence']['excellence']['independent']
    comparison = final['contrasts']
    v = {
        'OLD_DELTA': number(old['estimate']), 'OLD_CI': interval(old['ci95']),
        'OLD_LOSS_DELTA': number(old_loss['estimate']), 'OLD_LOSS_CI': interval(old_loss['ci95']),
        'OLD_DELTA_EN': f'{old["estimate"]:.3f}', 'OLD_CI_EN': str([round(x, 3) for x in old['ci95']]),
        'PRIMARY_DELTA': number(primary['estimate'], 4), 'PRIMARY_CI': interval(primary['ci95'], 4),
        'PRIMARY_DELTA_EN': f'{primary["estimate"]:.4f}', 'PRIMARY_CI_EN': str([round(x, 4) for x in primary['ci95']]),
        'PRIMARY_MIXED': str(primary['tasks']),
        'INDEPENDENT_PP': number(independent['primary']['estimate']*100, 2),
        'INDEPENDENT_PP_CI': interval(independent['primary']['ci95'], 2, 100),
        'OLD_B_WT': number(next(r for r in original['signal']['metrics'] if r['checkpoint']=='32' and r['predictor']=='B')['within_task_auc']),
        'OLD_H_WT': number(next(r for r in original['signal']['metrics'] if r['checkpoint']=='32' and r['predictor']=='H')['within_task_auc']),
        'OLD_HB_WT': number(next(r for r in original['signal']['metrics'] if r['checkpoint']=='32' and r['predictor']=='HB')['within_task_auc']),
        'DS_B_MLP_WT': number(metrics['deepseek', 'B_MLP']['within_task_auc'], 4),
        'DS_HB_MLP_WT': number(metrics['deepseek', 'HB_MLP']['within_task_auc'], 4),
        'QW_MLP_DELTA': number(comparison['qwen']['MLP']['estimate'], 4),
        'QW_MLP_CI': interval(comparison['qwen']['MLP']['ci95'], 4),
        'COMMON_DELTA': number(final['secondary_cross_model_common_mixed']['estimate'], 4),
        'COMMON_CI': interval(final['secondary_cross_model_common_mixed']['ci95'], 4),
        'TRAIN_SD_LINEAR': number(data['power']['repeated_source_fitting']['training_sensitivity_sd']['linear']),
        'TRAIN_SD_MLP': number(data['power']['repeated_source_fitting']['training_sensitivity_sd']['MLP']),
        'ACTIVE_HOURS': number(data['status']['resource_accounting']['charged_active_session_hours'], 2),
        'RAW_HOURS': number(data['status']['resource_accounting']['raw_session_wall_hours'], 2),
        'SLEEP_HOURS': number(data['status']['resource_accounting']['excluded_os_suspension_hours'], 2),
    }
    risks = [r for r in original['signal']['risk_sets'] if r['checkpoint'] in ('8', '16', '32', '64')]
    v['OLD_RISK_TABLE'] = table(['Tokens', 'Ativas', 'Tarefas ativas', 'Mistas'],
        [[r['checkpoint'], r['trajectories'], r['tasks'], r['mixed_tasks']] for r in risks],
        'tab-original-risk', 'Qwen em HumanEval+: população ativa e tarefas que identificam WT-AUC.')
    rows = []
    for family, title in [('B', 'B'), ('H', 'H'), ('HB', 'H+B')]:
        r = next(r for r in original['signal']['metrics'] if r['checkpoint']=='32' and r['predictor']==family)
        rows.append([title, number(r['roc_auc']), number(r['within_task_auc'])+' '+interval(r['within_task_auc_ci95']),
                     number(r['log_loss']), number(r['brier'])])
    v['OLD_METRICS_TABLE'] = table(['Probe', 'AUC agregada', 'WT-AUC e IC95%', 'Log loss', 'Brier'], rows,
        'tab-original-metrics', 'Probes lineares da primeira etapa, fora da amostra por tarefa em HumanEval+, t=32.', [0.6, 1, 1.8, 1, 1])
    policies = {r['policy']: r for r in original['policies']['summaries'] if r['budget']==20}
    v['OLD_POLICY_TABLE'] = table(['Política', 'Resolvidas / 132', 'Tokens / tarefa'],
        [[title, round(policies[key]['resolution']*132), number(policies[key]['completion_tokens_per_task'], 1)]
         for key, title in [('fixed','Vinte completas'), ('verifier','Verificador'), ('B','B'), ('HB','H+B')]],
        'tab-original-policy', 'Primeira etapa: replay com N=20 e toda a população de tarefas.')
    for key, name in [('HB minus verifier', 'OLD_POLICY'), ('HB minus B', 'OLD_POLICY_VS_B')]:
        row = next(r for r in original['policies']['comparisons'] if r['budget']==20 and r['comparison']==key)
        v[name+'_COVERAGE'] = effect(row['resolution_delta'])
        v[name+'_TOKENS'] = effect(row['token_delta'], 2)
    v['INDEPENDENT_TABLE'] = table(['Gate em t=8', 'Corretos mantidos', 'Corretos ativos', 'Retenção e IC95%'],
        [[r['family'].replace('HB','H+B'), r['retained_correct'], r['active_correct'],
          number(r['recall']*100, 2)+'% '+interval(r['ci95'], 2, 100)] for r in independent['recall']],
        'tab-independent-gates', 'Validação independente anterior, 120 tarefas Qwen/MBPP+. Intervalos por tarefas, em percentuais.', [0.8,1,1,1.6])
    names = [('HB_original','H+B original'), ('HB_pca64','H+B com PCA64'), ('H_residual_B','Correção residual de B')]
    v['EXPLORATORY_TABLE'] = table(['Leitura', 'WT-AUC', 'ΔWT-AUC contra B e IC95%'],
        [[title, number(r['within_task_auc']), effect(r['delta_wt_auc_vs_B'])]
         for key,title in names for r in data['exploratory']['summaries'] if r['family']==key],
        'tab-exploratory-readouts', 'Diagnósticos posteriores no mesmo banco Qwen de origem; intervalos exploratórios condicionais.', [1.4,0.8,2.2])
    oracle_rows = [r for r in data['evidence']['excellence']['economics']['summaries']
                   if r['dataset']=='HumanEval+' and r['checkpoint']==32 and r['budget']==20 and r['policy'] in ('verifier','oracle')]
    if len(oracle_rows) != 2:
        raise ValueError('Historical oracle comparison is not uniquely identified')
    v['OLD_ORACLE_TABLE'] = table(['Política', 'Resolução', 'Tokens / tarefa'],
        [[r['policy'], number(r['resolution']), number(r['tokens_per_task'], 2)] for r in oracle_rows],
        'tab-original-oracle', 'Limite contrafactual na primeira etapa, N=20 e checkpoint de 32 tokens.')
    v['POWER_TABLE'] = table(['Ganho verdadeiro simulado', 'Detecção positiva', 'Largura esperada do IC95%'],
        [[number(r['effect']), number(100*r['positive_detection'], 1)+'%', number(r['mean_interval_width'])]
         for r in data['power']['planning_rows']], 'tab-cross-power',
        'Planejamento para cem tarefas, condicional aos scores e à heterogeneidade histórica usada como aproximação.')
    v['FINAL_POPULATION_TABLE'] = table(['Gerador', 'Geradas', 'Ativas', 'Tarefas ativas', 'Tarefas mistas'],
        [[model_name(name), r['total_generated_candidates'], r['active_candidates'], r['active_tasks'], r['mixed_active_tasks']]
         for name in ('deepseek','qwen') for r in [metrics[name,'B_MLP']]],
        'tab-cross-population', 'Teste externo completo: cem tarefas e vinte tentativas por modelo, com elegibilidade T>32.')
    v['FINAL_CONTRAST_TABLE'] = table(['Gerador / classe', 'ΔWT-AUC e IC95%', 'Mistas', 'Estatuto'],
        [[model_name(name)+' / '+('logística' if kind=='linear' else kind), effect(comparison[name][kind], 4),
          comparison[name][kind]['tasks'], 'Primário' if (name,kind)==('deepseek','MLP') else 'Secundário']
         for name,kind in [('deepseek','MLP'),('deepseek','linear'),('qwen','MLP'),('qwen','linear')]],
        'tab-cross-contrasts', 'H+B menos B dentro da mesma classe. Bootstrap pareado de 10.000 réplicas.', [1.3,2.5,0.5,0.8])
    v['INTERACTION_TABLE'] = table(['Gerador', 'Γ e IC95%', 'Tarefas mistas'],
        [[model_name(name), effect(comparison[name]['Gamma'], 4), comparison[name]['Gamma']['tasks']] for name in ('deepseek','qwen')],
        'tab-cross-interaction', 'Mudança secundária do incremento de H ao passar da logística para MLP.')
    rows=[]
    for name in ('deepseek','qwen'):
        for family in ('B_linear','HB_linear','B_MLP','HB_MLP'):
            r=metrics[name,family]
            label=('H+B' if family.startswith('HB') else 'B')+' / '+('logística' if family.endswith('linear') else 'MLP')
            rows.append([model_name(name), label, number(r['roc_auc']), number(r['within_task_auc']), number(r['log_loss']), number(r['brier'])])
    v['FINAL_METRICS_TABLE'] = table(['Gerador','Features / probe','AUC agregada','WT-AUC','Log loss','Brier'],rows,
        'tab-cross-metrics','Métricas pontuais do teste externo aos 32 tokens. Populações ativas próprias de cada modelo.', [0.9,1.3,0.9,0.8,0.8,0.8])
    v['FINAL_LOSS_TABLE'] = table(['Gerador / classe', 'Δlog loss e IC95%', 'ΔBrier e IC95%'],
        [[model_name(name)+' / '+('logística' if kind=='linear' else kind), effect(comparison[name][kind]['delta_log_loss'],4),
          effect(comparison[name][kind]['delta_brier'],4)] for name in ('deepseek','qwen') for kind in ('linear','MLP')],
        'tab-cross-losses','Contrastes secundários H+B−B. Valores negativos indicam menor perda; IC pontuais sem ajuste de multiplicidade.', [1.1,2.1,2.1])
    op = {(r['model'], r['budget'], r['policy']):r for r in final['operational']}
    audited_op = {(r['model'], r['budget'], r['policy']):r for r in data['finalization']['operational']}
    for model, prefix in (('deepseek', 'DS'), ('qwen', 'QW')):
        for family, label in (('verifier', 'VERIFIER'), ('B_MLP', 'B_MLP'), ('HB_MLP', 'HB_MLP')):
            v[f'{prefix}_{label}_COST_PER_SOLUTION'] = number(op[model,20,family]['aggregate_tokens_per_resolved_task'],2)
            v[f'{prefix}_{label}_ABORTS'] = str(audited_op[model,20,family]['aborts_started'])
            v[f'{prefix}_{label}_STARTED'] = str(audited_op[model,20,family]['attempts_started'])
    policy_order=[('fixed','Vinte completas'),('verifier','Verificador'),('B_linear','B / logística'),
                  ('HB_linear','H+B / logística'),('B_MLP','B / MLP'),('HB_MLP','H+B / MLP'),('oracle','Oracle')]
    for model, key in [('deepseek','DS'),('qwen','QW')]:
        rows=[]
        for policy,title in policy_order:
            r=op[model,20,policy]
            rows.append([title,r['resolved_tasks'],number(r['mean_completion_tokens'],2),
                         number(r['retention_all_active_correct']*100,2)+'%',r['false_aborts_started'],r['lost_tasks']])
        v[key+'_OPERATIONAL_TABLE']=table(['Política','Resolvidas','Tokens / tarefa','Retenção','Ab. corretos','Perdidas'],rows,
            f'tab-{model}-operational',f'{model_name(model)}: replay N=20 nas cem tarefas externas. Retenção entre corretos ativos potenciais.',
            [1.4,0.7,1,0.9,0.7,0.7])
    v['OPERATIONAL_EFFECT_TABLE']=table(['Gerador / gate','Δtokens e IC95%','Δresolução e IC95%'],
        [[model_name(name)+' / '+family.replace('HB_','H+B / ').replace('B_','B / '),
          effect(op[name,20,family]['delta_mean_tokens_vs_verifier'],2),effect(op[name,20,family]['delta_coverage_vs_verifier'],3)]
         for name in ('deepseek','qwen') for family in ('B_MLP','HB_MLP')],
        'tab-cross-operational-effects','Diferenças pareadas contra parada por verificador. Resultados operacionais secundários.', [1.3,2,2])
    # Accounting uses the exact OS-evidence correction already frozen with the experiment.
    import sys
    if str(root/'src') not in sys.path:
        sys.path.insert(0,str(root/'src'))
    from slm_steering.final_extension.resources import accounting
    intervals=json.loads((root/'reports/final_extension/resource_accounting_adjustments.json').read_text(encoding='utf-8'))['intervals']
    resource_rows=[]
    for model,phase,title in [('deepseek','pilot','Piloto DeepSeek'),('qwen','technical','Técnico Qwen'),
                             ('deepseek','source','Origem DeepSeek'),('deepseek','external','Externo DeepSeek'),('qwen','external','Externo Qwen')]:
        p=root/f'runs/final_extension/v1/{model}/{phase}/manifest.json'
        m=json.loads(p.read_text(encoding='utf-8'))
        input_paths.append(p.relative_to(root).as_posix())
        peaks=[s['hardware']['peak_reserved_mib'] for s in m['sessions'] if s.get('hardware')]
        resource_rows.append([title,m['completed_attempts'],number(accounting(m['sessions'],intervals)['charged_active_session_hours'],2),number(max(peaks)/1024,2)])
    v['RESOURCE_TABLE']=table(['Coleta','Tentativas','Horas ativas','Pico reservado / GiB'],resource_rows,
        'tab-cross-resources','Sessões da extensão final na RTX 3050 Laptop. Valores de parede ativa e memória do PyTorch; treino dos probes em CPU.')
    selected=[]
    for name in ('deepseek','qwen'):
        for family in ('B_linear','HB_linear','B_MLP','HB_MLP'):
            r=data[name+'_probes'][family]
            label=('H+B' if family.startswith('HB') else 'B')+' / '+('logística' if family.endswith('linear') else 'MLP')
            selected.append([model_name(name),label,('C=' if family.endswith('linear') else 'Largura ')+number(r['chosen']['parameter'],2 if family.endswith('linear') else 0),
                             r['final_epochs'] if r['final_epochs'] is not None else '—',number(r['threshold'],4)])
    v['FINAL_PROBE_TABLE']=table(['Gerador','Procedimento','Seleção','Épocas','Limiar'],selected,
        'tab-source-probes','Configurações escolhidas exclusivamente em HumanEval+. Cada procedimento usa 106 tarefas para ajuste/seleção e 26 para calibração e limiar.',[0.8,1.3,1,0.6,0.8])
    figures(root,data)
    input_paths.append('reports/final_extension/calibration.csv')
    for name in ('risk_set_attrition', 'within_task_discrimination', 'paired_policy_effects'):
        input_paths.append(f'reports/excellence/print_ready/core/figures/{name}.svg')
    templates=root/'tcc/templates'
    generated=root/'tcc/generated'
    generated.mkdir(exist_ok=True)
    outputs={}
    for name in ('resumo','etapa_qwen','comparacao','discussion_findings','conclusao'):
        text=(templates/f'{name}.typ.tpl').read_text(encoding='utf-8')
        for key,value in v.items():
            text=text.replace('@@'+key+'@@',str(value))
        if re.search(r'@@[A-Z0-9_]+@@',text):
            raise ValueError(f'Unresolved editorial marker in {name}')
        output=generated/f'{name}.typ'
        output.write_text(text,encoding='utf-8')
        outputs[output.relative_to(root).as_posix()]=sha(output)
    for name in ('contrastes', 'calibracao'):
        path=root/f'reports/manuscript_revision_20261107/figures/{name}.svg'
        outputs[path.relative_to(root).as_posix()]=sha(path)
    manifest={'edition':'integrated_cross_model_v1','rendered_at_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Editorial integration of completed evidence; no statistical refit or new inference',
              'inputs':{p:sha(root/p) for p in input_paths},'generated':outputs,'compiled':compile_pdf}
    if compile_pdf:
        import typst
        target=root/'tcc/build/manuscrito.pdf'
        typst.compile(str(root/'tcc/main.typ'),output=str(target),root=str(root))
        manifest['pdf_sha256']=sha(target)
    active=[root/'tcc/main.typ',root/'tcc/lib.typ',root/'tcc/referencias.bib',root/'tcc/referencias_extensao.bib',
            root/'scripts/render_integrated_manuscript.py',root/'scripts/build_manuscript.py',root/'scripts/audit_integrated_manuscript.py']
    active+=list(templates.glob('*.typ.tpl'))
    active+=[root/'tcc/chapters'/p for p in ('01_introducao.typ','02_relacionados.typ','03_fundamentacao.typ','04_metodologia.typ',
                                        '07_discussao_integrada.typ','08_reprodutibilidade.typ')]
    manifest['editorial_sources']={p.relative_to(root).as_posix():sha(p) for p in active}
    destination=root/'reports/manuscript_revision_20261107'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest
