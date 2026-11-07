"""Integrate only explicitly identified evidence into the existing manuscript."""
from __future__ import annotations

import shutil

from slm_steering.final_extension.common import ROOT, REPORT, atomic_json, file_hash, now, read_json


def integrate(*, compile_pdf=True):
    status = read_json(REPORT / 'status.json')
    power = read_json(REPORT / 'power_results.json')
    metadata = read_json(REPORT / 'model_metadata.json')
    inventory = read_json(REPORT / 'task_inventory.json')
    pilot = status['pilot']
    result = read_json(REPORT / 'results.json') if status['external_results_complete'] else None
    backup = REPORT / 'manuscript_before_extension'
    if not (backup / 'manifest.json').exists():
        paths = ['tcc/main.typ', 'tcc/build/manuscrito.pdf', 'tcc/referencias.bib']
        hashes = {}
        for relative in paths:
            path = ROOT / relative
            if path.exists():
                target = backup / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                hashes[relative] = file_hash(path)
        atomic_json(backup / 'manifest.json', {'archived_at': now(), 'files': hashes})
    def format_number(x):
        return f'{x:.3f}'.replace('.', ',') if x is not None else 'indefinido'
    planning_rows = [r for r in power['planning_rows'] if r['effect'] > 0]
    table = '\n'.join(f'[{format_number(r["effect"])}], [{100*r["positive_detection"]:.1f}%], '
                      f'[{format_number(r["mean_interval_width"])}],' for r in planning_rows)
    qs = metadata['qwen']['files']['config.json']['contents']
    ds = metadata['deepseek']['files']['config.json']['contents']
    chapter = f'''= Robustez entre modelos e capacidade dos decodificadores de corretude

== Motivação e estatuto da extensão

O resultado original em 32 tokens, ΔWT-AUC(H+B − B) = −0,047, IC95%
[−0,099; 0,006], permanece inalterado. A perda logarítmica aumentou aproximadamente
0,038, IC95% [0,007; 0,071]. A validação independente anterior também permanece
uma conclusão específica sobre os gates congelados: retenção de 82,72% para H+B
e 99,08% para B, diferença de −16,36 pontos percentuais. Não se concluiu ausência
universal de informação latente nem benefício operacional conjunto.

A extensão separa duas limitações: dependência de uma família de SLM e
capacidade restrita do decodificador linear. O protocolo local
`FINAL_CROSS_MODEL_EXTENSION_v1` foi elaborado sob autorização do estudante;
não se atribui pré-registro independente ou aprovação adicional ao orientador.
Os instantes de travamento e hashes são artefatos distintos do texto do protocolo.
{('A avaliação externa está concluída; os resultados são apresentados abaixo.' if result else 'A avaliação externa desta extensão ainda não foi concluída. O desenho e os resultados técnicos abaixo não constituem teste da hipótese primária.')}

== Comparação controlada e diferenças arquiteturais

Qwen2.5-Coder-1.5B-Instruct possui {qs['num_hidden_layers']} camadas e estados
de dimensão {qs['hidden_size']}; DeepSeek-Coder-1.3B-Instruct possui
{ds['num_hidden_layers']} camadas, estados de dimensão {ds['hidden_size']} e
arquitetura nativa Llama. Os pesos e tokenizers são fixados por commits de
40 caracteres no protocolo. O
#link("https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct")[model card oficial do DeepSeek]
documenta o template e o token de término EOT. O template já inclui BOS,
de modo que a tokenização da nova extensão evita sua duplicação.

As mensagens e a especificação funcional são as mesmas, renderizadas pelos
templates próprios. O renderer histórico mantém a referência textual HumanEval
também em MBPP, uma limitação de formatação explicitada para preservar a
compatibilidade com os dados de origem. Não se igualam tokens ou coordenadas
latentes entre arquiteturas. Em cada modelo, a pergunta é se H incrementa B
nas suas próprias gerações e no conjunto ativo T > 32.

As quatro famílias são B/logística, H+B/logística, B/MLP e H+B/MLP. Os contrastes
são Δlinear e ΔMLP, sempre dentro da mesma classe; a interação secundária é
Γ = ΔMLP − Δlinear. Γ > 0 não basta para demonstrar ΔMLP > 0. O MLP tem uma camada
ReLU, largura 16 ou 32, dropout 0,1 e weight decay 0,001; três seeds entram com
pesos iguais. H é a saída do bloco de profundidade relativa 75% em 32 tokens,
com PCA64 e whitening ajustados no treino. Maior dimensão implica mais parâmetros
de entrada em H+B, mesmo mantendo a mesma largura e os mesmos grids.

A seleção usa validação de três folds agrupados por tarefa na origem. Early
stopping usa tarefas internas ao treino; uma reserva separada de 20% das tarefas
de origem ajusta calibração e limiar. Não há ajuste com os rótulos externos.
O banco Qwen de origem é reutilizado; o DeepSeek requer novas trajetórias nos
mesmos 132 IDs. Os quatro procedimentos e seus hashes devem estar travados
antes da avaliação prospectiva.

== Precisão, reserva e limites de generalização

O inventário verificou {inventory['total_tasks']} tarefas MBPP+,
{len(inventory['exposed_ids'])} IDs já expostos e {len(inventory['eligible_ids'])}
elegíveis após triagem lexical conservadora. Essa triagem não certifica
independência semântica completa ou ausência de exposição no pré-treinamento.
O tamanho planejado é {power['recommendation_tasks']} tarefas por modelo,
vinte tentativas por tarefa, condicionado ao orçamento medido; prefixos de uma
ordem SHA256 fixa definem reduções técnicas predefinidas. Os mesmos IDs externos
são usados em ambos os modelos e não são substituídos por resultado.

O planejamento usa 132 tarefas históricas, 92 mistas, e desvio-padrão das diferenças
pareadas de {format_number(power['paired_delta_sd'])}. Os efeitos populacionais
conhecidos abaixo foram simulados com reamostragem de heterogeneidade por tarefa
e intervalos normais de planejamento. Não é poder pós-hoc do efeito observado.

#table(columns: (1fr, 1fr, 1fr),
 [Efeito conhecido], [Detecção positiva], [Largura esperada IC95%],
 {table}
)

#figure(image("../../reports/final_extension/power_precision.svg", width: 100%),
 caption: [Poder e precisão de planejamento por número de tarefas independentes; sensibilidades de efeito e frequência de tarefas mistas.])

Ganhos pequenos, especialmente 0,02–0,03, permanecem difíceis de detectar.
A inferência final usa bootstrap pareado de tarefas, 10.000 réplicas, IC bilateral
de 95%, condicional aos probes ajustados. Ausência de diferença demonstrada
não é equivalência. A simulação normal é uma aproximação, e transportar a
heterogeneidade Qwen/HumanEval para DeepSeek/MBPP é outra. Sensibilidades de
treinamento não substituem uma caracterização completa da incerteza de ajuste.
Trajetórias ou pares positivo-negativo não contam como tarefas independentes.

== Execução técnica e resultados disponíveis

A GPU efetivamente acessível é RTX 3050 Laptop com 6 GB. A nova extensão usa
pesos BF16, projeções FP32, batch 1 e atenção eager. O runtime CUDA é 12,6.
O código gerado é avaliado pelo EvalPlus em WSL com namespaces de rede e
arquivos, ambiente limpo e restrições de recursos. A auditoria preserva os
resultados antigos, cujo worker não tinha esse isolamento de sistema operacional.
Os testes com o modelo real conferem seeds, captura seletiva, estado após o
token 32, equivalência prefixo/sequência completa e probabilidades condicionais.
'''
    if pilot:
        chapter += f'''
O piloto DeepSeek contém {pilot['tasks']} tarefas e {pilot['attempts']} tentativas:
{pilot['mixed_tasks']} tarefas mistas, {pilot['all_success_tasks']} com todos os
acertos e {pilot['all_failure_tasks']} sem acertos. A taxa agregada de acertos foi
{100*pilot['success_rate']:.1f}%; truncamento {100*pilot['truncation_rate']:.1f}% e
erro de parsing {100*pilot['parsing_error_rate']:.1f}%. A geração mediu
{pilot['tokens_per_second_deepseek']:.2f} tokens/s e
{pilot['mean_seconds_deepseek']:.2f} segundos por tentativa em média. Esses números
descrevem desenvolvimento, não a generalização final. A viabilidade segundo as
regras fixadas foi {str(pilot['feasible']).lower()}.
'''
    else:
        chapter += '\nO piloto ainda está em execução ou aguarda conclusão; não se publicam suas taxas parciais como resultados externos.\n'
    training = power.get('repeated_source_fitting')
    if training:
        chapter += f'''
Em análise adicional de estabilidade na origem Qwen, oito subamostras de 84
tarefas repetiram seleção, ajuste e calibração das quatro famílias, avaliadas
nas mesmas 26 tarefas de auditoria disjuntas. O desvio das diferenças de WT-AUC
foi {format_number(training['training_sensitivity_sd']['linear'])} para logística e
{format_number(training['training_sensitivity_sd']['MLP'])} para MLP. Subamostras
sobrepostas e treino menor não estimam sem viés a variância do procedimento final;
esses resultados são sensibilidades, sem rótulos MBPP novos ou seleção de seed.
'''
    if result:
        chapter += '\n=== Contrastes externos por modelo\n\n'
        primary = result['primary']
        if primary['estimate'] is not None:
            meanings = {'positive': 'O intervalo inteiramente acima de zero sustenta ganho incremental para o procedimento testado.',
                        'deterioration': 'O intervalo inteiramente abaixo de zero sustenta deterioração para o procedimento testado.',
                        'inconclusive': 'O intervalo inclui zero; não se demonstrou diferença nem equivalência.'}
            chapter += f"O contraste primário DeepSeek/MLP foi {format_number(primary['estimate'])}, IC95% [{format_number(primary['ci95'][0])}; {format_number(primary['ci95'][1])}]. {meanings[primary['interpretation']]} A conclusão se restringe ao estado no bloco 17, PCA64, decoder e população ativa desta avaliação.\n\n"
        for name in ('deepseek', 'qwen'):
            for kind in ('linear', 'MLP'):
                r = result['contrasts'][name][kind]
                if r['estimate'] is None:
                    chapter += f'{name}/{kind}: não há tarefas mistas suficientes para estimar WT-AUC.\n\n'
                else:
                    chapter += f"{name}/{kind}: ΔWT-AUC = {format_number(r['estimate'])}, IC95% [{format_number(r['ci95'][0])}; {format_number(r['ci95'][1])}], {r['tasks']} tarefas mistas; interpretação: {r['interpretation']}.\n\n"
            gamma = result['contrasts'][name]['Gamma']
            if gamma:
                chapter += f"A interação Γ no {name} foi {format_number(gamma['estimate'])}, IC95% [{format_number(gamma['ci95'][0])}; {format_number(gamma['ci95'][1])}]. É secundária e não substitui o incremento absoluto de H.\n\n"
        metric_rows = []
        for r in result['metrics']:
            wt = format_number(r['within_task_auc']) if r['within_task_auc'] is not None else 'indefinida'
            metric_rows.append(f"[{r['model']}], [{r['family']}], [{wt}], [{format_number(r['log_loss'])}], [{format_number(r['brier'])}],")
        chapter += '#table(columns: (1fr, 1.2fr, 1fr, 1fr, 1fr),\n[Modelo], [Probe], [WT-AUC], [Log loss], [Brier],\n' + '\n'.join(metric_rows) + '\n)\n\n'
        chapter += '\n=== Consequências operacionais secundárias\n\n'
        for name in ('deepseek', 'qwen'):
            operational = [r for r in result['operational'] if r['model'] == name and r['budget'] == 20]
            chapter += f'Para {name}, o orçamento N=20 produziu:\n\n'
            op_rows = []
            for r in operational:
                retention = f"{100*r['retention_all_active_correct']:.1f}%" if r['retention_all_active_correct'] is not None else 'indefinida'
                op_rows.append(f"[{r['policy']}], [{r['resolved_tasks']}], [{r['mean_completion_tokens']:.1f}], [{retention}], [{r['false_aborts_started']}], [{r['lost_tasks']}],")
            chapter += '#table(columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr),\n[Política], [Resolvidas], [Tokens médios], [Retenção], [Ab. corretos], [Perdidas],\n' + '\n'.join(op_rows) + '\n)\n\n'
            for family in ('B_linear', 'HB_linear', 'B_MLP', 'HB_MLP'):
                r = next(r for r in operational if r['policy'] == family)
                cost, coverage = r['delta_mean_tokens_vs_verifier'], r['delta_coverage_vs_verifier']
                chapter += f"{family}: diferença média de tokens contra verificador {cost['estimate']:.1f}, IC95% [{cost['ci95'][0]:.1f}; {cost['ci95'][1]:.1f}]; diferença de cobertura {format_number(coverage['estimate'])}, IC95% [{format_number(coverage['ci95'][0])}; {format_number(coverage['ci95'][1])}].\n\n"
        chapter += '''As comparações secundárias têm intervalos descritivos sem controle
de multiplicidade. Log loss, Brier, AUC agregada, calibração e replay completo
estão nas tabelas reconstruídas em `reports/final_extension/`; seu estimando e
denominadores são distintos da AUC intratarefa. O resultado primário se limita
ao procedimento DeepSeek/MLP e à população ativa avaliada.
'''
    else:
        chapter += '\nA comparação externa das quatro famílias, sua interação e o replay operacional permanecem pendentes. Não há efeito novo estimado a incorporar à conclusão empírica anterior.\n'
    chapter += '''
== Decisão sequencial sob orçamento finito

O problema de aborto pode ser descrito por custos futuros esperados, incluindo
geração, gate, verificador e uma penalidade λ ao encerrar sem solução. Todos
devem usar a mesma unidade. O estado contém prefixo observado h, crença sobre
a tarefa z, orçamento restante b e número k de novas tentativas disponíveis
além da atual. Sejam J o custo ótimo sem tentativa ativa e F o custo no
checkpoint. Com custo a até o checkpoint de uma nova tentativa:

$ J_k (b,z) = min(lambda, a + E[F_(k-1) (b-a,H_0,z)]) . $

No checkpoint após observar o score, as opções são encerrar (λ), abortar (J)
e continuar a geração até o próximo checkpoint ou término. O gate atual já
foi pago; a inclui a primeira aquisição, e os gates posteriores entram nas
respectivas transições. No término, o verificador
incorre em custo v; sucesso tem valor futuro zero, e fracasso retorna a J com
orçamento e crença atualizados. Ações que excedem o orçamento são inadmissíveis;
J com zero tentativas vale λ. Ainda pode ser ótimo continuar a tentativa ativa
quando não há uma nova disponível. Limites de tentativas e tokens asseguram
horizonte finito. A recursão escolhe o menor custo esperado admissível.

Em um caso simplificado com uma única decisão, sejam p a probabilidade de sucesso,
c o custo restante e V o custo ótimo de fallback, incluindo penalidade. Suponha
que fracasso e aborto levem ao mesmo estado futuro. Então:

$ Q_"continuar" = c + (1-p)V, quad Q_"abortar" = V . $
$ Q_"continuar" <= Q_"abortar" quad <==> quad p V >= c . $

V é custo, não probabilidade de sucesso. Se k contar a atual entre as oportunidades,
V recebe índice k−1. Sob orçamento estrito, completar reduz o recurso disponível
e geralmente altera o fallback; a desigualdade simplificada deixa de valer.
Fracassos também podem atualizar z, pois independência condicional à tarefa não
implica independência marginal. Estados internos não acrescentam informação
ao prompt e prefixo completos num modelo determinístico ideal, mas podem tornar
essa informação acessível a uma classe limitada de decodificadores.

O gate empírico de limiar fixo busca retenção na origem. Ele não estima V ou c e
não implementa a política ótima dessa recursão. AUC maior não garante decisões
melhores no limiar, nem tokens poupados garantem menor latência. O solver sintético
foi conferido contra enumeração de políticas iid com orçamento finito; a validação
é interna a esse modelo, não uma demonstração de economia nos benchmarks.

Esta formulação é uma extensão conceitual delimitada. O controle de computação
por valor esperado tem antecedentes em
#link("https://doi.org/10.1016/0004-3702(91)90015-C")[Russell e Wefald (1991)] e
#link("https://arxiv.org/abs/1711.06892")[Callaway et al. (2018)].
#link("https://arxiv.org/abs/2510.15945")[BEACON (Wan et al., 2025)] relaciona
parada ótima à amostragem de respostas completas. Não se reivindica um teorema
inédito. A derivação, as hipóteses e as fronteiras completas estão documentadas
em `docs/final_extension_decision.md`.
'''
    chapter_path = ROOT / 'tcc/chapters/09_robustez_modelos.typ'
    chapter_path.write_text(chapter, encoding='utf-8')
    main = ROOT / 'tcc/main.typ'
    source = main.read_text(encoding='utf-8')
    include = '#include "chapters/09_robustez_modelos.typ"'
    if include not in source:
        anchor = '#include "chapters/06_discussao.typ"'
        if anchor not in source:
            raise ValueError('Manuscript insertion point differs; preserve source for review')
        source = source.replace(anchor, include + '\n#pagebreak()\n' + anchor)
        main.write_text(source, encoding='utf-8')
    if compile_pdf:
        import typst
        output = ROOT / 'tcc/build/manuscrito.pdf'
        typst.compile(str(main), output=str(output), root=str(ROOT))
    atomic_json(REPORT / 'manuscript_integration.json', {'at': now(), 'external_results_included': result is not None,
                'pilot_results_included': pilot is not None, 'chapter_sha256': file_hash(chapter_path),
                'main_sha256': file_hash(main), 'compiled': compile_pdf,
                'previous_manuscript': str((backup / 'manifest.json').relative_to(ROOT))})
