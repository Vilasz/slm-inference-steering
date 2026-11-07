from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Build and execute the final evidence notebook (CPU; no generation)')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Sinais de corretude em geração de código
## Evidências disponíveis, reanálise e protocolo revisado

**João Felipe Vilas Boas · Orientador: Prof. Dário Oliveira**

Este notebook reúne resultados observados e sua proveniência. A reanálise das
representações **finais** é exploratória. O piloto HumanEval+ valida a infraestrutura.
As perguntas confirmatórias sobre prefixos de 32 tokens e aborto precoce continuam
pendentes do banco completo e do congelamento científico.

Execute as células na ordem. Elas leem os artefatos pequenos de `reports/final_review/`
e não carregam o modelo na GPU. As figuras e os números também alimentam o manuscrito.
'''), code('''from pathlib import Path
import hashlib
import json
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
REPORT = ROOT / "reports/final_review"
evidence = json.loads((REPORT / "evidence.json").read_text(encoding="utf-8"))
pd.set_option("display.max_columns", 12)
pd.set_option("display.precision", 4)
display(Markdown(f"**Data de referência:** {evidence['as_of']} · **Escopo:** evidência exploratória e técnica"))
'''), md('''## 1. O que já pode ser afirmado

A primeira comparação usa as mesmas 150 tentativas históricas de 30 tarefas,
com o verificador local original. Não misturamos esses rótulos com HumanEval+.
O orçamento N é o máximo de tentativas; resolução é a fração de tarefas com pelo
menos um sucesso no prefixo de N candidatos, na ordem armazenada.
'''), code('''sampling = pd.DataFrame(evidence["legacy_sampling"])
display(sampling[["budget", "tasks", "solved", "resolution", "resolution_ci95", "fixed_tokens", "verifier_tokens", "token_saving_fraction"]])
display(Image(filename=str(REPORT / "figures/legacy_sampling.png")))
first, last = evidence["legacy_sampling"][0], evidence["legacy_sampling"][-1]
display(Markdown(f"Com N=1, foram resolvidas **{first['solved']}/{first['tasks']}** tarefas; "
                f"com N=5, **{last['solved']}/{last['tasks']}**. Gerar todos os candidatos consumiu "
                f"**{last['fixed_tokens']:,} tokens**; parar após aprovação consumiria "
                f"**{last['verifier_tokens']:,} tokens** no mesmo banco, uma redução de "
                f"**{100*last['token_saving_fraction']:.2f}%**. Essa economia é atribuível ao verificador. "
                "Não mede aborto de um candidato ainda incompleto nem economia de tempo online."))
'''), md('''## 2. Quantas tarefas sustentam a discriminação intratarefa?

Uma tarefa com todos os candidatos corretos ou todos incorretos não permite
calcular AUC intratarefa. Ela continua contribuindo para perdas probabilísticas
e métricas de resolução. As categorias abaixo descrevem cinco tentativas observadas;
não significam que a probabilidade verdadeira seja zero ou um.
'''), code('''composition = pd.DataFrame(evidence["legacy_task_composition"])
display(composition.groupby("category").agg(tarefas=("task_id", "size"), tentativas=("attempts", "sum"), acertos=("correct", "sum")))
display(Image(filename=str(REPORT / "figures/legacy_task_composition.png")))
'''), md('''## 3. Reanálise exploratória dos estados finais

Cinco folds externos por tarefa, três internos para selecionar camada e regularização.
O resultado avalia o procedimento de seleção. Não usamos a melhor camada encontrada
no teste como resultado final. Os estados históricos foram retokenizados e não
dispõem da validação causal da nova coleta.

- **B tardio:** log(1 + comprimento do prompt) e log(1 + comprimento final).
- **H final:** representação no último token da resposta.
- **H+B tardio:** concatenação das duas fontes.

O comprimento final é permitido nesta análise tardia; seria vazamento em uma
política precoce. Portanto, esta comparação não é o endpoint H+B versus B em 32 tokens.
'''), code('''late = evidence["legacy_reanalysis"]
metrics = pd.DataFrame([{**{k: row[k] for k in ["family", "roc_auc", "within_task_auc", "log_loss", "brier", "mixed_tasks"]},
                         "WT-AUC IC95%": row["wt_auc_interval"]["ci95"]} for row in late["metrics"]])
display(metrics)
display(Image(filename=str(REPORT / "figures/legacy_late_probes.png")))
display(pd.DataFrame([{"efeito": "ΔWT-AUC: H+B tardio − B tardio", **late["delta_wt_auc"]},
                      {"efeito": "Δlog loss: H+B tardio − B tardio", **late["delta_log_loss"]}]))
'''), md('''**Leitura:** discriminação, ganho incremental e utilidade operacional são perguntas
distintas. O intervalo de ΔWT-AUC inclui zero nesta reanálise. A redução da log loss
sustenta melhoria probabilística no regime histórico, condicionada aos modelos ajustados,
mas não demonstra ganho de ordenação intratarefa nem utilidade precoce.

Os intervalos usam 2.000 reamostragens pareadas de tarefas. Na WT-AUC, a população
é o subconjunto de tarefas mistas; na perda, todas as tarefas. O bootstrap mantém
fixos os modelos OOF e não representa toda a incerteza de retreinamento.

A antiga AUC máxima de um único split não é diretamente comparável à avaliação
aninhada completa acima. Nenhum dos dois resultados demonstra predição antes do término.
'''), md('''## 4. Estado da coleta e validação técnica

Os bancos v0.1 e v0.2 têm definições diferentes de rótulo/margem e sementes
dependentes da versão. Os resultados não são agrupados. Uma sessão parcial pode
demonstrar viabilidade técnica, mas não permite escolher a temperatura nem estimar
o efeito confirmatório. Tempos de desenvolvimento podem incluir contenção local.
'''), code('''display(pd.DataFrame(evidence["pilot"]))
display(pd.read_csv(REPORT / "tables/pilot_temperatures.csv"))
display(Markdown("**Histórico de exposição:** " + str(evidence["exposure"]) +
                ". Presença em arquivos demonstra geração prévia, não quais resultados foram inspecionados."))
'''), md('''## 5. Perguntas principais e evidência ainda necessária

| Pergunta | Teste planejado | Estado desta entrega |
|---|---|---|
| Q1: previsibilidade precoce | AUC e WT-AUC por checkpoint em tarefas externas | Banco principal pendente |
| Q2: ganho em 32 tokens | ΔWT-AUC e Δlog loss pareadas, H+B contra B | Não estimado |
| Q3: utilidade do aborto | Resolução/tokens contra verificador e B; validação online | Não estimado |
| Transferência MBPP+ | Aplicação de sistema congelado sem ajuste local | Implementada, execução pendente |

Uma permutação intratarefa testa associação; não testa isoladamente informação
adicional sobre B. A v0.2 usa testes por fold e combinação conservadora. A política
busca todos os limiares admissíveis, inclui ausência de aborto e verifica término
antes do gate. O alvo funcional admite programas truncados que passam na suíte.
'''), code('''confirmatory_path = ROOT / "reports/protocol_v0.2/results.json"
if confirmatory_path.exists():
    confirmatory = json.loads(confirmatory_path.read_text(encoding="utf-8"))
    display(Markdown("### Artefato confirmatório disponível posteriormente"))
    display(confirmatory["signal"]["H2"])
    display(confirmatory["policies"].get("H3", {}))
else:
    display(Markdown("**Não há artefato confirmatório para apresentar nesta versão.**"))
'''), md('''## 6. Proveniência e verificações

As tabelas e figuras derivam de `evidence.json` e dos arquivos listados abaixo.
Se um banco crescer após esta entrega, o hash deixa de coincidir: reconstrua a
evidência e o notebook. Um arquivo bruto ausente não invalida a visualização dos
resultados arquivados, mas impede sua rechecagem local.
'''), code('''provenance = []
for relative, expected in evidence["inputs"].items():
    path = ROOT / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    provenance.append({"arquivo": relative, "sha256": expected[:16] + "…",
                       "checagem": "confere" if actual == expected else "ausente" if actual is None else "alterado"})
display(pd.DataFrame(provenance))

import xml.etree.ElementTree as ET
test_report = ROOT / "reports/research/test_results_v0.2.xml"
if test_report.exists():
    suites = ET.parse(test_report).getroot().iter("testsuite")
    display(pd.DataFrame([{k: suite.attrib.get(k) for k in ["tests", "failures", "errors", "skipped"]} for suite in suites]))
'''), md('''## 7. Reprodução e continuidade

Na raiz do repositório, usando o Python do ambiente:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/build_research_delivery.py --reanalyze-legacy
.venv/Scripts/python.exe scripts/build_final_notebook.py --execute
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
```

A reanálise usa CPU e os arquivos históricos de `runs/`. O notebook em si apenas
lê as tabelas e figuras. Veja `docs/research_workflow.md` para coleta, congelamento,
probes e transferência. O protocolo exige aprovação científica real antes da
coleta definitiva; esta entrega não registra aprovação em nome do orientador.

O manuscrito está em `tcc/main.typ` e o PDF em `tcc/build/manuscrito.pdf`.
''')]
    saved_evidence = json.loads((ROOT / 'reports/final_review/evidence.json').read_text(encoding='utf-8'))
    pilot_cell = next(c for c in cells if c.cell_type == 'code' and 'display(pd.DataFrame(evidence["pilot"]))' in c.source)
    pilot_cell['source'] += '\nprogress = REPORT / "figures/pilot_sampling_progress.png"\nif progress.exists():\n    display(Image(filename=str(progress)))\n'
    pilot_cell['source'] += '\nfor name in ["pilot_task_temperature_grid", "pilot_lengths_and_verifier"]:\n    path = REPORT / f"figures/{name}.png"\n    if path.exists():\n        display(Image(filename=str(path)))\n'
    late_cell = next(c for c in cells if c.cell_type == 'code' and 'late = evidence["legacy_reanalysis"]' in c.source)
    late_cell['source'] += '\nby_task = REPORT / "figures/legacy_incremental_by_task.png"\nif by_task.exists():\n    display(Image(filename=str(by_task)))\n'
    cells[-1]['source'] = cells[-1]['source'].replace(
        'O protocolo exige aprovação científica real antes da\ncoleta definitiva; esta entrega não registra aprovação em nome do orientador.',
        'A execução integral foi autorizada pelo estudante. O desenho e seus fontes\nestão copiados em `reports/research/execution_freeze_v0.2/`. A aprovação do\norientador permanece pendente; não se reivindica pré-registro independente.')
    if saved_evidence.get('confirmatory_results_available'):
        cells[0]['source'] = '''# Sinais de corretude em geração de código
## Experimento de desenho fixo: HumanEval+ e transferência MBPP+

Os resultados principais vêm de 2.640 trajetórias em 132 tarefas, após piloto
separado. Todos os números abaixo são lidos dos artefatos verificados.
Os dados históricos permanecem em seção separada. Não se reivindica pré-registro
independente; exposição histórica e aprovação pendente são divulgadas.
'''
        cells[1]['source'] = cells[1]['source'].replace('evidência exploratória e técnica', 'avaliação principal de desenho fixo')
        question_cell = next(c for c in cells if c.cell_type == 'markdown' and c.source.startswith('## 5.'))
        question_cell['source'] = '''## Decisões das perguntas principais

As decisões derivam dos critérios fixados, sem escolher o melhor resultado
secundário. Bootstrap condicional aos modelos OOF; cada tarefa é uma unidade.
Permutação de associação não substitui comparação incremental com B.
'''
        main_cells = [md('''## Resultados principais em 32 tokens

Elegibilidade exige T > t. AUC agregada e AUC intratarefa respondem perguntas
diferentes. Intervalos usam 2.000 reamostragens de tarefas e não refazem o treino.
'''), code('''main = evidence["main"]
signal, policies = main["signal"], main["policies"]
display(pd.DataFrame(signal["risk_sets"]))
display(pd.DataFrame([r for r in signal["metrics"] if r["checkpoint"] == "32"]))
display(signal["H2"])
for name in ["risk_set_attrition", "primary_predictor_comparison", "primary_task_deltas", "primary_roc_pr"]:
    display(Image(filename=str(ROOT / f"reports/protocol_v0.2/figures/{name}.png")))
display(Image(filename=str(ROOT / "reports/protocol_v0.2/figures/calibration_32.png")))
display(Image(filename=str(ROOT / "reports/protocol_v0.2/figures/within_task_discrimination.png")))
'''), md('''## Utilidade operacional: custo, resolução e risco

As políticas usam as mesmas sementes e contam apenas tentativas iniciadas.
A margem de resolução é −0,03. Economia de tokens não implica menor latência.
'''), code('''display(pd.DataFrame(policies["summaries"]))
display(pd.DataFrame(policies["comparisons"]))
display(policies["H3"])
for name in ["compute_resolution", "paired_policy_effects", "policy_cost_and_false_aborts"]:
    display(Image(filename=str(ROOT / f"reports/protocol_v0.2/figures/{name}.png")))
decision = "sustentada pelos critérios definidos" if signal["H2"]["supported"] else "não sustentada pelos critérios definidos"
display(Markdown(f"**Q2:** {decision}. **Benefício operacional conjunto:** "
                 f"{policies['H3']['operational_benefit_claim_supported']}. "
                 "Um resultado negativo ou inconclusivo é preservado na conclusão."))
if "online" in evidence:
    display(pd.DataFrame(evidence["online"]["validations"]))
    display(Markdown("Tempos online descritivos: tarefas pré-selecionadas por ID, "
                     "ordem das políticas rotacionada e aquecimento excluído; sem inferência de latência."))
'''), md('''## Transferência externa congelada

Modelos, calibradores e limiares ajustados somente em HumanEval+.
Os rótulos MBPP+ são usados para avaliar, sem recalibração ou retreino.
'''), code('''if "transfer" in evidence:
    display(pd.DataFrame(evidence["transfer"]["metrics"]))
    display(pd.DataFrame(evidence["transfer"]["deltas"]))
    display(pd.DataFrame(evidence["transfer"]["policies"]["summaries"]))
    display(Image(filename=str(ROOT / "reports/transfer_v0.2/figures/frozen_transfer_signal.png")))
else:
    display(Markdown("Transferência ainda pendente; nenhum resultado é inferido."))
''')]
        if saved_evidence.get('supplemental'):
            main_cells += [md('''## Diagnósticos exploratórios após a análise principal

Estes diagnósticos foram definidos depois da inspeção dos resultados. Mantêm
modelos, calibradores, limiares e critérios principais congelados. Os intervalos
adicionais são pontuais, sem correção para comparações múltiplas, condicionais
aos escores fixos. Folds compartilham dados de treino; não são replicações
independentes. A exclusão de uma tarefa não refaz o treinamento.
'''), code('''extra = evidence["supplemental"]
SUPPLEMENT = REPORT / "supplemental"
display(Markdown("**Reprodução dos contrastes principais:**"))
display(extra["primary_reproduction"])
display(pd.DataFrame(extra["influence_summary"]))
display(pd.json_normalize([r for r in extra["fold_contrasts"]
                          if r["comparison"] == "HB minus B"]))
display(Image(filename=str(SUPPLEMENT / "figures/fold_stability.png")))
'''), md('''### Mesmas tentativas e sensibilidade ao histórico

A coorte comum conserva as identidades presentes em 8, 16, 32 e 64 tokens,
condicionando em T > 64. Ela não identifica um efeito causal do tempo.
Excluir tarefas históricas só modifica a avaliação; não remove a exposição
anterior da pesquisa nem cria uma nova amostra confirmatória.
'''), code('''display(pd.json_normalize([r for r in extra["contrasts"]
    if r["checkpoint"] == "32" and r["comparison"] == "HB minus B"]))
display(extra["historical_exposure_sensitivity"])
display(Image(filename=str(SUPPLEMENT / "figures/common_cohort.png")))
'''), md('''### Calibração e transporte do alvo de retenção

ECE usa dez bins fixos da probabilidade da classe positiva com peso igual por
tarefa. Depende de discretização e tamanho amostral; os percentis do bootstrap
podem não conter o ECE empírico. Interprete junto a log loss e Brier.
A origem usa escores OOF; o alvo usa ajuste em toda a origem. Diferenças entre
bancos não isolam causalmente uma mudança de domínio.

Retenção usa todos os corretos ativos no gate, inclusive candidatos nunca
iniciados no replay. Cada política pode usar um checkpoint diferente.
O alvo 0,97 é empírico no desenvolvimento, sem garantia externa.
'''), code('''display(pd.DataFrame([{k: v for k, v in row.items() if k != "reliability"}
                      for row in extra["calibration"]]))
display(Image(filename=str(SUPPLEMENT / "figures/calibration_transfer.png")))
display(pd.DataFrame([dict(row, dataset=dataset)
    for dataset, item in extra["correct_retention"].items() for row in item["aggregate"]]))
display(pd.DataFrame([dict(row, dataset=dataset)
    for dataset, item in extra["correct_retention"].items() for row in item["by_fold"]]))
display(Image(filename=str(SUPPLEMENT / "figures/correct_retention.png")))
'''), md('''### Critérios necessários e limites da conclusão

As decisões abaixo aplicam as margens originais. Ausência de suporte ao ganho
incremental não demonstra ausência universal de sinal. Os limites dos intervalos
descrevem a precisão alcançada sob as suposições e os modelos ajustados.
Não se calcula poder observado nem se define margem de equivalência após o resultado.
'''), code('''display(pd.DataFrame(extra["decision_ledger"]))
display(Markdown((SUPPLEMENT / "README.md").read_text(encoding="utf-8")))
''')]
        if saved_evidence.get('excellence'):
            main_cells += [md('''## Aprofundamento científico executado em 17/10/2026

As famílias abaixo foram definidas depois de conhecer os resultados originais.
Todas usam as mesmas tarefas externas originais. A seleção, o PCA e a calibração
respeitam as partições internas; o residual exclui os rótulos pontuados também
na seleção e calibração do primeiro estágio. Os intervalos adicionais são
pontuais e condicionais, sem confirmação independente ou cobertura simultânea.
'''), code('''exc = evidence["excellence"]
EXCELLENCE = ROOT / "reports/excellence"
display(pd.DataFrame(exc["models"]))
display(pd.DataFrame(exc["recovery_vs_original_HB"]))
display(Image(filename=str(EXCELLENCE / "figures/model_ablation.png")))
display(Markdown("PCA e residual reduzem a desvantagem original, sem demonstrar ganho sobre B. "
                 "A regularização conjunta mais forte não resolveu a desvantagem na grade testada."))
'''), code('''display(pd.DataFrame(exc["geometry"]["training_spectra"]))
display(pd.DataFrame(exc["geometry"]["same_layer_direction_cosines"]))
display(pd.DataFrame(exc["geometry"]["learning_curves"]))
display(pd.DataFrame(exc["prefix_diagnostics"]["summaries"]))
display(Image(filename=str(EXCELLENCE / "geometry/geometry_learning.png")))
display(Markdown("PCA foi ajustado no treino externo. A curva de tarefas usa algoritmos fixos; "
                 "não estima a curva do procedimento original selecionado nem identifica causalidade."))
'''), code('''display(pd.DataFrame(exc["transfer"]["false_abort_concentration"]))
display(pd.DataFrame(exc["transfer"]["target_coverage"]))
display(Image(filename=str(EXCELLENCE / "transfer/retention_curves.png")))
display(Image(filename=str(EXCELLENCE / "transfer/conditional_scores.png")))
display(Markdown("Origem OOF, desenvolvimento da origem e alvo congelado são avaliações distintas. "
                 "As curvas no alvo não escolhem um novo limiar."))
'''), code('''oracle = [r for r in exc["economics"]["summaries"] if r["policy"] == "oracle" and r["budget"] == 20]
display(pd.DataFrame(oracle))
display(pd.DataFrame(exc["oracle_concentration"]))
display(Image(filename=str(EXCELLENCE / "economics/oracle_frontier.png")))
display(Markdown("O oracle conhece os rótulos e preserva todos os corretos; não é uma política online "
                 "nem um limite inferior universal para políticas que podem descartar sucessos."))
display(exc["hardware"])
display(pd.DataFrame(json.loads((EXCELLENCE / "resource_budget.json").read_text())["scenarios"]))
display(Markdown("A avaliação independente aprovada está concluída em seção própria."
                 if exc.get("independent") else
                 "A extensão independente ainda não tem resultados completos nesta publicação."))
''')]
            if saved_evidence['excellence'].get('independent'):
                main_cells += [md('''## Avaliação independente aprovada

O protocolo foi aprovado e congelado antes da primeira trajetória nova. A revisão
do orientador foi informada pelo estudante. São 20 tarefas técnicas disjuntas
e 120 principais, com 20 tentativas cada; somente as principais entram na inferência.
O endpoint único compara retenções de corretos ativos no mesmo t=8, com gates
congelados na origem. As demais métricas são descritivas. Não há ajuste no alvo.
'''), code('''independent = exc["independent"]
display(independent["primary"])
display(pd.DataFrame(independent["recall"]))
display(pd.DataFrame(independent["audit"]))
display(Image(filename=str(EXCELLENCE / "independent_v1/independent_retention.png")))
display(Markdown("O suporte exige limite superior do IC bilateral de 95% abaixo de zero. "
                 "A decisão nova não modifica Q2 ou H3 originais; um intervalo que inclua zero não demonstra equivalência."))
'''), code('''display(pd.DataFrame(independent["secondary_predictive_metrics"]))
display(pd.DataFrame(independent["policy_summaries"]))
display(pd.DataFrame(independent["secondary_HB_vs_B_effects"]))
display(Image(filename=str(EXCELLENCE / "independent_v1/independent_secondary.png")))
display(Markdown((EXCELLENCE / "independent_v1/README.md").read_text(encoding="utf-8")))
''')]
        cells[2:2] = main_cells
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
        'display_name': 'SLM Research (.venv)', 'language': 'python', 'name': 'slm-research'},
        'language_info': {'name': 'python', 'version': '3.12'}})
    path = ROOT / 'notebooks/final_research_summary.ipynb'
    archive = ROOT / 'notebooks/legacy/final_research_summary_v0.1.ipynb'
    if path.exists() and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)
    if args.execute:
        NotebookClient(notebook, timeout=180, kernel_name='slm-research', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbf.write(notebook, path)
    if args.execute:
        exporter = HTMLExporter()
        exporter.embed_images = True
        html, _ = exporter.from_notebook_node(notebook)
        (ROOT / 'reports/final_review/notebook.html').write_text(html, encoding='utf-8')
    print(f'Notebook {"executed" if args.execute else "created"}: {path}')


if __name__ == '__main__':
    main()
