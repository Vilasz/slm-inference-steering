# Guia de execução, análise e escrita do TCC

Revisão em 2026-10-03, baseada no manuscrito Typst enviado, no protocolo v0.1,
no código atual e nos artefatos locais. Este guia é uma revisão metodológica;
não representa aprovação do protocolo nem resultados confirmatórios.

## 1. Decisão principal

Mantenha Q1–Q3 como núcleo: previsibilidade temporal, discriminação entre
trajetórias da mesma tarefa e utilidade operacional. O próximo passo é conciliar
o apêndice de emendas do manuscrito com o protocolo e a implementação. Só depois
congele o desenho e execute a análise confirmatória.

O texto já distingue dificuldade da tarefa de informação sobre a trajetória,
separa previsão de utilidade e admite resultados inconclusivos. Preserve essas
distinções. A prioridade é produzir evidência reproduzível para elas.

O estado auditado hoje é:

| Item | Estado observado |
|---|---|
| Piloto | 39 de 960 trajetórias; 13 tarefas representadas |
| Temperatura selecionada | Não; o piloto está incompleto |
| Banco definitivo | Ainda não gerado |
| Q1, Q2 e Q3 confirmatórias | Ainda não executadas |
| GPU | RTX 3050 Laptop de 6 GB |
| Pico reservado medido anteriormente | 4.598 MiB, aproximadamente 4,49 GiB |
| Precisão validada na geração | Pesos BF16, projeções lineares FP32, eager, batch 1 |
| Tempo extrapolado anteriormente | Cerca de 3 h para piloto e 8,1 h para banco definitivo; estimativas provisórias |

A auditoria desta revisão está em
[`tcc_review_pilot_audit.json`](../reports/research/tcc_review_pilot_audit.json).
Tempos excluem carregamento, análise estatística e MBPP+ e precisam ser
reestimados após um piloto completo.

## 2. Ajustes anteriores ao congelamento

| Ponto | Implementação v0.1 | Decisão necessária para seguir o manuscrito |
|---|---|---|
| Corretude e truncamento | `bank.py` força `passed=False` quando há truncamento, mesmo após executar o verificador | Usar aprovação funcional como alvo principal; guardar truncamento separadamente. Nomear “passa e não truncou” como endpoint distinto, se mantido |
| Término no checkpoint | A política usa `generated_tokens >= t`; geração chama aborto antes de verificar EOS | Verificar término antes de aborto; para continuação usar `T > t`. Separar existência do vetor de elegibilidade operacional |
| Permutação | `reporting.py` usa permutação do delta H+B−B no pool OOF como requisito de H2 | Testar associação intratarefa por fold externo e combinar conservadoramente, conforme o manuscrito; avaliar ganho sobre B por diferenças externas pareadas e perdas |
| Seleção de política | `policy.py` testa apenas o maior limiar admissível por checkpoint | Avaliar todos os limiares da grade admissível pelo custo por tarefa; incluir ausência de aborto; fixar regra de empate |
| Calibração e desenvolvimento | OOF da configuração escolhida também participou da seleção dessa configuração; calibrador e limiar usam os mesmos dados de desenvolvimento | Adotar cross-fitting que inclua seleção interna ou declarar o otimismo de desenvolvimento; manter toda avaliação externa separada. Não chamar recall de ajuste de garantia |
| Margem entre logits | `generation.py` calcula depois de top-p e registra ausência se só resta um token | O manuscrito define margem antes de top-p, depois da temperatura. Escolher e congelar uma definição; harmonizar geração, features e política online |
| Intercepto do probe | `probes.py` usa `solver='liblinear'`, que regulariza o intercepto sintético | O manuscrito declara intercepto sem penalidade. Alinhar o solver ou a equação; registrar normalização dos pesos e interpretação de `C` |
| Robustez | Ainda não há um fluxo confirmatório completo de transferência MBPP+ | Implementar geração e aplicação do sistema congelado; distinguir transferência de retreino/recalibração local |

A documentação oficial do
[scikit-learn 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)
explica a penalização do intercepto pelo `liblinear`.

O critério antes subjetivo de aumento material de inválidos/truncamentos já foi
quantificado na configuração atual: até **5 pontos percentuais** acima da menor
taxa entre temperaturas. Atualize esse trecho do manuscrito ou registre uma nova
decisão prospectiva. Não altere o critério para favorecer resultados do piloto.

Crie uma versão sucessora do protocolo, com data, justificativas e resultados
anteriores conhecidos. Atualize configuração, validações de versão, gates,
manifestos e testes conjuntamente. Os scripts atuais têm convenções e verificações
específicas da v0.1: trocar apenas o nome do YAML não faz uma migração.

Preserve o banco original. Caso reutilize trajetórias na nova versão, gere um
artefato derivado com proveniência e regra de migração explícitas. O resultado
base/plus do verificador está preservado nos registros e pode apoiar uma
reclassificação funcional auditável. A margem anterior ao top-p pode exigir
nova extração; ela não é recuperável em geral apenas da margem filtrada salva.
Não misture silenciosamente regimes de features ou rótulos.

Antes de novas inferências confirmatórias, os testes devem cobrir: término
exatamente em `t`, aborto antes do término, ausência de aborto, contabilidade
pareada, rótulo funcional de um programa truncado válido e isolamento de labels
externos. Alterar labels de teste deve deixar modelo, calibrador e limiar daquele
fold idênticos. Use dados sintéticos nesses testes.

### Histórico de exposição

O inventário dos JSONL antigos encontrou **30 tarefas HumanEval**, com **4** no
piloto atual e **26** entre as 132 chamadas de reservadas. Consulte
[`legacy_task_presence.json`](../reports/research/legacy_task_presence.json).
O inventário verifica presença de IDs, sem recalcular desempenho; comprova
geração anterior, não quais saídas ou métricas foram inspecionadas.

Registre quais análises antigas influenciaram escolhas. Separação por tarefa
evita vazamento no ajuste do modelo, mas não demonstra ausência de adaptação do
pesquisador ao benchmark. Descreva o congelamento como prospectivo em relação
às novas análises, com divulgação do histórico. Uma redefinição da amostra deve
ser datada, justificada e acordada antes da análise, sem excluir tarefas por
seus resultados. Faça inventário equivalente antes de considerar MBPP+ intocado.

## 3. O que executar agora

No PowerShell, a partir da raiz do repositório:

```powershell
$pythonPesquisa = ".\.venv\Scripts\python.exe"

# Diagnóstico do ambiente e validações existentes.
& $pythonPesquisa scripts/diagnose_cuda.py
& $pythonPesquisa -m pytest -q

# Mostrar o planejamento atual sem carregar o modelo.
& $pythonPesquisa scripts/build_candidate_bank.py --phase pilot --dry-run

# Verificar o banco existente, sem gerar novas trajetórias.
& $pythonPesquisa scripts/audit_candidate_bank.py --bank-dir runs/candidate_bank/pilot_v0.1 --output reports/research/pilot_audit.json
```

Leia `passed`, `complete`, `attempts`, `expected_attempts` e
`confirmatory_ready` na auditoria. Hoje ela passa para a integridade do banco,
mas informa `complete=false` e `confirmatory_ready=false`. Isso é compatível
com um piloto parcial. Testes aprovados não garantem concordância com o novo
manuscrito: as diferenças da seção 2 precisam de verificações próprias.

Se optar por completar primeiro o **piloto técnico v0.1 sem alterar sua
configuração**, o comando existente é:

```powershell
& $pythonPesquisa scripts/build_candidate_bank.py --phase pilot --output-dir runs/candidate_bank/pilot_v0.1 --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --time-budget-seconds 600
```

Repita para retomar. O limite pausa entre tentativas e pode ser ultrapassado
pelo término da tentativa em andamento. Ele não reduz as 960 trajetórias
planejadas. Não execute duas instâncias escrevendo nesse banco.

Minha recomendação é resolver antes as definições da seção 2, evitando gerar
features que precisem ser reextraídas. Não modifique protocolo ou configuração
e tente retomar o mesmo banco: a checagem de hashes recusará a mistura.

Durante o piloto, observe completude, tarefas mistas por temperatura, taxa
funcional, inválidos/truncamentos, comprimentos, tempo e VRAM. Escolha temperatura
pela regra congelada, sem consultar AUC de probes. Os 39 registros atuais ainda
não permitem a seleção planejada.

## 4. Sequência posterior às correções

Esta sequência depende de nova implementação, piloto válido e aprovação/
congelamento. **Não execute os blocos seguintes como análise definitiva agora.**
A v0.2 abaixo é um nome sugerido para arquivos futuros; eles ainda não existem.

1. Executar e auditar o piloto da versão revisada; selecionar o regime pela
   regra predefinida. Preparar o registro de congelamento com
   `prepare_sampling_freeze.py`, atualizado para a nova versão. Registrar a
   aprovação real e o commit/tag exigidos pelo protocolo.
2. Gerar 20 tentativas completas por tarefa confirmatória, inclusive depois de
   um sucesso. Isso constrói o banco comum usado no replay. Com a amostra atual,
   são 132 × 20 = 2.640 trajetórias.
3. Auditar completude, hashes, seeds, verificações e ativações; congelar o banco.
4. Produzir predições externas com cinco folds agrupados por tarefa, seleção
   interna e calibração. Treinar B, H e H+B e executar os controles.
5. Aplicar as políticas B e H+B escolhidas exclusivamente no desenvolvimento.
   Validar online reprodução de prefixos, abortos, resoluções e custos.
6. Gerar tabelas, intervalos e figuras com o procedimento estatístico revisado.
7. Executar a transferência MBPP+ com todos os ajustes congelados e escopo
   definido antes de inspecionar seu resultado.

Modelo de comandos **para depois da migração**, com os entrypoints existentes:

```powershell
$configPesquisa = ".\configs\protocol_v0.2.yaml"
$bancoPesquisa = ".\runs\candidate_bank\definitive_v0.2"
$probesPesquisa = ".\runs\probes\protocol_v0.2"

& $pythonPesquisa scripts/build_candidate_bank.py --config $configPesquisa --phase definitive --pilot-manifest runs/candidate_bank/pilot_v0.2/manifest.json --output-dir $bancoPesquisa --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --time-budget-seconds 600

& $pythonPesquisa scripts/audit_candidate_bank.py --bank-dir $bancoPesquisa --output reports/research/definitive_v0.2_audit.json

& $pythonPesquisa scripts/run_grouped_probes.py --config $configPesquisa --bank-dir $bancoPesquisa --output-dir $probesPesquisa

& $pythonPesquisa scripts/validate_online_replay.py --config $configPesquisa --bank-dir $bancoPesquisa --probes-dir $probesPesquisa --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --max-tasks 1

& $pythonPesquisa scripts/generate_tcc_artifacts.py --config $configPesquisa --probes-dir $probesPesquisa --output-dir reports/protocol_v0.2
```

O exemplo de uma tarefa online verifica funcionamento, mas não basta para
estimar ganho de latência. Defina prospectivamente um conjunto que cubra as
decisões relevantes e meça tempos pareados incluindo captura, transferência,
score, prefill e verificação. Ausência de abortos na amostra torna essa parte
da validação pouco informativa. MBPP+ requer um fluxo adicional ainda pendente;
os antigos scripts MBPP não implementam automaticamente essa transferência.

### Orçamento da máquina

Mantenha um modelo de 1,5B, batch 1 e a precisão que passou nos testes causais.
Depois da geração, probes, bootstraps, permutações por fold e replay trabalham
sobre vetores e registros salvos, sem precisar manter o modelo na GPU.

Priorize completar HumanEval+ antes de ampliar modelos, steering ou análises
geométricas. Para MBPP+, congele um subset por hash e seu tamanho com base no
tempo disponível. Se o orçamento não permitir a extensão, reporte-a como
limitação; o protocolo atual a classifica como secundária. Não mude precisão,
temperatura, limite de tokens ou amostra durante o banco para melhorar métricas.

## 5. O que olhar nos resultados

| Pergunta | Artefatos previstos | Leitura principal |
|---|---|---|
| Qualidade da amostra | `manifest.json`, `tables/task_composition.csv`, `tables/risk_sets.csv` | Quantas tarefas, trajetórias ativas, tarefas mistas e sucessos sustentam cada estimativa? |
| Q1: sinal temporal | `temporal_prediction.png`, `within_task_discrimination.png`, `tables/predictive_performance.csv` | Comparar AUC agregada e WT-AUC, acompanhadas de ICs. Mudança de checkpoint também muda a população elegível |
| Q2: ganho incremental | `results.json`, predições externas, teste por fold revisado | Em 32 tokens, examinar ΔWT-AUC(H+B−B), seu IC pareado e Δlog loss. O p-valor de associação responde a outra hipótese |
| Confiabilidade dos scores | `calibration_32.png`, log loss, Brier e AP | Ranking e calibração são propriedades diferentes. AUC boa não garante retenção de sucessos pelo limiar |
| Q3: utilidade | `selection_audit.json`, `policy_replay.json`, `tables/adaptive_policy.csv`, `compute_resolution.png` | Comparar resolução e custo pareados contra parada por verificador e política B |
| Validação operacional | `online_validation.json` e medições de tempo | A política interrompe realmente antes de gerar o futuro? Há economia após incluir seu overhead? |

As figuras ficam no subdiretório `figures/` da saída final. Os nomes acima
identificam os artefatos previstos pelos scripts atuais; a revisão pode acrescentar
colunas e diagnósticos.

Para Q2, preserve o endpoint de 32 tokens. Segundo o manuscrito, evidência
incremental positiva exige ΔWT-AUC positivo com IC de 95% inteiramente acima
de zero e Δlog loss negativo. Reporte também o IC da perda; se incluir zero,
qualifique a evidência probabilística. H+B superior ao acaso não basta para
demonstrar superioridade sobre B.

Para Q3, use N=20 como principal e N=1/5/10 como secundários. Com Δresolução
definido como política menos comparador, a condição planejada é limite inferior
do IC de 95% **maior que −0,03**. Para Δtokens, o limite superior deve ser
**menor que zero**. Compare com ambos os comparadores. Uma meta de 97% de recall
no desenvolvimento não garante 97% no teste nem preservação da resolução.

| Padrão observado | Interpretação defensável |
|---|---|
| AUC agregada alta, WT-AUC perto de 0,5 | O preditor pode reconhecer dificuldade entre tarefas; falta evidência de discriminação entre tentativas |
| WT-AUC acima de 0,5, H+B sem ganho sobre B | Há sinal intratarefa, mas não ganho incremental demonstrado para as classes comparadas |
| Ganho incremental, sem economia sob a margem | Há valor preditivo; utilidade operacional ainda não demonstrada |
| Menos tokens, com perda de resolução além da margem | O trade-off não satisfaz o critério principal |
| ICs amplos incluindo efeitos úteis e nulos | Resultado inconclusivo no orçamento estudado |
| Ganho principal sem transferência MBPP+ | A evidência tem alcance restrito ao regime avaliado |

Reamostre tarefas inteiras, usando os mesmos índices para comparar métodos.
Milhares de tentativas ou pares não equivalem a milhares de unidades
independentes. Os ICs sobre scores OOF fixos são aproximados e condicionais aos
ajustes; não cobrem automaticamente toda a variabilidade de retreinamento.

### Como usar a comparação atualmente aberta no IDE

`runs/phase1_pilot/comparison.md` compara cinco tarefas: 1.497 tokens para gerar
todos os candidatos e 942 na parada após sucesso do verificador. Porém, a linha
**“Tokens ate sucesso/orcamento” é 942 nos dois métodos**. A redução bruta de
aproximadamente 37% mostra o efeito de dispensar tentativas posteriores a um
sucesso. Não evidencia economia por identificar e abortar uma trajetória
incorreta ainda incompleta. Use a tabela como motivação exploratória, com sua
amostra e verificador originais declarados. Atribua ganho do probe apenas à
comparação adicional contra parada por verificador e contra B.

## 6. Como reorganizar e melhorar o documento

Uma estrutura de sete capítulos torna o encadeamento mais fácil de acompanhar:

1. **Introdução:** problema, pergunta, Q1–Q3, objetivos, contribuição pretendida
   e escopo. Atualize o estado do repositório e retire do corpo a narrativa sobre
   “documento substituto”; ela pode ficar no histórico de versões.
2. **Trabalhos relacionados:** organize por tarefa versus trajetória, dados
   usados pelo preditor e operação de interrupção. Inclua uma tabela com modelo,
   benchmark, instante, split, controles, alvo e garantia. Distingua artigos,
   preprints e teses.
3. **Fundamentação:** concentre a formulação nos tokens, estado causal, corretude
   pelo verificador, WT-AUC, comparação incremental e custo até solução. Mova
   derivações que não sustentam uma decisão metodológica para apêndices.
4. **Metodologia:** versão congelada, hardware/precisão, amostra, extração,
   rótulos, CV agrupada, calibração, testes e seleção da política. Inclua uma
   tabela única de parâmetros e um diagrama dos splits. Justifique 32 tokens,
   a margem de 3 pp e a meta empírica de 97% como escolhas de desenho.
5. **Resultados:** comece pela execução efetiva e composição da amostra; depois
   Q1, Q2, Q3 e robustez. Toda afirmação quantitativa deve apontar para tabela,
   figura ou artefato gerado. Apresente efeitos e ICs, além de p-valores.
6. **Discussão:** responda o que cada resultado permite concluir e explique
   limitações: poucos problemas mistos, seleção de sobreviventes, dependência
   entre folds, histórico de exposição, possível contaminação do benchmark no
   treinamento e alcance da suíte de testes. Predição não demonstra mecanismo
   causal; economia de tokens não demonstra economia de tempo.
7. **Conclusão:** responda Q1–Q3 com os números efetivos e o alcance do estudo.
   Escreva resumo e abstract finais depois disso, retirando placeholders.

Mantenha nos apêndices o protocolo aprovado, emendas, invariantes, comandos,
metadados completos e resultados secundários. Evite repetir no corpo e em
diversos apêndices as mesmas ressalvas. Preserve uma seção clara de limitações.

Pacote mínimo para o capítulo de resultados: tabela do ambiente e amostra;
tabela dos conjuntos sob risco; figura temporal de AUC e WT-AUC; tabela da
comparação principal H+B−B em 32 tokens; figura de calibração; tabela de
resolução/custo/false-abort com ICs; figura custo–resolução; medições online.
Acrescente MBPP+ se executado. Informe recall entre corretos ativos e global,
com seus denominadores.

Modelo de redação de Q2, para preencher apenas com resultados efetivos:

> Em 32 tokens, entre [m] tarefas mistas e [n] trajetórias ativas, a diferença
> WT-AUC(H+B−B) foi [d], com IC de 95% [a; b]. A diferença de log loss foi [l],
> com IC [c; e]. Os resultados [sustentam/não sustentam/são inconclusivos para]
> o ganho incremental nas classes de preditores e no regime avaliados.

Não conclua ausência de informação universal porque um probe linear falhou,
nem reivindique novidade apenas pelo uso de hidden states em Best-of-N.

### Revisão bibliográfica e edição

Conferi páginas primárias e resumos dos trabalhos recentes citados. Fixe as
versões consultadas e leia os métodos completos antes de preencher a tabela
comparativa:

- [Di Cicco, arXiv:2606.14530v3](https://arxiv.org/abs/2606.14530v3): estado final
  do prompt e corretude da primeira tentativa; não é a mesma unidade de
  comparação intratarefa proposta aqui.
- [Klassert, arXiv:2609.13975v1](https://arxiv.org/abs/2609.13975v1): tese sobre
  estimação introspectiva em código; o resumo compara features estáticas e
  dinâmicas. Não trate o resumo como auditoria completa dos splits.
- [Ruan et al., arXiv:2607.06503](https://arxiv.org/abs/2607.06503): cascata de
  abortos em episódios de agentes com controle/certificação de retenção;
  distinto da meta empírica deste desenho.
- [Zhu et al., arXiv:2509.12886](https://arxiv.org/abs/2509.12886): dificuldade
  percebida a partir de representações iniciais e inferência adaptativa.
- [Wang et al., ST-BoN, NeurIPS 2025](https://openreview.net/pdf?id=BcKYVmh3yH):
  seleção e truncamento com consistência de estados internos entre amostras.

O anexo faz referência a `referencias.bib`, mas esse arquivo não foi fornecido
nem encontrado no repositório nesta revisão. A conferência das chaves, entradas
bibliográficas e compilação integral ainda está pendente. Fixe autor, título,
ano, versão, veículo e DOI/URL de cada fonte; não atribua revisão por pares
apenas por estar no arXiv.

Na preparação final, atualize data e estado da capa, confira o formato exigido
pelo curso, chamadas de figuras/tabelas/equações e tradução do abstract. Troque
“o repositório não foi auditado” por um relato datado do que foi verificado e das
divergências restantes. O estado final da geração é referência tardia, sem
garantia de ser limite superior do desempenho precoce.

## 7. Organização dos arquivos

Organização sugerida, sem mover ou apagar os artefatos atuais:

```text
tcc/                         # criar ao incorporar o fonte completo
  main.typ
  referencias.bib
  chapters/
  figures/                   # exportadas da análise, sem edição de valores
configs/                     # versões congeladas e registro de sampling
docs/TCC_GUIDELINE.md         # roteiro de execução e escrita
docs/legacy/                 # documentação exploratória anterior
runs/candidate_bank/         # bancos e ativações por versão
runs/probes/                 # predições, ajustes e decisões por versão
reports/research/            # ambiente, auditorias, histórico e execução
reports/protocol_vX/          # tabelas, resultados e figuras finais
RESULTS.md                   # estado real da evidência
```

Guarde cada tabela e figura junto do comando, versão do código e hashes dos
insumos que a produziram. Preserve as experiências antigas como exploratórias.
Uma conclusão negativa ou inconclusiva, com comparação adequada e limitações
claras, continua respondendo à pergunta do TCC.
