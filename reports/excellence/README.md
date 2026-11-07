# Aprofundamento científico executado

Os diagnósticos abaixo utilizam dados já examinados e são exploratórios. A avaliação independente tem seção própria. Resultados e protocolo v0.2 foram preservados.

## Por que H+B perdeu para B?

| Família | WT-AUC | Log loss | ΔWT-AUC contra B [IC95%] | Δlog loss contra B [IC95%] |
|---|---:|---:|---|---|
| B_original | 0.62816 | 0.63022 | 0.00000 [0.00000; 0.00000] | 0.00000 [0.00000; 0.00000] |
| B_reproduced | 0.62816 | 0.63022 | 0.00000 [0.00000; 0.00000] | -0.00000 [-0.00000; 0.00000] |
| HB_original | 0.58113 | 0.66807 | -0.04703 [-0.09937; 0.00558] | 0.03785 [0.00666; 0.07062] |
| HB_pca16 | 0.62398 | 0.63338 | -0.00417 [-0.03762; 0.03187] | 0.00316 [-0.01024; 0.01668] |
| HB_pca64 | 0.62975 | 0.63180 | 0.00160 [-0.03014; 0.03377] | 0.00158 [-0.01207; 0.01567] |
| HB_stronger | 0.54758 | 0.67638 | -0.08058 [-0.14435; -0.01534] | 0.04616 [0.01328; 0.08092] |
| H_original | 0.56811 | 0.68687 | -0.06005 [-0.11433; -0.00395] | 0.05665 [0.02413; 0.09017] |
| H_residual_B | 0.63453 | 0.62895 | 0.00637 [-0.00874; 0.02438] | -0.00127 [-0.00575; 0.00316] |
| confidence_only | 0.62418 | 0.63156 | -0.00397 [-0.00821; -0.00002] | 0.00134 [-0.01671; 0.01931] |
| entropy_pair | 0.60356 | 0.63067 | -0.02459 [-0.04605; -0.00562] | 0.00045 [-0.01954; 0.02000] |
| last_nll | 0.50231 | 0.69971 | -0.12585 [-0.17721; -0.07427] | 0.06949 [0.03471; 0.10351] |
| logit_margin | 0.63108 | 0.65635 | 0.00293 [-0.02309; 0.03221] | 0.02614 [0.00195; 0.04852] |
| max_entropy | 0.51463 | 0.65618 | -0.11353 [-0.16127; -0.06826] | 0.02596 [-0.00395; 0.06014] |
| mean_entropy | 0.61281 | 0.63842 | -0.01535 [-0.03517; 0.00266] | 0.00820 [-0.01047; 0.02566] |
| mean_nll | 0.63777 | 0.65641 | 0.00961 [-0.01775; 0.03762] | 0.02619 [0.00112; 0.05014] |
| nll_pair | 0.64037 | 0.65649 | 0.01222 [-0.01566; 0.04086] | 0.02627 [0.00088; 0.05046] |
| prompt_only | 0.50000 | 0.67415 | -0.12816 [-0.17453; -0.08334] | 0.04393 [0.01850; 0.06870] |

Média de NLL e margem preservam grande parte do ranking na estimativa pontual, mas têm perda maior que B.
PCA e a correção residual reduzem a desvantagem de H+B; seus intervalos contra B continuam incluindo zero.
Penalização conjunta ainda mais forte, isoladamente, não resolveu a desvantagem nesta grade.
Isso sustenta sensibilidade à parametrização, sem identificar dimensionalidade como causa única ou provar redundância completa.

PCA é ajustado apenas nos treinos de cada split; a correção residual usa offsets de B cuja seleção e calibração excluem o rótulo pontuado.
O residual tem ligação logística nativa, sem um segundo Platt. C, camada e todos os pré-processamentos são escolhidos dentro do treino externo.
As 70 unidades foram executadas duas vezes após uma limpeza de nomes de variáveis; o erro máximo das probabilidades foi 0.

Prefixos compartilhados foram verificados em 8/16/32/64 tokens. Em 32, nenhuma das 92 tarefas mistas principais tem um único prefixo;
portanto o colapso integral dos prefixos por tarefa não explica o contraste principal. Em 8 tokens, são 15/100 mistas na origem e 3/24 no alvo.
Scores de prompt/prefixo idênticos conferem dentro da tolerância numérica; contagens e colisões parciais estão em `prefixes/`.

## Transporte do gate

O gate H+B da origem em t=8 descartou 90/366 corretos potenciais em MBPP+, distribuídos por 12 tarefas.
Cinco tarefas concentram 60 desses descartes. As tarefas Mbpp/166 e Mbpp/259 perderam todos os seus sucessos.
Entre corretos ativos efetivamente iniciados, 23/48 foram abortados; esse denominador difere dos 366 corretos potenciais.
O ranking intratarefa H+B em t=8 no alvo é aproximadamente 0,494. Curvas mostram a distribuição dos corretos em torno do limiar congelado.
A comparação inclui origem OOF, desenvolvimento da origem usado na seleção e modelo de origem completo aplicado ao alvo.
As diferenças não isolam causalmente domain shift. Nenhum limiar foi ajustado com os rótulos MBPP+.

## Limite econômico retrospectivo

| Banco | t | Economia oracle contra verificador [IC95%] | Resolução |
|---|---:|---|---:|
| HumanEval+ | 8 | 0.77748 [0.70528; 0.82339] | 0.84848 |
| HumanEval+ | 16 | 0.69704 [0.61702; 0.75009] | 0.84848 |
| HumanEval+ | 32 | 0.54476 [0.44614; 0.61655] | 0.84848 |
| HumanEval+ | 64 | 0.34107 [0.23431; 0.42797] | 0.84848 |
| MBPP+ | 8 | 0.76253 [0.57359; 0.83565] | 0.84375 |
| MBPP+ | 16 | 0.64999 [0.45991; 0.73917] | 0.84375 |
| MBPP+ | 32 | 0.44256 [0.23889; 0.56248] | 0.84375 |
| MBPP+ | 64 | 0.18756 [0.05679; 0.30212] | 0.84375 |

O oracle conhece rótulos finais e preserva todos os corretos. É limite de custo somente na classe com essa restrição e checkpoint fixo.
Políticas que perdem sucessos podem custar menos em certos caminhos; por isso a fronteira sempre acompanha resolução.

| Banco, t=8, N=20 | Parcela da economia oracle nas tarefas não resolvidas |
|---|---:|
| HumanEval+ | 0.7144 |
| MBPP+ | 0.8612 |

A derivação econômica condiciona na tarefa, admite dependência entre custo e sucesso da mesma tentativa e exige independência entre tentativas.
No limite ilimitado, a condição é E[D]/E[T] < r para p>0 e r>0. Para N finito, custo e cobertura devem ser comparados conjuntamente.
Retenção global inclui os sucessos anteriores ao gate; não é sempre igual à retenção ativa.

## Recursos, validação e continuidade

Benchmark técnico: 50 pares / 100 gerações em 26 tarefas conhecidas, tokens e estados exatamente iguais entre implementações.
GPU detectada: RTX 3050 Laptop 6 GB. A RTX 4060 mencionada na missão não está disponível neste ambiente.
Tempos descritivos e componentes separados estão em `hardware/`; não se reivindica aceleração da política a partir de tokens.

Artefatos: [auditoria](scientific_audit.md), [reprodução](reproduction.json), [orçamento](resource_budget.md),
[progresso](PROGRESS.md), [protocolo proposto](../../docs/protocols/EXCELLENCE_EXTENSION_v1.md).

## Reprodução estatística sem geração na GPU

```powershell
.venv/Scripts/python.exe scripts/run_final_scientific_audit.py
.venv/Scripts/python.exe scripts/run_excellence.py report
```

[Guia completo de reprodução](../../docs/FINAL_REPRODUCIBILITY.md), com reexecução estatística em pastas separadas e publicação.
A busca exploratória já foi executada duas vezes e tem 70 unidades arquivadas; a auditoria compara as probabilidades sem reabrir uma seleção de vencedores.

### Medições na GPU, separadas da análise

```powershell
.venv/Scripts/python.exe scripts/run_operational_benchmark.py
```

Esse comando usa tarefas antigas, exige a coleta independente encerrada e conserva resultados concluídos com o mesmo fonte.
A coleta científica tem executor próprio com aprovação vinculada a hashes e não é acionada pelos comandos estatísticos.

## Latência operacional descritiva

72 execuções; seis tarefas conhecidas; três repetições; N=3; quatro modos.
[Plano, dados e componentes de tempo](operational_v1/README.md). Não se reivindica aceleração robusta.

## Avaliação independente aprovada e concluída

20 tarefas técnicas disjuntas e 120 principais, 20 tentativas por tarefa; gates de origem mantidos em t=8.
Retenção H+B−B: -0.16359060402684564; IC95% [-0.2402298061170319, -0.0942741120201171]; suporte pelo critério aprovado: True.
Aprovação explícita do estudante; revisão do orientador informada por ele. Snapshot e hashes precedem a primeira trajetória.
Único endpoint confirmatório novo; não modifica Q2/H3 originais. Resultados secundários permanecem descritivos.

[Relatório, auditoria e dados completos](independent_v1/README.md).
