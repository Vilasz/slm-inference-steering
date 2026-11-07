#set page(width: 28cm, height: 15.75cm, margin: 1.1cm, numbering: "1")
#set text(font: "Arial", size: 17pt, lang: "pt")
#set par(leading: 0.8em)
#set heading(numbering: none)
#show heading: set text(size: 27pt, fill: rgb("234b68"))
= Sinais precoces de corretude

*João Felipe Vilas Boas · FGV EMAP*

Prever a trajetória, acrescentar informação e economizar inferência.

Qwen2.5-Coder-1.5B-Instruct · HumanEval+ · MBPP+

Um resultado negativo aprofundado por diagnóstico e análise econômica.

#pagebreak()
= Tarefa, trajetória e ação

- AUC agregada pode reconhecer dificuldade do prompt.
- WT-AUC compara corretos e incorretos da mesma tarefa.
- Probabilidades precisam sustentar decisões de aborto.
- Resolução e custo avaliam o sistema sequencial.

Um estado determinístico do prompt empata as tentativas da mesma tarefa.

#pagebreak()
= Desenho e história

- Piloto: 960 trajetórias, 32 tarefas.
- HumanEval+: 2.640 tentativas, 132 tarefas.
- MBPP+ congelado: 640 tentativas, 32 tarefas.
- Em 32 tokens: 2.158 candidatos ativos e 92 tarefas mistas.

26 tarefas principais constam de artefatos anteriores. O registro histórico não contém aprovação do orientador; não se reivindica pré-registro independente.

#pagebreak()
= Validação por tarefas

- Cinco folds externos; tarefa inteira permanece no mesmo fold.
- Seleção interna de C e camada com perda ponderada por tarefa.
- Cross-fitting de toda a seleção antes de Platt.
- Bootstrap pareado de tarefas, condicional aos escores ajustados.

No residual, a seleção e calibração do primeiro estágio também excluem o rótulo pontuado.

#pagebreak()
= Resultado original em 32 tokens

#table(columns: 4, inset: 7pt,
 table.header([*Preditor*], [*AUC*], [*WT-AUC*], [*Log loss*]),
 [B], [0,701], [0,628], [0,630], [H], [0,592], [0,568], [0,687],
 [H+B], [0,646], [0,581], [0,668])

ΔWT-AUC = −0,0470 [−0,0994; 0,0056].

Δlog loss = +0,0379 [0,0067; 0,0706].

Os critérios originais de ganho incremental e benefício conjunto não foram satisfeitos.

#pagebreak()
= Leituras alternativas: o que aprendemos?

- NLL média e margem preservam parte importante do ranking.
- PCA64: WT-AUC 0,630; log loss 0,632.
- Residual ancorado em B: WT-AUC 0,635; log loss 0,629.
- Intervalos contra B incluem zero; penalização conjunta mais forte não resolveu.

Sensibilidade à parametrização é sustentada; causa única e superioridade geral não são identificadas.

#pagebreak()
= Retenção não é garantia externa

#image("../transfer/retention_curves.svg", width: 23cm)

#text(size: 13pt)[Origem OOF, desenvolvimento e alvo congelado são avaliações distintas. Limiares definidos apenas na origem.]

#pagebreak()
= Candidatos descartados e cobertura

- H+B no alvo: 276/366 corretos ativos mantidos, 75,41%.
- 90 descartes distribuídos por 12 tarefas; cinco concentram 60.
- Mbpp/166 perde oito sucessos; Mbpp/259 perde seu único sucesso.
- Entre corretos ativos iniciados: 23/48 abortados.

O denominador muda com o caminho de execução. Duas tarefas perdem resolução.

#pagebreak()
= Custo até sucesso

Condicionado à tarefa, com tentativas iid:

$ E[C(N)]=E[D] sum_(j=0)^(N-1)(1-p r)^j. $

Para orçamento ilimitado, p>0 e r>0:

$ E[D]/E[T] < r. $

Retenção global, heterogeneidade, cobertura finita e overhead precisam entrar.
Tokens, latência, energia e FLOPs são medidas distintas.

#pagebreak()
= Oracle retrospectivo

#image("../economics/oracle_frontier.svg", width: 23cm)

#text(size: 13pt)[Em t=8, N=20: economia idealizada 77,75% em HumanEval+ e 76,25% em MBPP+. Conhece os rótulos; preserva todos os corretos.]

#pagebreak()
= Integridade e limites

- Métricas e os três scorers principais reconstituídos.
- 70 unidades de leitura exploratória repetidas com probabilidades idênticas.
- 50 pares / 100 gerações: tokens e estados idênticos entre capturas.
- 72 execuções de tarefa: quatro modos, três repetições, N=3.
- RTX3050 Laptop 6 GB; tempos operacionais descritivos.

Um modelo, dois benchmarks, exposição histórica e intervalos condicionais.
Nenhuma causa semântica, garantia externa ou aceleração da política é presumida.

#pagebreak()
= Avaliação independente e contribuição

120 novas tarefas MBPP+, 2.400 tentativas principais e 400 técnicas disjuntas.

B e H+B de origem, no mesmo t=8, sem ajuste no alvo.

Retenção H+B−B: -0,1636 [-0,2402, -0,0943]. Critério próprio: satisfeito.

Congelamento anterior às trajetórias; aprovação do estudante e revisão do orientador informada por ele.

O novo endpoint não modifica Q2/H3 originais nem identifica uma causa de domain shift.
