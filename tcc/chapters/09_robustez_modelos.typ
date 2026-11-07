= Robustez entre modelos e capacidade dos decodificadores de corretude

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
A avaliação externa está concluída; os resultados são apresentados abaixo.

== Comparação controlada e diferenças arquiteturais

Qwen2.5-Coder-1.5B-Instruct possui 28 camadas e estados
de dimensão 1536; DeepSeek-Coder-1.3B-Instruct possui
24 camadas, estados de dimensão 2048 e
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

O inventário verificou 378 tarefas MBPP+,
173 IDs já expostos e 189
elegíveis após triagem lexical conservadora. Essa triagem não certifica
independência semântica completa ou ausência de exposição no pré-treinamento.
O tamanho planejado é 100 tarefas por modelo,
vinte tentativas por tarefa, condicionado ao orçamento medido; prefixos de uma
ordem SHA256 fixa definem reduções técnicas predefinidas. Os mesmos IDs externos
são usados em ambos os modelos e não são substituídos por resultado.

O planejamento usa 132 tarefas históricas, 92 mistas, e desvio-padrão das diferenças
pareadas de 0,266. Os efeitos populacionais
conhecidos abaixo foram simulados com reamostragem de heterogeneidade por tarefa
e intervalos normais de planejamento. Não é poder pós-hoc do efeito observado.

#table(columns: (1fr, 1fr, 1fr),
 [Efeito conhecido], [Detecção positiva], [Largura esperada IC95%],
 [0,020], [10.3%], [0,124],
[0,030], [15.6%], [0,124],
[0,050], [36.1%], [0,124],
[0,075], [64.9%], [0,124],
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

O piloto DeepSeek contém 12 tarefas e 240 tentativas:
8 tarefas mistas, 1 com todos os
acertos e 3 sem acertos. A taxa agregada de acertos foi
41.7%; truncamento 0.4% e
erro de parsing 0.8%. A geração mediu
7.80 tokens/s e
20.56 segundos por tentativa em média. Esses números
descrevem desenvolvimento, não a generalização final. A viabilidade segundo as
regras fixadas foi true.

Em análise adicional de estabilidade na origem Qwen, oito subamostras de 84
tarefas repetiram seleção, ajuste e calibração das quatro famílias, avaliadas
nas mesmas 26 tarefas de auditoria disjuntas. O desvio das diferenças de WT-AUC
foi 0,036 para logística e
0,045 para MLP. Subamostras
sobrepostas e treino menor não estimam sem viés a variância do procedimento final;
esses resultados são sensibilidades, sem rótulos MBPP novos ou seleção de seed.

=== Contrastes externos por modelo

O contraste primário DeepSeek/MLP foi -0,004, IC95% [-0,065; 0,055]. O intervalo inclui zero; não se demonstrou diferença nem equivalência. A conclusão se restringe ao estado no bloco 17, PCA64, decoder e população ativa desta avaliação.

deepseek/linear: ΔWT-AUC = -0,004, IC95% [-0,067; 0,055], 58 tarefas mistas; interpretação: inconclusive.

deepseek/MLP: ΔWT-AUC = -0,004, IC95% [-0,065; 0,055], 58 tarefas mistas; interpretação: inconclusive.

A interação Γ no deepseek foi -0,000, IC95% [-0,031; 0,032]. É secundária e não substitui o incremento absoluto de H.

qwen/linear: ΔWT-AUC = -0,055, IC95% [-0,135; 0,021], 35 tarefas mistas; interpretação: inconclusive.

qwen/MLP: ΔWT-AUC = -0,007, IC95% [-0,094; 0,077], 35 tarefas mistas; interpretação: inconclusive.

A interação Γ no qwen foi 0,048, IC95% [0,018; 0,081]. É secundária e não substitui o incremento absoluto de H.

#table(columns: (1fr, 1.2fr, 1fr, 1fr, 1fr),
[Modelo], [Probe], [WT-AUC], [Log loss], [Brier],
[deepseek], [B_linear], [0,516], [0,728], [0,267],
[deepseek], [HB_linear], [0,512], [0,723], [0,265],
[deepseek], [B_MLP], [0,522], [0,726], [0,266],
[deepseek], [HB_MLP], [0,518], [0,718], [0,262],
[qwen], [B_linear], [0,660], [0,638], [0,223],
[qwen], [HB_linear], [0,604], [0,641], [0,225],
[qwen], [B_MLP], [0,640], [0,624], [0,217],
[qwen], [HB_MLP], [0,632], [0,652], [0,230],
)


=== Consequências operacionais secundárias

Para deepseek, o orçamento N=20 produziu:

#table(columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
[Política], [Resolvidas], [Tokens médios], [Retenção], [Ab. corretos], [Perdidas],
[B_MLP], [71], [1260.4], [100.0%], [0], [0],
[B_linear], [71], [1260.4], [100.0%], [0], [0],
[HB_MLP], [71], [1243.7], [99.4%], [1], [0],
[HB_linear], [71], [1256.9], [100.0%], [0], [0],
[fixed], [71], [2709.3], [100.0%], [0], [0],
[oracle], [71], [304.4], [100.0%], [0], [0],
[verifier], [71], [1260.4], [100.0%], [0], [0],
)

B_linear: diferença média de tokens contra verificador 0.0, IC95% [0.0; 0.0]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

HB_linear: diferença média de tokens contra verificador -3.4, IC95% [-10.3; 0.0]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

B_MLP: diferença média de tokens contra verificador 0.0, IC95% [0.0; 0.0]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

HB_MLP: diferença média de tokens contra verificador -16.7, IC95% [-35.5; -3.3]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

Para qwen, o orçamento N=20 produziu:

#table(columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
[Política], [Resolvidas], [Tokens médios], [Retenção], [Ab. corretos], [Perdidas],
[B_MLP], [75], [271.8], [96.2%], [0], [0],
[B_linear], [75], [282.8], [97.8%], [0], [0],
[HB_MLP], [75], [300.1], [99.1%], [0], [0],
[HB_linear], [75], [282.1], [97.5%], [0], [0],
[fixed], [75], [791.9], [100.0%], [0], [0],
[oracle], [75], [200.4], [100.0%], [0], [0],
[verifier], [75], [318.7], [100.0%], [0], [0],
)

B_linear: diferença média de tokens contra verificador -35.9, IC95% [-76.1; -9.8]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

HB_linear: diferença média de tokens contra verificador -36.6, IC95% [-75.1; -11.3]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

B_MLP: diferença média de tokens contra verificador -47.0, IC95% [-93.1; -16.3]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

HB_MLP: diferença média de tokens contra verificador -18.7, IC95% [-40.6; -4.5]; diferença de cobertura 0,000, IC95% [0,000; 0,000].

As comparações secundárias têm intervalos descritivos sem controle
de multiplicidade. Log loss, Brier, AUC agregada, calibração e replay completo
estão nas tabelas reconstruídas em `reports/final_extension/`; seu estimando e
denominadores são distintos da AUC intratarefa. O resultado primário se limita
ao procedimento DeepSeek/MLP e à população ativa avaliada.

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
