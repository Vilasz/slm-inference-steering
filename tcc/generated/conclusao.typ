= Considerações finais <chap-conclusao>

Este trabalho avaliou sinais latentes precoces de corretude em geração de código, com cinco perguntas articuladas. Sobre *previsibilidade (Q1)*, o probe H isolado alcançou WT-AUC pontual de 0,568 no experimento original. Capacidade preditiva dos estados não implica automaticamente incremento sobre observáveis ou benefício de interrupção; a avaliação separou esses níveis, preservando a tarefa como unidade de generalização.

Quanto ao *incremento (Q2)*, a primeira etapa avaliou Qwen2.5-Coder-1.5B-Instruct em 2.640 tentativas de 132 tarefas HumanEval+, com validação aninhada e probes lineares. Em 32 tokens, ΔWT-AUC(H+B−B) foi −0,047, IC95% [−0,099; 0,006]; Δlog loss foi 0,038, IC95% [0,007; 0,071]. O critério incremental original não foi satisfeito, sem benefício operacional conjunto demonstrado.

Sobre *capacidade e modelo (Q3)*, os diagnósticos exploratórios motivaram a comparação prospectiva com DeepSeek-Coder-1.3B-Instruct e logística/MLP nos dois geradores. Oito procedimentos foram desenvolvidos em HumanEval+ e congelados antes de quatro mil gerações externas nas mesmas cem tarefas MBPP+ por modelo. No contraste primário DeepSeek/MLP, ΔWT-AUC foi −0,0042, IC95% [−0,0648; 0,0548], em 58 tarefas mistas: resultado inconclusivo, sem ganho, deterioração ou equivalência demonstrados.

A interação secundária no Qwen reduz uma desvantagem linear, sem demonstrar incremento positivo. Melhorias secundárias de probabilidade no DeepSeek também não substituem o teste de ordenação. Não se estabeleceu efeito uniforme das ativações ou da maior expressividade.

Quanto ao *transporte (Q4)*, a validação independente em 120 tarefas MBPP+ confirmou uma limitação dos gates anteriores de oito tokens: retenção de 82,72% para H+B e 99,08% para B, diferença de −16,36 pontos percentuais. Essa confirmação específica não estende o resultado a todo procedimento de leitura latente.

Sobre *decisão e custo (Q5)*, alguns replays economizam tokens sem perda observada de tarefas. O achado secundário não certifica retenção populacional, aceleração ou superioridade econômica geral de H+B. O oracle delimita oportunidade sob informação perfeita; Bellman explica o papel de custo restante, orçamento e reinício, sem tornar os gates empíricos ótimos.

Os intervalos são condicionais aos probes ajustados; poucas tarefas mistas e variabilidade de treinamento limitam a precisão. Dois geradores e funções curtas públicas não representam toda a geração de software nem eliminam contaminação no pré-treinamento. Os resultados não demonstram ausência universal de informação latente.

A contribuição é uma avaliação controlada que separa previsibilidade, incremento e economia. Avançar requer novas tarefas e distribuições, baselines textuais mais fortes e medições online de custo, preservando a separação entre desenvolvimento e teste.
