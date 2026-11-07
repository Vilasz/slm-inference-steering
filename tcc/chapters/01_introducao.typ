#import "../lib.typ": *

= Introdução <chap-introducao>

== Gerar uma solução e decidir quando continuar

Um modelo de linguagem pode produzir programas diferentes para a mesma especificação. Alguns passam nos testes; outros falham por erros de sintaxe, de execução ou de lógica. Essa variabilidade permite melhorar a cobertura por amostragem repetida: mesmo quando a primeira tentativa falha, outra pode fornecer uma solução válida. HumanEval consolidou a avaliação por corretude funcional e a medida pass\@k @chen2021codex. Estudos de amostragem repetida também mostram que o orçamento de inferência pode aumentar a cobertura de tarefas, desde que exista um meio adequado de verificar ou selecionar as respostas @brown2024.

O aumento de cobertura exige mais computação. Para cada candidato, o sistema processa o enunciado, gera tokens e verifica o programa extraído. Parte desse esforço é aplicada a respostas que não serão utilizadas. Surge então a possibilidade de reconhecer uma tentativa pouco promissora enquanto ela ainda está sendo gerada e interrompê-la antes do término.

A dificuldade está no compromisso entre economia e perda de soluções. Um prefixo aparentemente ruim pode admitir uma continuação correta. Ao interromper a primeira solução que resolveria o problema, o controlador talvez precise iniciar outras tentativas ou encerre a busca sem sucesso. Portanto, uma previsão de corretude não basta para demonstrar uma política econômica: é necessário avaliar o momento da decisão, os sucessos descartados, as tarefas resolvidas e o custo efetivo.

Este trabalho investiga esse problema em Qwen2.5-Coder-1.5B-Instruct e DeepSeek-Coder-1.3B-Instruct, dois modelos pequenos especializados em código. O termo *Small Language Model* (SLM) designa aqui seu porte aproximado, sem pressupor uma fronteira universal entre modelos pequenos e grandes. Os experimentos foram executados em uma GPU pessoal de 6 GB. Essa restrição motiva um desenho compacto, mas não substitui os controles necessários à interpretação científica.

== O que significa um sinal de corretude

Há quatro perguntas distintas por trás da expressão “prever corretude”. A primeira é reconhecer tarefas em que o gerador costuma acertar. A segunda é distinguir tentativas corretas e incorretas *da mesma tarefa*. A terceira é verificar se o estado interno acrescenta poder preditivo a estatísticas observáveis do prefixo. A quarta é estabelecer se esse acréscimo melhora uma decisão de interrupção. Uma resposta favorável em um nível não implica uma resposta favorável no seguinte.

Considere um classificador que atribui a mesma probabilidade elevada a todas as respostas de um problema fácil e uma probabilidade baixa a todas as respostas de um problema difícil. Ele pode ordenar bem as respostas quando os dois problemas são misturados, mas não indica qual tentativa merece continuar dentro de nenhum deles. A avaliação intratarefa é necessária para separar essas situações.

As representações internas são vetores calculados pelo Transformer a partir do enunciado e dos tokens já processados. Um *probe*, ou decodificador estatístico de corretude, relaciona essas representações ao resultado funcional futuro, sem alterar os pesos do gerador. Ele é diferente do mecanismo que amostra o próximo token. Neste texto, $H$ representa o estado latente selecionado e $B$ reúne observáveis como logprobabilidades, entropia e margem de logits. O contraste central compara um probe que usa apenas $B$ com outro que usa $H+B$.

Esse contraste depende também da capacidade do classificador. Se apenas $H+B$ recebesse uma rede neural, uma eventual melhora poderia decorrer da classe de funções mais expressiva, e não do uso de $H$. Por isso, a segunda etapa cruza os dois conjuntos de features com regressão logística e um MLP pequeno de uma camada oculta, aplicando o mesmo desenho aos dois geradores. A comparação entre modelos e entre classes de probes integra, assim, a pergunta sobre o alcance dos resultados.

#caixa("Pergunta central")[Em modelos pequenos de geração de código, estados internos disponíveis após um prefixo curto melhoram a discriminação de corretude entre tentativas da mesma tarefa, além dos observáveis escolhidos, e esse ganho depende da classe do probe ou se traduz em uma política de interrupção útil?]

“Além dos observáveis” significa além do vetor $B$ especificado. Não significa informação adicional em relação ao prompt, ao prefixo completo e ao conhecimento integral do gerador, pois o estado interno é uma função desses elementos. Tampouco se interpreta um probe como consciência ou conhecimento infalível do futuro.

== Objetivos e sequência de evidências

O objetivo geral é avaliar o valor incremental e operacional dos estados internos precoces, distinguindo o que cada procedimento consegue acessar e o que permanece incerto. Cinco perguntas articulam a obra:

#grade(caption: [Perguntas que organizam as duas etapas do estudo.],
 [*Pergunta*], [*Evidência necessária*],
 [Q1 — Previsibilidade], [Estimar corretude separando diferenças entre tarefas e entre tentativas.],
 [Q2 — Incremento], [Comparar $H+B$ e $B$ com a mesma informação temporal e a mesma classe de probe.],
 [Q3 — Capacidade e modelo], [Investigar a contribuição de $H$ com logística e MLP, em Qwen e DeepSeek.],
 [Q4 — Transporte], [Avaliar probabilidades e decisões congeladas em tarefas externas, sem reajuste no alvo.],
 [Q5 — Decisão e custo], [Medir resolução, custo e distância a um oracle contrafactual que conhece os resultados.],
)

Essas perguntas organizam a redação final; não substituem retrospectivamente os identificadores ou os critérios dos protocolos executados. A investigação teve duas etapas, conectadas por diagnósticos exploratórios. A primeira avaliou Qwen em HumanEval+, com probes lineares, validação aninhada por tarefa e checkpoint principal de 32 tokens. A transferência para MBPP+ e uma validação independente de gates congelados examinaram o transporte das decisões. Após a inspeção desses resultados, ablações com PCA e correção residual investigaram a sensibilidade à parametrização.

A segunda etapa transformou essa limitação em uma pergunta prospectiva: no DeepSeek, $H+B$ com MLP melhora a WT-AUC em relação a $B$ com MLP aos 32 tokens? Os procedimentos foram desenvolvidos em HumanEval+, e ambos os geradores foram avaliados nas mesmas cem tarefas MBPP+ sem resultados locais anteriores. Cada um produziu vinte respostas por tarefa: quatro mil novas gerações externas. DeepSeek/MLP é o contraste confirmatório primário dessa etapa; as demais comparações e o replay são secundários.

Essa cronologia é parte da interpretação científica. Os diagnósticos motivaram a segunda etapa, mas não se tornam confirmatórios porque agora aparecem em uma obra integrada. Da mesma forma, a extensão não altera os números ou as decisões originais. O encadeamento permite investigar o alcance de um resultado inicial sem apresentar escolhas posteriores como se tivessem sido antecipadas.

== Contribuições e delimitação

A contribuição empírica combina controle por tarefa, comparação com observáveis, duas classes de probes e avaliação externa em duas famílias de SLMs. A contribuição metodológica é tornar explícito por que AUC agregada, ordenação intratarefa, qualidade probabilística e utilidade de interrupção exigem estimandos distintos. A formulação sequencial relaciona probabilidade de sucesso, custo restante, orçamento e penalidade por insucesso; trata-se de uma aplicação delimitada de programação dinâmica, sem reivindicação de teorema inédito.

Nenhuma das duas etapas demonstrou ganho incremental de WT-AUC em seu contraste primário. Esse resultado não autoriza uma afirmação universal de ausência de sinal latente. Há diferenças entre as métricas probabilísticas e de ranking, e entre as consequências operacionais dos modelos. A análise dessas diferenças, com seus intervalos e denominadores, constitui o centro da discussão.

O alcance permanece restrito a dois modelos, dois benchmarks de funções Python relativamente curtas e às classes de probes examinadas. Ambos os benchmarks são públicos; a auditoria de exposição local não elimina contaminação no pré-treinamento. Não foram avaliadas intervenções causais nas ativações, projetos com múltiplos arquivos ou políticas que reinvistam a economia em novas tentativas.

A fundamentação e a metodologia antecedem os resultados das duas etapas, apresentados no @chap-primeira e no @chap-comparacao. A discussão e a conclusão reúnem as respostas à pergunta central.
