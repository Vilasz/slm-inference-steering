== Interpretação dos resultados do desenho fixo

O incremento primário não satisfaz os critérios estabelecidos antes do banco
principal. A comparação por tarefa evita confundir facilidade entre problemas
com ordenação de trajetórias de um mesmo problema. O valor observado é -0,047, IC 95% [-0,099; 0,006].
Mesmo um efeito positivo não demonstra mecanismo causal: o estado é uma
representação do prefixo produzido pelo próprio modelo.

Os critérios conjuntos para afirmar benefício operacional não foram satisfeitos. A conclusão operacional depende das duas comparações pareadas,
do risco de perder sucessos e da execução real. O custo em tokens não mede
automaticamente tempo, energia ou custo de prefill. A transferência constitui
um teste de transportabilidade limitado a outro benchmark de código.

Os intervalos de bootstrap mantêm os modelos OOF fixos; não incluem toda a
variabilidade do procedimento de treino. As tarefas HumanEval não são
inteiramente inéditas ao histórico da pesquisa: 26 tarefas principais estão
presentes em artefatos anteriores. O protocolo é de desenho fixo com execução
autorizada pelo estudante; o registro histórico v0.2 não contém aprovação do
orientador arquivada e não se reivindica pré-registro independente. Resultados secundários e históricos
preservam a regra principal e seus critérios de confirmação.

== Síntese dos diagnósticos exploratórios

O intervalo principal de ΔWT-AUC vai de -0,0994 a 0,0056. O Δlog loss é 0,0379, IC 95% [0,007; 0,071]. Esses limites descrevem a precisão alcançada sob modelos fixos; não são uma prova de equivalência nem de ausência geral de sinal.

Em MBPP+, o gate H+B retém 276 dos 366 candidatos corretos ativos (0,7541, IC 95% [0,619; 0,877]). O alvo empírico de 0,97 no desenvolvimento não constitui garantia externa. A retenção observada descreve o transporte do gate congelado, sem identificar isoladamente as causas de diferenças entre origem e alvo.


== O que explica e o que permanece indeterminado

As ablações distinguem o resultado de um procedimento da alegação de ausência
de informação. PCA e o offset ancorado em B reduzem a desvantagem do H+B
original, mas não demonstram superioridade sobre B nos intervalos adicionais.
Penalização conjunta mais forte não oferece a mesma recuperação. Portanto,
a parametrização merece atenção; dimensão, coordenadas, penalização e
calibração não foram isoladas como causas independentes. A baixa correlação
entre escores tampouco certifica utilidade incremental ou independência condicional.

O diagnóstico do gate revela uma falha de transporte de decisões de alta retenção
empírica. Os descartes se concentram em parte das tarefas e podem eliminar toda
sua cobertura. Diferenças entre origem OOF, desenvolvimento e aplicação de um
modelo completo ao alvo impedem uma atribuição causal simples a domain shift.
O desenho independente mantém os scorers e compara ambos os gates na mesma
população ativa, sem escolher o melhor classificador exploratório.

O oracle mostra que há oportunidade econômica no banco quando é possível
identificar perfeitamente os descartes. Grande parte desse potencial está em
tarefas não resolvidas. A lacuna entre esse contrafactual e os gates disponíveis
separa um problema de oportunidade computacional de um problema de decisão
estatística. Ela não demonstra que um método online consiga fechar a lacuna.

== Alcance da avaliação independente

A nova comparação usa 120 tarefas e o mesmo checkpoint para os dois gates. O contraste de retenções é -0,1636, IC95% [-0,240; -0,094] e satisfaz seu critério próprio. A revisão da proposta pelo orientador foi informada pelo estudante antes do congelamento. Esse registro não cria aprovação retroativa do protocolo original. A confirmação nova, quando sustentada, refere-se ao contraste de decisões congeladas; não identifica uma causa do transporte nem valida ganho incremental, latência ou outros modelos.
