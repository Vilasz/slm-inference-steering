#import "../lib.typ": *

= Discussão <chap-discussao>

#include "../generated/discussion_findings.typ"

== O que o resultado inconclusivo permite afirmar

A interpretação deve preservar a diferença entre uma afirmação sobre o procedimento avaliado e uma afirmação sobre toda a informação disponível. Os probes receberam representações específicas, tiveram capacidade e regularização limitadas e foram ajustados em amostras finitas. Não demonstrar incremento nesse desenho significa que a evidência não estabeleceu a superioridade desse procedimento sobre seu baseline. Não significa que todas as funções dos estados internos sejam incapazes de prever corretude.

Também não se conclui equivalência porque o intervalo inclui zero. No contraste primário DeepSeek/MLP, o intervalo ainda comporta ganhos de aproximadamente 0,05 de WT-AUC. A análise de planejamento já indicava poder limitado para diferenças pequenas; o teste externo encontrou menos tarefas mistas do que o cenário histórico de referência. A contribuição não é eliminar essa incerteza por linguagem mais forte, mas explicitar os efeitos que a avaliação ainda não distingue.

A evidência original é mais específica em outro aspecto: a diferença de log loss positiva indica pior qualidade probabilística de H+B naquele ajuste. A extensão acrescenta que esse comportamento não é uniforme. No DeepSeek, as perdas secundárias favorecem H+B; no Qwen com MLP, o contrário ocorre. Essas observações ampliam a interpretação do fenômeno, sem substituir o contraste primário por uma métrica escolhida após conhecer seu sinal.

== Capacidade do probe e acesso ao sinal

A comparação entre logística e MLP enfrenta uma limitação real da primeira etapa. Um probe linear não representa todas as relações entre prefixo e resultado; por outro lado, acrescentar flexibilidade aumenta a necessidade de regularização e de dados. Os resultados não sustentam que a simples troca por um MLP resolva a questão incremental.

A interação positiva secundária no Qwen exige uma leitura particularmente cuidadosa. $Gamma$ compara dois incrementos: ele pode crescer porque H+B melhora, porque B piora ao mudar de classe, ou pelos dois motivos. As métricas pontuais mostram que esses movimentos podem coexistir. Portanto, maior interação não isola uma melhoria absoluta do uso das ativações. O teste de H+B/MLP contra B/MLP continua sendo necessário.

A igualdade de largura e orçamento de seleção controla uma parte importante da comparação, mas não iguala o número de parâmetros: H+B possui mais entradas. O PCA64 reduz essa diferença e mantém o mesmo número de componentes latentes nos dois geradores, sem garantir que tais componentes expressem o mesmo conteúdo. O whitening e a seleção pela perda também podem influenciar a estabilidade. Os resultados dizem respeito a esse procedimento completo, não a uma medida direta da quantidade de informação semanticamente representada.

As ablações exploratórias da primeira etapa já mostravam que o resultado era sensível à leitura. A nova avaliação aproveita essa motivação sem apagar sua origem posterior. Não é uma repetição exata do H+B original: além do benchmark e dos candidatos, mudaram a representação reduzida e o esquema de desenvolvimento. A comparação controlada relevante ocorre entre as quatro famílias dentro da segunda etapa.

== Modelo, tokenizer e população sob risco

Avaliar duas famílias amplia o escopo em relação ao Qwen isolado, mas não torna a comparação uma experiência causal sobre arquitetura. Os modelos diferem em pré-treinamento, ajuste instrucional, vocabulário, formatação e comprimento das respostas. Essas diferenças afetam tanto os observáveis quanto a composição dos candidatos que chegam a 32 tokens.

O contraste de comprimentos é expressivo: mais da metade das respostas Qwen do teste termina até o checkpoint, enquanto quase todas as respostas DeepSeek continuam ativas. O mesmo número de tokens, portanto, não implica a mesma fração de programa observada ou a mesma oportunidade de economia. No Qwen, parte importante dos sucessos e fracassos já foi encaminhada ao verificador antes que o gate possa atuar.

Essa seleção também limita comparações diretas de WT-AUC entre modelos. As 58 tarefas mistas do DeepSeek e as 35 do Qwen não são dois grupos independentes nem a mesma população. A análise nas 28 tarefas mistas comuns conserva o pareamento, mas responde a uma pergunta mais restrita. Ela não permite declarar que uma família contém universalmente mais sinal precoce que a outra.

Um checkpoint semanticamente alinhado, por instrução de código ou por proporção do comprimento, seria outra condição experimental. Seu desenvolvimento deveria ocorrer sem escolher o ponto que produz o melhor resultado nas tarefas externas já examinadas. A constatação atual é uma limitação do uso de posições tokenizadas fixas, e não autorização para alterar retrospectivamente o endpoint.

== Probabilidades, limiares e transporte

Ordenação e qualidade probabilística podem variar separadamente. Uma transformação monotônica das previsões preserva a AUC, mas altera perdas e decisões em limiares numéricos. Mesmo entre modelos que não são simples transformações um do outro, melhorar o nível das probabilidades não implica produzir uma ordenação intratarefa mais informativa. Por isso, as melhorias secundárias de log loss e Brier no DeepSeek são compatíveis com uma WT-AUC incremental inconclusiva.

A política utiliza um limiar escolhido em tarefas de calibração da origem. Sua retenção de desenvolvimento é um resultado no conjunto de ajuste e não uma certificação para MBPP+. A validação independente dos gates anteriores demonstra essa limitação de forma direta: um alvo de retenção alto no desenvolvimento não impediu uma perda substancial de corretos no transporte. O novo teste, com outros probes, não revoga essa constatação.

Também não se identifica a causa exclusiva da mudança como domain shift. Prevalência, dificuldade, comprimento, formatação e o ajuste final sobre a origem podem contribuir. A comparação entre OOF, desenvolvimento e aplicação congelada precisa permanecer explícita. Recalibrar sobre o conjunto externo e então avaliar nesse mesmo conjunto eliminaria justamente o teste de transporte que se pretende realizar.

== O que a economia observada representa

O replay distingue a economia disponível por parar após verificar um sucesso daquela que depende de interromper um candidato. A primeira já está presente no baseline com verificador. A contribuição do gate deve ser medida contra esse baseline, com a mesma ordem, população e limite de tentativas.

Na segunda etapa, algumas políticas reduzem tokens sem perda observada de tarefas, mas os tamanhos e os compromissos são diferentes. No DeepSeek, a economia de H+B/MLP é pequena em relação ao custo do verificador. No Qwen, B/MLP gera menos tokens que H+B/MLP, enquanto H+B/MLP preserva mais corretos ativos potenciais. Essa diferença ilustra por que uma preferência econômica depende da penalidade atribuída aos sucessos descartados e da disponibilidade de outros sucessos na tarefa.

Nenhuma perda observada não é garantia de nenhuma perda futura. Em um conjunto em que todas as diferenças de resolução são zero, reamostrar as mesmas tarefas também produz diferenças zero. Um intervalo empírico degenerado não representa evidência de que um evento raro seja impossível em outras tarefas. A avaliação operacional desta etapa é secundária e não fornece uma nova certificação de não inferioridade.

Além disso, o custo de tokens não incorpora toda a política. O estado após o token do checkpoint exige processamento, a captura transfere dados, e o probe precisa ser executado. Uma economia pequena pode ser anulada por esses custos em tempo, mesmo quando o score em lote parece barato. A pesquisa mede condições técnicas, mas não estabelece aceleração online ou retorno financeiro.

O oracle utiliza informação indisponível no momento da decisão. Sua economia mostra que existe trabalho dispensável sob conhecimento perfeito, sem indicar que esse trabalho possa ser identificado com alta retenção a partir do prefixo. A distância observada não é uma meta garantida para outro classificador.

== A contribuição da formulação sequencial

A recursão de Bellman reúne elementos que AUC e retenção isoladas não representam: tentativas restantes, orçamento, custo de completar e valor de abandonar para reiniciar. No caso simplificado, continuar quando p vezes o custo de fallback excede o custo restante decorre da comparação dos dois custos esperados. A interpretação depende de ambos os caminhos conduzirem ao mesmo fallback após fracasso.

No sistema real, completar uma tentativa consome recurso e pode atualizar a informação sobre a tarefa. A condição simplificada deixa de ser geral. Essa distinção evita apresentar um limiar fixo ajustado por recall como uma política ótima. As probabilidades estimadas, seus erros de calibração e o custo de adquirir o sinal são parte da decisão, não detalhes externos a ela.

A contribuição matemática é explicativa e delimitada. As identidades de probabilidade, o custo por soma geométrica e a programação dinâmica têm antecedentes. Sua utilidade aqui é mostrar que um bom decodificador de corretude e um bom controlador de inferência não são o mesmo objeto.

== Limitações de precisão e de validade

A unidade independente relevante para generalização é a tarefa. Milhares de trajetórias e de pares positivo-negativo não substituem centenas de tarefas distintas. A restrição às mistas e ao conjunto ativo reduz ainda mais o tamanho efetivo para certos estimandos. As perdas, a AUC e o replay possuem denominadores diferentes, que precisam acompanhar cada conclusão.

Os intervalos condicionam nos probes treinados. Oito repetições de seleção e ajuste na origem mostram sensibilidade à perturbação das tarefas, mas não constituem uma estimativa completa da incerteza de treinamento. Não se deve somar essa dispersão mecanicamente ao IC externo ou apresentá-la como uma cobertura frequentista certificada de todo o procedimento.

As garantias de isolamento também têm escopo. Separar folds por tarefa evita um tipo de vazamento, mas não elimina decisões humanas informadas por exposições anteriores. A primeira etapa declara 26 tarefas principais presentes no histórico. A nova reserva excluiu resultados locais anteriores e variantes lexicais identificadas; não certifica independência semântica total ou ausência de contaminação no pré-treinamento.

Por fim, passar em testes finitos não é uma prova de correção para todas as entradas. Parser, limites de execução e cobertura das suítes integram a medida. A reavaliação dos labels Qwen no sandbox novo sustenta a compatibilidade dos desfechos reutilizados, sem tornar o verificador perfeito. As conclusões permanecem condicionais a um cenário com acesso a verificação funcional forte.

== Continuidade científica

As próximas perguntas devem preservar a separação entre desenvolvimento e teste. Prioridades são ampliar o número de tarefas realmente novas, avaliar funções e projetos mais longos, comparar baselines textuais mais fortes e medir a política online com overhead completo. Um checkpoint definido por evento de código exigiria um novo protocolo.

Aumentar a capacidade dos probes também continua sendo uma possibilidade, mas já não pode ser descrita como uma direção inteiramente ausente deste estudo: um MLP pequeno foi efetivamente avaliado. Outras arquiteturas precisariam de amostra, regularização e controles próprios, sem usar o conjunto externo atual como fonte reiterada de seleção.

Intervenções nas ativações respondem a uma pergunta causal diferente. Uma direção que prediz corretude não necessariamente permite aumentar sua probabilidade quando modificada. Avaliar steering, ou reinvestir a economia em novas tentativas, requer experiências adicionais. A conclusão desta obra é o que as duas etapas permitem afirmar sobre leitura e decisão, mantendo essas extensões como hipóteses futuras.
