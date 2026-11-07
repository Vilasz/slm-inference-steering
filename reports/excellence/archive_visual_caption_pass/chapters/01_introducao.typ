#import "../lib.typ": *

= Introdução

== Contexto e motivação

Modelos de linguagem especializados em código permitem formular a síntese de programas como geração de sequências condicionada a uma especificação. Uma única execução pode produzir uma solução incorreta mesmo quando o modelo atribui probabilidade positiva a soluções corretas. Por isso, amostrar diversos candidatos amplia a oportunidade de encontrar uma solução que passe em um verificador. A avaliação de Codex introduziu o HumanEval e consolidou o uso de corretude funcional e de pass\@k na análise de geração de código @chen2021codex. Estudos de amostragem repetida também demonstram que o orçamento de inferência é uma dimensão relevante para ampliar a cobertura de tarefas solucionadas @brown2024.

O benefício da repetição tem um custo. Cada candidato utiliza computação para processar o prompt, gerar tokens e executar testes. Parte desse esforço é gasta em candidatos que, quando completados, falham no verificador. Surge então uma possibilidade: identificar durante a geração trajetórias com baixa probabilidade de sucesso e interrompê-las antes de consumir todo o orçamento previsto. Essa possibilidade só é útil se a identificação ocorrer cedo, se não eliminar sucessos em excesso e se seu custo adicional não anular a economia.

O projeto estuda essa questão em um Small Language Model (SLM), usando o termo para descrever o porte do modelo escolhido, sem pressupor uma fronteira universal entre modelos pequenos e grandes. A infraestrutura validada é uma máquina pessoal com GPU NVIDIA RTX 3050 Laptop de 6 GB. A limitação de recursos motiva um desenho compacto, centrado em um modelo, um benchmark principal e uma verificação externa de robustez.

== Problema científico

Um classificador pode prever corretude porque reconhece que certos problemas são fáceis. Isso não implica que consiga distinguir duas respostas diferentes ao mesmo problema. Essa distinção é decisiva para a interrupção de um candidato: uma avaliação que permanece constante entre todas as tentativas do prompt orienta alocação de orçamento por tarefa, mas não identifica qual tentativa merece continuar.

A pergunta central é:

#caixa("Pergunta de pesquisa")[As representações internas precoces de um SLM contêm informação que distingue trajetórias corretas e incorretas geradas para a mesma tarefa, além da dificuldade do problema e das estatísticas observáveis escolhidas, e essa informação permite reduzir o custo de inferência com múltiplas tentativas?]

A expressão “além dos observáveis” refere-se a uma comparação operacional entre classes de preditores especificadas. Não significa informação adicional em relação ao texto completo do prefixo e ao conhecimento integral do modelo, pois o estado interno é calculado a partir desses elementos. O trabalho tampouco interpreta previsibilidade como consciência, introspecção humana ou conhecimento infalível do futuro.

== Objetivos e sequência de evidências

O objetivo geral é caracterizar o valor preditivo e operacional de estados internos de prefixos em geração verificável de código. As três perguntas do protocolo original organizam o núcleo experimental; duas perguntas complementares delimitam o transporte e a oportunidade econômica:

#grade(
  [*Pergunta*], [*Evidência necessária*],
  [Q1 — Previsibilidade], [Medir em que posições os estados internos predizem o resultado final em tarefas não utilizadas no treinamento.],
  [Q2 — Especificidade], [Comparar trajetórias corretas e incorretas do mesmo problema e avaliar o ganho sobre estatísticas observáveis do prefixo.],
  [Q3 — Utilidade], [Avaliar se uma política de interrupção reduz custo mantendo a resolução dentro de uma margem previamente declarada.],
  [Q4 — Transporte], [Examinar a retenção dos gates congelados em outras tarefas, sem recalibração no alvo.],
  [Q5 — Oportunidade], [Medir a distância entre os gates disponíveis e um oracle contrafactual que preserva todos os corretos.],
)

A contribuição está no encadeamento entre essas evidências. Uma AUC agregada elevada responde apenas a parte de Q1. Uma classificação melhor dentro da tarefa responde a parte de Q2. A existência de informação estatística útil não implica, por si só, que interromper gerações produza ganho operacional. Q4 e Q5 não alteram os critérios originais de Q2 e Q3. O resultado negativo principal é aprofundado por ablações de complexidade, diagnóstico de transferência e uma análise contrafactual do espaço econômico disponível. A extensão independente tem um único estimando de retenção e autorização próprios.

== Delimitação e estado da entrega

O núcleo experimental utiliza Qwen2.5-Coder-1.5B-Instruct, HumanEval+ e múltiplas gerações sob uma configuração de amostragem congelada. MBPP+ foi a extensão externa executada. A avaliação adicional reserva outras tarefas do mesmo MBPP+, sem resultados locais anteriores; não constitui um terceiro benchmark. Foi aprovada pelo estudante, que informou revisão do orientador, e congelada antes das novas trajetórias. Seu estado e seus resultados são apresentados separadamente no capítulo empírico. Activation Steering, efeitos causais de intervenções, transferência entre modelos e extração de características por autoencoders esparsos ficam como trabalhos futuros. Não são condições para completar o TCC.

A avaliação principal usa o desenho fixo v0.2, com piloto separado, banco completo e validação por tarefa. A reanálise histórica permanece exploratória e é apresentada separadamente. As decisões empíricas seguem os critérios definidos antes do banco principal; os capítulos divulgam a exposição histórica e a autorização de execução.

O arquivo `RESEARCH_PROTOCOL.md` registra as decisões experimentais. A versão 0.2 foi executada sob desenho fixo, autorização prospectiva do estudante e snapshot de fontes e configuração anterior ao banco principal. Seu registro histórico mantém a aprovação do orientador como pendente; não se reivindica pré-registro independente. A aprovação posterior da extensão não cria aprovação retroativa do v0.2. A exposição anterior a 26 tarefas principais é divulgada. Protocolos, emenda administrativa, hashes e registros de autorização preservam essa cronologia, sem apresentar escolhas posteriores aos resultados como antecipadas.
