// Documento substituto completo. Compilar junto de referencias.bib.
// Para usar a identidade visual original, altere logo-path para "FGV_logo.png".
#let logo-path = ""
#let titulo = "Sinais Latentes Precoces de Corretude em Geração de Código"
#let subtitulo = "Discriminação entre trajetórias da mesma tarefa e eficiência de Best-of-N em Small Language Models"
#set document(title: titulo, author: "João Felipe Vilas Boas", description: subtitulo)
#set text(font: "New Computer Modern", size: 11pt, lang: "pt", region: "BR")
#set page(paper: "a4", margin: (top: 2.7cm, bottom: 2.5cm, left: 3cm, right: 2.4cm))
#set par(justify: true, leading: 0.65em, first-line-indent: 1.15cm)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set outline(indent: auto)
#set table(inset: 6pt, stroke: 0.35pt + rgb("c5cbd1"))
#show heading: it => block(above: 1.2em, below: 0.7em)[#text(hyphenate: false, size: if it.level == 1 {14pt} else {11.5pt}, weight: "bold", it)]
#show figure.caption: set text(size: 9pt)
#show raw: set text(size: 8.5pt)
#let caixa(titulo, body) = block(width: 100%, inset: 10pt, fill: rgb("f3f5f7"), stroke: (left: 2pt + rgb("315a8c")), above: 0.8em, below: 0.8em)[
  #set par(first-line-indent: 0pt)
  *#titulo* #linebreak() #body
]
#let pendente(body) = caixa("Seção empírica a completar", body)
#let grade(..cells) = {
  set text(size: 9pt)
  set par(first-line-indent: 0pt, justify: false)
  let c = cells.pos()
  table(columns: (1.05fr, 2.3fr), table.header(..c.slice(0, 2)), ..c.slice(2))
}
#align(center)[
  #if logo-path != "" { image(logo-path, width: 28%); v(0.8cm) }
  #text(size: 13pt, weight: "bold")[FUNDAÇÃO GETULIO VARGAS]
  #v(0.2cm)
  ESCOLA DE MATEMÁTICA APLICADA
  #v(2.8cm)
  #text(size: 18pt, weight: "bold", hyphenate: false, titulo)
  #v(0.7cm)
  #text(size: 12pt, subtitulo)
  #v(2.2cm)
  João Felipe Vilas Boas
  #v(0.5cm)
  Orientador: Prof. Dário Oliveira
  #v(1.2cm)
  #text(size: 10pt)[Versão de trabalho: fundamentação teórica, metodologia\ e estrutura para incorporação dos resultados]
  #v(1fr)
  Rio de Janeiro\ Setembro de 2026
]
#pagebreak()
#set page(numbering: "i")
#counter(page).update(1)
#heading(numbering: none, outlined: false)[Resumo]
Este trabalho investiga se representações internas de um modelo de linguagem de pequeno porte contêm sinais precoces sobre a corretude funcional de uma geração de código. O foco está na distinção entre informação sobre a dificuldade de uma tarefa e informação que diferencia trajetórias estocásticas produzidas para o mesmo problema. A proposta utiliza um modelo especializado em código, múltiplas tentativas por tarefa e avaliação funcional por testes automatizados. Estados internos extraídos de prefixos são utilizados por classificadores lineares, comparados a estatísticas observáveis de probabilidade e entropia. A validação separa tarefas entre treinamento e avaliação, controla a seleção de camadas e hiperparâmetros e considera a dependência entre tentativas do mesmo problema. A análise central mede a discriminação dentro da tarefa e o ganho incremental em relação aos observáveis escolhidos. Em seguida, avalia-se uma política de interrupção precoce de candidatos com baixa pontuação, sob uma meta empírica de retenção de trajetórias corretas, comparando custo e resolução com amostragem repetida e parada por verificador. O documento apresenta a fundamentação matemática, o desenho experimental e a estrutura de apresentação dos resultados. Nesta versão, os experimentos confirmatórios ainda não são reportados; a existência, a intensidade e a utilidade operacional do sinal permanecem questões empíricas.

#par(first-line-indent: 0pt)[*Palavras-chave:* modelos de linguagem; geração de código; representações internas; probing; amostragem repetida; inferência adaptativa.]

#heading(numbering: none, outlined: false)[Abstract]
This work investigates whether early hidden representations of a small language model predict the functional correctness of stochastic code-generation trajectories. It distinguishes task-level difficulty information from signals that discriminate successful and unsuccessful trajectories for the same task. The proposed design combines repeated sampling, execution-based evaluation, linear probes, observable confidence baselines, and task-grouped validation. Its central analysis measures within-task discrimination and the incremental predictive value of hidden states relative to the selected observable features. A subsequent experiment evaluates whether early termination of low-scoring candidates improves the compute–resolution trade-off under an empirical successful-trajectory retention target. This version presents the theoretical foundations, experimental methodology, and reporting structure. It contains no confirmatory experimental findings and makes no claim that trajectory-specific prediction or computational savings have already been established.
#pagebreak()
#outline(title: "Sumário", depth: 2)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)

= Introdução

== Contexto e motivação

Modelos de linguagem especializados em código permitem formular a síntese de programas como geração de sequências condicionada a uma especificação. Uma única execução pode produzir uma solução incorreta mesmo quando o modelo atribui probabilidade positiva a soluções corretas. Por isso, amostrar diversos candidatos amplia a oportunidade de encontrar uma solução que passe em um verificador. A avaliação de Codex introduziu o HumanEval e consolidou o uso de corretude funcional e de pass\@k na análise de geração de código @chen2021codex. Estudos de amostragem repetida também demonstram que o orçamento de inferência é uma dimensão relevante para ampliar a cobertura de tarefas solucionadas @brown2024.

O benefício da repetição tem um custo. Cada candidato utiliza computação para processar o prompt, gerar tokens e executar testes. Parte desse esforço é gasta em candidatos que, quando completados, falham no verificador. Surge então uma possibilidade: identificar durante a geração trajetórias com baixa probabilidade de sucesso e interrompê-las antes de consumir todo o orçamento previsto. Essa possibilidade só é útil se a identificação ocorrer cedo, se não eliminar sucessos em excesso e se seu custo adicional não anular a economia.

O projeto estuda essa questão em um Small Language Model (SLM), usando o termo para descrever o porte do modelo escolhido, sem pressupor uma fronteira universal entre modelos pequenos e grandes. A infraestrutura de referência é uma máquina pessoal com GPU NVIDIA RTX 3050, conforme o planejamento do projeto. A versão exata da GPU, a memória disponível e o modo de execução deverão ser registrados no experimento. A limitação de recursos motiva um desenho compacto, centrado em um modelo, um benchmark principal e uma verificação externa de robustez.

== Problema científico

Um classificador pode prever corretude porque reconhece que certos problemas são fáceis. Isso não implica que consiga distinguir duas respostas diferentes ao mesmo problema. Essa distinção é decisiva para a interrupção de um candidato: uma avaliação que permanece constante entre todas as tentativas do prompt orienta alocação de orçamento por tarefa, mas não identifica qual tentativa merece continuar.

A pergunta central é:

#caixa("Pergunta de pesquisa")[As representações internas precoces de um SLM contêm informação que distingue trajetórias corretas e incorretas geradas para a mesma tarefa, além da dificuldade do problema e das estatísticas observáveis escolhidas, e essa informação permite reduzir o custo de inferência com múltiplas tentativas?]

A expressão “além dos observáveis” refere-se a uma comparação operacional entre classes de preditores especificadas. Não significa informação adicional em relação ao texto completo do prefixo e ao conhecimento integral do modelo, pois o estado interno é calculado a partir desses elementos. O trabalho tampouco interpreta previsibilidade como consciência, introspecção humana ou conhecimento infalível do futuro.

== Objetivos e sequência de evidências

O objetivo geral é caracterizar o valor preditivo e operacional de estados internos de prefixos em geração verificável de código. Três perguntas organizam a investigação:

#grade(
  [*Pergunta*], [*Evidência necessária*],
  [Q1 — Previsibilidade], [Medir em que posições os estados internos predizem o resultado final em tarefas não utilizadas no treinamento.],
  [Q2 — Especificidade], [Comparar trajetórias corretas e incorretas do mesmo problema e avaliar o ganho sobre estatísticas observáveis do prefixo.],
  [Q3 — Utilidade], [Avaliar se uma política de interrupção reduz custo mantendo a resolução dentro de uma margem previamente declarada.],
)

A contribuição pretendida está no encadeamento entre essas evidências. Uma AUC agregada elevada responde apenas a parte de Q1. Uma classificação melhor dentro da tarefa responde a parte de Q2. A existência de informação estatística útil não implica, por si só, que interromper gerações produza ganho operacional.

== Delimitação e estado da entrega

O núcleo experimental utiliza Qwen2.5-Coder-1.5B-Instruct, HumanEval+ e múltiplas gerações sob uma configuração de amostragem congelada. MBPP+ é a única extensão externa planejada. Activation Steering, efeitos causais de intervenções, transferência entre modelos e extração de características por autoencoders esparsos ficam como trabalhos futuros. Não são condições para completar o TCC.

Este documento substitui a organização anterior, que distribuía o projeto por separabilidade, causalidade, sobrevivência, transferência e inferência adaptativa. A notação e a discussão matemática foram concentradas nos objetos necessários para Q1–Q3. O estado de implementação do repositório não foi auditado nesta revisão; as descrições operacionais abaixo são requisitos e decisões de desenho, não declarações de que os módulos já estejam concluídos.

O arquivo `RESEARCH_PROTOCOL.md` deve funcionar como registro executável das decisões experimentais. A presente versão explicita, no apêndice, correções metodológicas que precisam ser incorporadas a esse protocolo antes de sua versão definitiva. Não se deve chamar de pré-registrada uma escolha feita depois de observar os resultados que ela pretende confirmar.

= Trabalhos relacionados e posicionamento

== Amostragem repetida e verificação funcional

#cite(<chen2021codex>, form: "prose") formulam a avaliação de programas gerados por meio de testes e apresentam um estimador de pass\@k baseado em múltiplos candidatos. #cite(<brown2024>, form: "prose") estudam o aumento de cobertura obtido com mais amostras e destacam a relevância da seleção e verificação. Esses trabalhos motivam a separação entre a capacidade de gerar ao menos uma solução correta e a capacidade de identificá-la.

A qualidade do verificador também importa. #cite(<liu2023evalplus>, form: "prose") apresentam EvalPlus e a ampliação dos testes de HumanEval para HumanEval+, mostrando que testes mais completos revelam falhas que os testes originais não identificam. Neste projeto, a variável resposta é passar em uma versão fixada dessa suíte. Isso fornece um critério reproduzível, mas não uma prova de correção para todas as entradas possíveis.

== Representações internas e estimação de corretude

#cite(<kadavath2022>, form: "prose") diferenciam a avaliação de uma resposta específica da estimativa de que o modelo saberá responder uma pergunta. Essa distinção é próxima da separação entre nível da tarefa e nível da trajetória adotada aqui. #cite(<azaria2023>, form: "prose") estudam classificadores sobre ativações internas para identificar veracidade de afirmações, comparando-os também a medidas baseadas na probabilidade da sequência. A abordagem de Representation Engineering organiza a investigação de fenômenos de alto nível em padrões distribuídos das ativações @zou2023.

Em geração de código, #cite(<dicicco2026>, form: "prose") reporta decodificação linear de corretude a partir do estado do último token do prompt em Qwen3-4B-Instruct-2507 sobre tarefas de LiveCodeBench. O resultado demonstra a relevância de estados anteriores à geração para discriminação entre tarefas. Por construção, porém, um estado determinístico do mesmo prompt é compartilhado pelas suas diferentes tentativas. Essa observação motiva uma avaliação adicional entre trajetórias do mesmo problema.

#cite(<klassert2026>, form: "prose") investiga estimação de incerteza introspectiva em geração de código, com características estáticas e dinâmicas e avaliações no nível da resposta e da linha. O resumo do trabalho reporta sinal de corretude no nível da resposta, sem ganhos consistentes das estratégias dinâmicas mais elaboradas. Isso reforça a opção por um probe simples como experimento principal; não justifica assumir que uma dinâmica temporal mais complexa será superior.

== Dificuldade e interrupção adaptativa

#cite(<zhu2025>, form: "prose") estimam dificuldade percebida pelo modelo a partir de representações iniciais e aplicam essa informação a estratégias de inferência adaptativa. #cite(<ruan2026>, form: "prose") estudam interrupção de episódios de agentes a partir de probes em estados internos, com controle de retenção e certificação em dados independentes. Seu cenário, unidade temporal e procedimento de garantia diferem de uma regra empírica ajustada em prefixos de código.

#let stbon-ok = true
#if stbon-ok [O método ST-BoN @stbon2025 utiliza estados internos em uma estratégia de truncamento e seleção de candidatos. Esse antecedente impede apresentar o uso de hidden states para reduzir o custo de Best-of-N como novidade isolada. A comparação do presente estudo concentra-se na corretude funcional supervisionada, na discriminação dentro da tarefa e na comparação controlada com observáveis.]

== Contribuição pretendida e limites da reivindicação

O trabalho não reivindica ter sido o primeiro a prever corretude com hidden states, o primeiro a estimar dificuldade nem o primeiro a interromper inferência com sinais internos. A contribuição pretendida é uma avaliação integrada e reproduzível, em um regime delimitado de geração de código, de três aspectos: separação entre tarefa e trajetória; poder incremental sobre baselines observáveis especificados; e utilidade em uma política sequencial de interrupção.

As referências recentes acima foram conferidas em suas páginas primárias para esta versão. Resultados resumidos não substituem a leitura integral dos métodos antes da comparação final. Em especial, alegações de ausência de trabalhos idênticos exigem revisão bibliográfica mais ampla; não são inferidas de uma busca limitada.

= Formulação probabilística e notação

== Tarefas, candidatos e configuração

Seja $cal(X)$ o espaço de tarefas e seja $x_i in cal(X)$ o prompt da tarefa $i$, incluindo especificação, assinatura e template fixado. O modelo tem parâmetros $theta$ congelados. Uma configuração $c$ reúne temperatura, top-p, limite de tokens, regras de parada, precisão e versão do tokenizer. A tentativa $j$ produz a sequência aleatória $Y_(i j)=(Y_(i j 1), dots, Y_(i j T_(i j)))$, de comprimento $T_(i j)$.

Distinguem-se a distribuição original do modelo, $p_theta$, e a distribuição efetivamente usada pelo decodificador, $q_(theta,c)$. A lei da sequência amostrada é

$ q_(theta,c) (y | x_i) = product_(u=1)^(T) q_(theta,c) (y_u | x_i,y_(<u)). $

O comprimento é determinado por término natural ou limite computacional. A distribuição da sequência incorpora as regras de parada. Tratar todos os candidatos como tendo comprimento fixo esconderia uma dimensão essencial do custo e da elegibilidade para interrupção.

#grade(
 [*Símbolo*], [*Significado*],
 [$i,j$], [Tarefa e tentativa.],
 [$x_i,c,theta$], [Prompt, configuração de inferência e parâmetros do modelo.],
 [$T_(i j)$], [Número de tokens gerados, incluindo os tokens de término conforme convenção registrada.],
 [$S_(i j)$], [Indicador de aprovação na suíte funcional.],
 [$ell,t$], [Camada do Transformer e posição relativa à completion.],
 [$h_(i j ell t)$], [Estado interno após processar o token de completion $t$.],
 [$B_(i j t),z_(i j t)$], [Vetor de observáveis e score do probe.],
 [$pi_i$], [Probabilidade de sucesso de uma tentativa da tarefa $i$ no regime fixado.],
 [$N,q$], [Número máximo de tentativas e limiar de interrupção.],
 [$A_i,C_i$], [Resolução da tarefa e custo acumulado de uma política.],
)

== Corretude operacional

Um verificador $V$ recebe a tarefa e o programa extraído da completion. Sua saída define

$ S_(i j) = V(x_i,Y_(i j)) in {0,1}, quad pi_i = P(S_(i j)=1 | x_i,c). $

O verificador é mantido fixo em versão, testes, ambiente e limites de execução. Falhas de sintaxe, execução, asserção e extração são classificadas como insucesso. Um timeout do programa é uma falha sob o orçamento especificado. Uma interrupção da infraestrutura de avaliação é um erro de medição e deve ser reexecutada ou marcada como dado indisponível, sem virar automaticamente um rótulo negativo.

Truncamento no limite de geração é registrado como atributo do candidato. O programa extraído é avaliado pela mesma regra funcional: um candidato que terminou por limite pode, excepcionalmente, já conter uma solução válida. Caso o protocolo deseje definir sucesso como “passa e não truncou”, deve nomear essa variável separadamente e não confundi-la com corretude funcional. Esta versão adota a aprovação no verificador como rótulo principal.

== Informação disponível durante a geração

Seja $cal(F)_(i j t)$ a informação observável até o prefixo de comprimento $t$: prompt, tokens amostrados, distribuições de decodificação calculadas e estados acessíveis até esse instante. O resultado final é observado apenas depois de completar e testar o programa. O alvo ideal de um preditor precoce é

$ p^*_(i j t) = P(S_(i j)=1 | cal(F)_(i j t)). $

Antes do término, o mesmo prefixo pode admitir continuações corretas e incorretas. Portanto, “trajetória com baixa chance de sucesso” não significa “trajetória inevitavelmente errada”. Dois candidatos que compartilham exatamente o mesmo prefixo podem receber a mesma previsão e posteriormente obter rótulos diferentes.

Pela propriedade da esperança condicional, a sequência ideal $p^*_(i j t)$ satisfaz, antes ou após absorção no término,

$ E[p^*_(i j,t+1) | cal(F)_(i j t)] = p^*_(i j t). $

Essa identidade é uma propriedade do preditor probabilístico ideal. Não implica que o probe ajustado seja uma martingala, que a confiança cresça monotonamente em cada trajetória ou que a AUC estimada aumente a cada checkpoint.

= Transformers, espaço latente e extração causal

== Representações e atenção causal

Um Transformer representa tokens por vetores e aplica sucessivos blocos de atenção e transformações não lineares @vaswani2017. Em notação esquemática, para uma matriz de estados $H$, uma cabeça de atenção calcula

$ Q=H W_Q, quad K=H W_K, quad U=H W_U, $
$ "Attn"(H)= "softmax"((Q K^top)/sqrt(d_k)+M) U. $

A máscara causal $M$ impede que uma posição consulte tokens posteriores. Conexões residuais, normalizações e transformações feed-forward produzem estados contextualizados. O espaço latente é o espaço vetorial dessas representações, com dimensão determinada pela arquitetura. Não é um espaço de significados com coordenadas semanticamente nomeadas por definição.

O estado $h_(i j ell t) in RR^d$ é uma função do prompt e do prefixo processado. Para um modelo determinístico em modo de avaliação,

$ h_(i j ell t) = f_(theta,ell) (x_i,Y_(i j,1:t)). $

A aleatoriedade da representação decorre principalmente da amostragem de tokens; pesos e procedimento de extração permanecem constantes. A investigação de padrões lineares nesses vetores testa acessibilidade do sinal a uma classe de decodificadores simples.

== Convenção temporal e alinhamento

O checkpoint $t=0$ corresponde ao último token do prompt. Para $t>0$, o estado é extraído após processar o token $t$ da completion, e não o estado que foi usado para amostrar esse token. Em um decodificador autoregressivo, os logits usados para amostrar $Y_t$ dependem do prefixo que termina em $Y_(t-1)$; o estado de $Y_t$ participa da predição seguinte. Confundir essas posições produz um deslocamento de um token e invalida a interpretação temporal.

Um pipeline que para imediatamente depois de amostrar $Y_t$ pode ainda não ter calculado o seu estado. Será necessário processá-lo, sem amostrar $Y_(t+1)$, para decidir a interrupção. Esse forward adicional ou antecipado deve ser incluído na latência e no custo de operação. Contar $t$ tokens gerados continua correto, mas não contabiliza sozinho todo o trabalho do Transformer.

== Propriedade do estado anterior à amostragem

#caixa("Proposição 1 — Ausência de discriminação antes da divergência")[Com prompt, template, pesos e execução determinística fixados, o estado do último token do prompt é igual entre tentativas da mesma tarefa. Qualquer score determinístico calculado apenas desse estado também é constante dentro da tarefa.]

*Demonstração.* Para todo $j$, $h_(i j ell 0)=f_(theta,ell) (x_i)$. Logo, $z_(i j 0)=g(f_(theta,ell) (x_i))$ independe de $j$. Cada comparação entre uma tentativa correta e uma incorreta da tarefa resulta em empate. Uma AUC que atribui peso $1/2$ aos empates vale $1/2$ dentro dessa tarefa.

Esse resultado não impede que `prompt_last` apresente boa AUC entre tarefas. Ele o torna um baseline de dificuldade e um controle de integridade: diferenças relevantes entre repetições do mesmo prompt precisam ser explicadas por mudanças de entrada, dropout, extração incorreta ou não determinismo numérico. Não devem ser interpretadas como previsão da seed futura.

== Teste de ausência de informação futura

Para cada camada e posição selecionadas, compare a extração usando apenas o prefixo e a extração na mesma posição de uma passagem sobre a sequência completa:

$ h_(ell,t) (x,y_(1:t)) approx h_(ell,t) (x,y_(1:T)). $

A igualdade é esperada matematicamente sob atenção causal. Diferenças numéricas podem decorrer de precisão e kernels. O teste deve fixar tolerância absoluta e relativa, modo de avaliação, backend e convenção de normalização. Uma falha exige diagnóstico antes de usar as ativações. Este é um teste de integridade da causalidade temporal da arquitetura, não uma demonstração de que uma direção latente causa corretude.

== Alcance informacional do estado interno

Se $H=f_theta (X,P)$, em que $P$ é o prefixo completo e o modelo é fixado, então $H$ não acrescenta informação aleatória quando $(X,P)$ já é integralmente conhecido. Em notação de informação mútua,

$ I(S;H | X,P)=0. $

A utilidade do estado está em representar o prefixo de modo que certas propriedades se tornem acessíveis a um preditor limitado. Assim, um ganho sobre NLL e entropia sustenta a afirmação de ganho sobre essas estatísticas, e não sobre todo método possível baseado no texto. Essa restrição impede transformar um resultado de probing em uma alegação excessiva sobre introspecção.

= Dificuldade da tarefa e sinal específico da trajetória

== Heterogeneidade entre problemas

A probabilidade $pi_i$ varia com a tarefa e a configuração. Em um banco com $n_i$ tentativas, a proporção $hat(pi)_i=sum_j S_(i j)/n_i$ é uma estimativa descritiva dessa probabilidade. Uma tarefa sem sucessos observados não possui necessariamente probabilidade verdadeira zero; a amostra pode simplesmente não ter encontrado sua região de soluções corretas.

Para descrição do banco, definem-se três categorias mutuamente exclusivas: todas as tentativas corretas, resultados mistos e todas as tentativas incorretas. Esses nomes se referem aos resultados observados. Não autorizam afirmar que o modelo sempre acertará ou nunca resolverá o problema.

O número de tentativas influencia a classificação. Sob amostragem independente e $n$ tentativas, a probabilidade de observar resultados mistos é

$ P("mista" | pi_i,n)=1-pi_i^n-(1-pi_i)^n. $

Assim, a análise entre tarefas mistas representa principalmente problemas com variabilidade detectável nesse orçamento. Não é uma amostra aleatória de todos os problemas de programação.

== Decomposição da associação

Para um score $Z$ e o indicador $S$, a lei da covariância total fornece

$ "Cov"(Z,S) = "Cov"(E[Z|X],E[S|X]) + E["Cov"(Z,S|X)]. $

O primeiro termo captura a associação entre tarefas; o segundo, a associação dentro da tarefa. A decomposição é conceitual e não uma decomposição da AUC. Ela mostra por que uma associação agregada pode ser explicada inteiramente pela dificuldade do prompt.

Como exemplo estritamente ilustrativo, considere duas tarefas: uma com 90% de acertos e outra com 10%. Um score constante igual a 0,9 para a primeira e 0,1 para a segunda pode ordenar bem exemplos agregados. Entretanto, não distingue qual tentativa acertará dentro de nenhuma das tarefas. Nenhum desempenho numérico desse exemplo é um resultado do modelo estudado.

== Discriminação dentro da tarefa

Sejam $cal(P)_i$ e $cal(N)_i$ os conjuntos de tentativas corretas e incorretas elegíveis da tarefa $i$. Define-se a AUC intratarefa por

$ "AUC"_i = 1/(|cal(P)_i| |cal(N)_i|) sum_(j in cal(P)_i) sum_(k in cal(N)_i) psi(z_(i j)-z_(i k)), $

em que $psi(a)$ vale 1 se $a>0$, $1/2$ se $a=0$ e 0 se $a<0$. A média por tarefas mistas é

$ "WT-AUC"=1/(abs(cal(M))) sum_(i in cal(M)) "AUC"_i. $

Cada tarefa informativa recebe o mesmo peso. Uma média por todos os pares de candidatos daria maior peso a tarefas com mais pares positivo-negativo; seria outro estimando. A unidade principal de inferência é a tarefa, mesmo que milhares de pares sejam construídos a partir dela.

Tarefas com apenas uma classe não têm AUC intratarefa definida. Elas continuam nas métricas probabilísticas e operacionais. A exclusão da WT-AUC deve ser acompanhada dos respectivos denominadores, para que um resultado elevado em poucas tarefas não pareça representar todo o benchmark.

== Informação incremental e hipótese operacional

Seja $B_(i j t)$ um vetor fixado de características observáveis do prefixo. Compare um modelo que usa apenas $B$ com outro que combina $B$ e a ativação. Os estimandos centrais são

$ Delta_W = "WT-AUC"_(H+B) - "WT-AUC"_B, $
$ Delta_L = "LogLoss"_(H+B) - "LogLoss"_B. $

Valores positivos de $Delta_W$ e negativos de $Delta_L$ indicam ganho, mas os dois critérios avaliam propriedades diferentes: ordenação dentro da tarefa e qualidade de previsão probabilística. Um ganho robusto sustenta a utilidade incremental da representação para a classe de modelos comparada. Não demonstra independência condicional universal nem que todos os fatores semânticos ou estilísticos foram controlados.

A hipótese nula de ausência de qualquer sinal dentro da tarefa, $S$ independente de $Z$ dado $X$, difere da hipótese de ausência de informação adicional sobre $B$, isto é, $S$ independente de $Z$ dado $(X,B)$. Essa distinção orienta a interpretação dos testes estatísticos posteriores.

= Amostragem repetida, seleção e custo

== Cobertura e pass\@k

Para $N$ candidatos, a cobertura da tarefa é

$ A_i (N)=bold(1){exists j <= N : S_(i j)=1}. $

Sob tentativas independentes e identicamente distribuídas condicionalmente à tarefa,

$ P(A_i (N)=1|x_i)=1-(1-pi_i)^N. $

O ganho de passar de $N-1$ para $N$ tentativas é $pi_i (1-pi_i)^(N-1)$, que decresce quando $0<pi_i<1$. Essa expressão não deve ser aplicada após substituir $pi_i$ por uma taxa global sem considerar heterogeneidade: em geral, a média das coberturas por tarefa difere da cobertura calculada com a média das probabilidades.

Com $n$ candidatos e $c_i$ aprovados, o estimador usual de pass\@k é @chen2021codex

$ hat("pass@k")_i = 1 - binom(n-c_i,k)/binom(n,k), quad k <= n. $

Esse estimador descreve a probabilidade de incluir um sucesso em um subconjunto de tamanho $k$. Já a resolução observada dos primeiros $k$ candidatos é $bold(1){sum_(j=1)^k S_(i j)>0}$. Ambos são úteis, mas não são números intercambiáveis. O replay operacional preserva uma ordem de tentativas e utiliza o segundo objeto.

== Terminologia: Best-of-N e verificador

“Best-of-N” frequentemente descreve gerar candidatos e escolher o melhor segundo um score. Neste documento, o baseline operacional é mais precisamente *amostragem repetida com verificador e parada no primeiro sucesso*. Ele é denominado BoN com verificador por continuidade com o projeto, com essa definição explícita.

A suíte completa de testes é acessada pela política apenas quando um candidato termina. Esse acesso representa um cenário de verificação forte: o programa não recebe os testes ocultos no prompt, mas o controlador pode utilizá-los para decidir a parada. Isso difere da seleção em produção, na qual os testes disponíveis podem ser incompletos. As conclusões operacionais devem ser limitadas a esse cenário, ou acompanhadas futuramente de uma separação entre testes de seleção e de avaliação final.

== Custo até sucesso ou esgotamento

Defina $K_i=min{j:S_(i j)=1}$, adotando $K_i=infinity$ quando não há sucesso no banco. O custo do baseline até o primeiro sucesso ou limite $N$ é

$ C_i^V (N)=sum_(j=1)^N T_(i j) bold(1){K_i>=j}. $

Tarefas não resolvidas contribuem com o custo de todas as tentativas iniciadas. Excluir essas tarefas do custo médio favoreceria políticas que simplesmente desistem de problemas difíceis.

A métrica primária de custo é tokens de completion por tarefa, $overline(C)=sum_i C_i/m$. Uma métrica secundária é o custo agregado por tarefa resolvida,

$ C_"por solução" = (sum_i C_i)/(sum_i A_i), $

que fica indefinido se nenhuma tarefa for resolvida. Ela inclui no numerador o custo das tarefas sem sucesso. Não é a média do custo apenas entre tarefas que tiveram sucesso.

== Retenção de candidatos e efeito sobre resolução

Considere uma política que mantém um candidato com indicador $R_(i j)$, com $R=1$ para candidatos que terminaram naturalmente antes da decisão. O sucesso útil é $S_(i j) R_(i j)$. Defina $r_i=P(R=1|S=1,x_i)$ e $a_i=P(R=1|x_i)$.

Sob tentativas independentes e a mesma regra por tentativa,

$ P(S R=1|x_i)=pi_i r_i, $
$ P(A_i^pi (N)=1|x_i)=1-(1-pi_i r_i)^N. $

A precisão entre candidatos mantidos é $P(S=1|R=1,x_i)=pi_i r_i/a_i$, quando $a_i>0$. Essa precisão pode crescer mesmo quando a cobertura com orçamento fixo diminui. Portanto, “maior fração de acertos entre os mantidos” não basta como evidência de melhoria do sistema.

#caixa("Proposição 2 — Limite da interrupção no mesmo banco")[Quando a política usa exatamente os mesmos $N$ candidatos potenciais e só pode descartá-los, seu conjunto de sucessos é subconjunto do conjunto de sucessos do baseline. Logo, $A_i^pi (N)<=A_i^V (N)$ para cada tarefa.]

*Demonstração.* Se um sucesso é encontrado pela política, o candidato correspondente está no banco original e também passa no verificador do baseline. A política não cria novos candidatos além do limite e não altera o conteúdo dos mantidos. Assim, não pode resolver uma tarefa que nenhum dos $N$ candidatos originais resolvia.

O ganho esperado nesse regime é reduzir custo com perda pequena ou nula de resolução. Para superar cobertura em um mesmo orçamento de tokens, seria necessário reinvestir a economia em novas tentativas, com um banco maior e uma comparação própria. Essa extensão não será confundida com o experimento principal.

== Condição econômica para interromper

Uma política pode economizar tokens em um candidato e aumentar o custo total da tarefa ao eliminar um sucesso precoce, forçando outras tentativas. Por isso, o custo do replay não é necessariamente monotônico no limiar de interrupção.

Sob tentativas iid por tarefa, seja $D$ o custo de uma tentativa sob a política e $s=pi_i r_i$ sua probabilidade de sucesso útil. Como iniciar uma tentativa depende dos resultados anteriores, independentes da tentativa atual,

$ E[C_i^pi (N)|x_i]=E[D|x_i] sum_(j=0)^(N-1)(1-s)^j. $

Para $s>0$, a soma vale $(1-(1-s)^N)/s$; para $s=0$, vale $N$. Essa expressão mostra que a redução do custo por tentativa deve compensar a possível redução na probabilidade de sucesso por tentativa. A análise empírica usa os caminhos efetivos do banco e não depende de estimar $pi_i$ para substituir nessa fórmula.

== Tokens, latência e fronteira custo–resolução

Tokens gerados são uma medida auditável de trabalho de decodificação, mas não equivalem a FLOPs, energia ou tempo. Prefill, comprimento do contexto, cache, extração de estados, cálculo de entropia, transferência GPU–CPU, inferência do probe e verificação têm custos próprios. Os resultados devem distinguir economia de tokens e aceleração medida em execução.

Uma política domina outra no plano descritivo se tem resolução ao menos igual e custo não maior, com uma desigualdade estrita. Na presença de incerteza amostral, a comparação deve reportar diferenças pareadas e intervalos. A fronteira apresentada será a das políticas e limiares previamente definidos; ela não será reinterpretada como uma fronteira ótima global de todos os algoritmos possíveis.

= Desenho experimental e construção dos dados

== Modelo e ambiente

O modelo principal planejado é Qwen2.5-Coder-1.5B-Instruct, integrante de uma família especializada em código @hui2024. Deve-se registrar o identificador completo, revisão imutável dos pesos, tokenizer, template de chat, precisão e backend. A escolha da variante Instruct é específica; não deve ser substituída silenciosamente pela variante base.

A viabilidade na RTX 3050 deve ser medida no piloto. Memória, batch size e eventual quantização dependem da configuração real. Quantização, se necessária, é uma condição experimental que pode modificar as representações; geração e extração devem utilizar a mesma condição. Não se deve gerar candidatos com uma implementação e reconstruir estados de outra sem tratar a diferença como uma análise separada.

== Benchmark principal e unidade de generalização

HumanEval+ é o benchmark principal, avaliado pelo EvalPlus. O snapshot usual contém 164 tarefas, cujo número e identificadores serão confirmados no arquivo efetivamente usado. A avaliação de generalização separa tarefas, não apenas tentativas. Problemas muito semelhantes ou duplicados, se identificados antes dos resultados, devem compartilhar o mesmo grupo de particionamento.

O acesso prévio a algumas tarefas para desenvolvimento não desaparece com a geração de novas seeds. Tarefas usadas para decisões de método devem ser registradas como desenvolvimento. O particionamento verifica a generalização a prompts retidos dentro do benchmark, não certifica ausência de contaminação do benchmark no pré-treinamento do modelo.

== Piloto e congelamento

O planejamento reserva 32 tarefas, selecionadas deterministicamente pelo hash de `protocol-v0.1-pilot:` concatenado a `task_id`, para desenvolvimento. As demais 132 tarefas constituem o conjunto confirmatório, caso nenhuma exclusão adicional por exposição prévia ou corrupção seja necessária. As 32 tarefas não entram nos folds de teste confirmatórios.

O piloto testa temperaturas 0,6, 0,8 e 1,0, com top-p igual a 0,95, dez tentativas por tarefa e limite inicial de 512 tokens. A finalidade é verificar execução, quantidade de tarefas mistas, taxas de truncamento e custo. AUC de probes não deve ser usada para escolher a temperatura. A regra proposta privilegia o número de tarefas mistas, com taxa agregada de aprovação entre 10% e 90%; empates são resolvidos por taxa mais próxima de 50%, menor comprimento médio e menor temperatura.

O critério de “aumento material de truncamento” do protocolo anterior precisa ser quantificado ou removido antes do congelamento. Uma regra subjetiva não constitui critério reproduzível. Se nenhuma configuração for viável, a revisão deverá usar apenas o conjunto de desenvolvimento e ser registrada antes de abrir o conjunto confirmatório. O valor 0,8 é uma expectativa inicial, não um resultado conhecido do piloto.

== Banco de candidatos

A configuração definitiva prevê 20 tentativas para cada tarefa confirmatória. Com 132 tarefas, isso corresponde a 2.640 trajetórias potenciais. As ativações e estatísticas são extraídas desse mesmo banco. Gerar bancos distintos para cada método acrescentaria variação de candidatos à comparação dos métodos.

Cada tentativa tem seed própria, derivada de uma função hash documentada do protocolo, versão do benchmark, tarefa e índice da tentativa, convertida para o intervalo aceito pelo gerador. Deve-se verificar colisões na conversão. A seed é reinicializada antes de cada tentativa, de modo que interromper uma amostra anterior não altere o fluxo aleatório da seguinte.

A independência estatística entre amostras é uma aproximação do desenho de seeds e do decodificador. A reprodutibilidade bit a bit depende também do hardware e das bibliotecas. O banco arquivado, com hashes, é a referência empírica; uma seed isolada não substitui a preservação dos candidatos.

== Contrato dos registros

Cada registro contém identificação de tarefa e tentativa, seed, prompt exato, IDs de tokens, completion bruta, código extraído, critério de término, rótulo funcional, classe de erro, custos e referências para as features. O manifesto global contém modelo, tokenizer, benchmark, verificador, versões, configuração, commit e hashes.

#grade(
 [*Grupo de dados*], [*Campos essenciais*],
 [Identidade], [`run_id`, `task_id`, `attempt_id`, `seed`, versões do protocolo e do banco.],
 [Geração], [Prompt e template, tokens, texto bruto, parâmetros, motivo de parada, comprimento e tempo.],
 [Verificação], [Código extraído, aprovação, tipo de falha, versão da suíte e tempo de execução.],
 [Preditores], [Camada, checkpoint, ponto de hook, risco ativo, NLL, entropia e caminho/hash das ativações.],
 [Proveniência], [Commit, revisões imutáveis, hardware, precisão, dependências e integridade dos arquivos.],
)

Tentativas duplicadas em conteúdo são preservadas se resultam da amostragem prevista. Não se elimina uma solução porque ela parece repetitiva, trivial ou longa. Remoção por corrupção deve conservar o registro da causa e os totais antes e depois. Falhas do programa gerado e falhas da infraestrutura são categorias distintas.

== Verificação segura e estável

O código gerado é executado em ambiente isolado, sem credenciais, sem acesso de rede e com limites de tempo, memória e processos. Essas condições protegem a máquina experimental e tornam a medição reproduzível. Reexecuções usam a mesma política de limites. Se o ambiente produz resultados flutuantes, essa instabilidade deve ser diagnosticada antes de usá-la como variável de aprendizado.

O parser de completion para código é determinístico e definido antes da avaliação. Regras para remover cercas Markdown, tratar prefixos e extrair funções não podem variar após a inspeção de exemplos que falharam. Alterar o parser muda o experimento e exige versionar os rótulos.

= Preditores precoces e modelos estatísticos

== Checkpoints e conjunto sob risco

São planejados os checkpoints 0, 8, 16, 32 e 64 tokens; a representação final é uma referência tardia. O endpoint confirmatório de especificidade é 32 tokens. As outras posições descrevem a evolução temporal e participam da escolha interna da política operacional.

A ordem da decisão é: primeiro verificar se a geração terminou; se terminou, encaminhar ao verificador; caso contrário, avaliar a regra de interrupção. Assim, para uma decisão depois de $t$ tokens, o conjunto sob risco é

$ cal(R)_t={(i,j):T_(i j)>t}. $

O uso da desigualdade estrita evita classificar como abortada uma completion que terminou exatamente no checkpoint. Um estado existente em $T=t$ pode ser usado em uma análise descritiva separada, mas já não corresponde à mesma decisão de continuação. Gerações curtas não são preenchidas com zeros nem recebem uma ativação futura fictícia.

As curvas em diferentes checkpoints condicionam em conjuntos diferentes. Deve-se reportar, em cada ponto, candidatos ativos, tarefas representadas, tarefas mistas e prevalência. Uma análise descritiva adicional sobre o conjunto comum que permanece ativo até 64 tokens ajuda a distinguir mudança de população e mudança de informação; ela representa essa população restrita, e não todo o benchmark.

== Camadas e convenção do estado

As quatro profundidades candidatas correspondem aos blocos mais próximos de 25%, 50%, 75% e 100% da profundidade. Os índices são calculados uma vez, registrados e deduplicados se necessário. O manifesto especifica se o vetor corresponde à saída residual do bloco antes ou depois de normalização. O embedding inicial não conta como bloco Transformer.

A captura deve reproduzir o estado disponível na inferência incremental. Salvar todas as ativações de todos os tokens não é requisito: podem-se preservar apenas os vetores selecionados, tokens e metadados, reduzindo o armazenamento. A extração completa offline serve para construir os dados de pesquisa; a política online deverá obter seus estados sem executar o futuro que pretende economizar.

== Estatísticas observáveis

Sejam $a_(u,v)$ os logits do token $v$ no passo $u$. A distribuição bruta é $p_u (v)="softmax"(a_u)_v$. O decodificador aplica temperatura, seleção do núcleo e renormalização para produzir $q_u$. A probabilidade bruta e a efetivamente amostrada são objetos diferentes.

Para a distribuição de amostragem, definem-se

$ "NLL"_u=-log q_u (Y_u), quad "Ent"_u=-sum_(v:q_u (v)>0) q_u (v) log q_u (v). $

Até $t$, as características principais incluem média de NLL, NLL do último token, média de entropia, entropia máxima e média da diferença entre os dois maiores logits. Para esta última, usam-se os logits finitos antes do top-p, após a escala de temperatura, evitando uma margem infinita quando o filtro deixa apenas um token. A base do logaritmo é natural e a unidade é nat.

A logprobabilidade do prefixo é $sum_(u=1)^t log q_u (Y_u)$. Em um checkpoint fixo, ela é uma transformação linear da NLL média; incluir ambas não cria uma fonte independente de informação. O comprimento final, o tempo total futuro e o resultado dos testes nunca entram em features precoces. O número de tokens $t$ também é constante em uma análise de checkpoint fixo.

Comprimento do prompt e da assinatura são controles de tarefa. Se a assinatura não puder ser extraída com regra confiável, seu indicador de ausência é registrado. Estatísticas equivalentes da distribuição bruta podem ser uma análise de sensibilidade previamente definida, porque entropia após top-p depende do filtro de amostragem.

== Baselines e modelo principal

#grade(
 [*Modelo*], [*Informação utilizada*],
 [B0 — Prior], [Prevalência do treino, sem rótulos de outras tentativas da tarefa de teste.],
 [B1 — Prompt], [Comprimento e atributos observáveis do prompt.],
 [P — Estado inicial], [Representação `prompt_last`; baseline de variação entre tarefas.],
 [B — Observáveis], [Atributos do prompt e estatísticas de confiança do prefixo.],
 [H — Latente], [Estado interno de uma camada e posição escolhidas no treino.],
 [H+B — Combinado], [Estado interno concatenado aos mesmos observáveis usados por B.],
)

B, H e H+B utilizam o mesmo tipo de classificador principal: regressão logística com regularização L2. O número de hiperparâmetros e o critério de seleção são controlados. B deve receber seu próprio ajuste de regularização no treino, para que H+B não vença um baseline deliberadamente subajustado.

== Probe linear e função objetivo

Para um vetor de features $a$, o probe define

$ z=b+w^top a, quad hat(p)=sigma(z)=1/(1+exp(-z)). $

Se a tarefa $i$ contribui com $n_i$ tentativas elegíveis, cada tentativa recebe peso proporcional a $1/n_i$. Uma forma da função objetivo é

$ J(b,w)=1/m_"tr" sum_(i in cal(I)_"tr") 1/n_i sum_j ell(S_(i j),sigma(b+w^top a_(i j))) + lambda/2 norm(w)_2^2, $

com $ell(s,p)=-s log p-(1-s) log(1-p)$. O intercepto não é penalizado. A relação exata entre $lambda$ e o parâmetro `C` depende da normalização usada pela implementação; a configuração preserva a definição da biblioteca, sem assumir que qualquer convenção equivale numericamente a $lambda=1/C$.

Padronização é ajustada apenas no conjunto de treinamento de cada split. Uma direção por diferença de médias,

$ v=hat(mu)_1^"tr"-hat(mu)_0^"tr", $

pode ser um diagnóstico geométrico secundário. A direção e sua orientação são construídas exclusivamente com rótulos de treino. Separação visual em projeções escolhidas depois de olhar todo o banco não constitui evidência confirmatória.

== Calibração e interpretação dos scores

Uma transformação logística de calibração, $hat(p)_"cal"=sigma(a z+b)$, pode ser ajustada com previsões de desenvolvimento produzidas fora da amostra. Calibração trata a relação entre score e frequência observada de sucesso; é distinta de discriminação @guo2017. Uma transformação estritamente crescente preserva a ordenação e a AUC.

A interrupção por limiar necessita principalmente de uma boa ordenação e de um limiar validado. Probabilidades bem calibradas facilitam interpretação e comparações, mas não são condição matemática para obter economia. Será reportado o desempenho probabilístico, sem apresentar calibração como garantia de retenção de corretos em tarefas novas.

= Validação, inferência estatística e controles

== Separação por tarefa e avaliação aninhada

As tentativas de um problema compartilham especificação e estrutura de solução. Dividi-las aleatoriamente entre treino e teste permite que o treinamento se beneficie de características do próprio problema avaliado. A separação exigida é

$ cal(I)_"tr" ∩ cal(I)_"te" = emptyset, $

em que os conjuntos contêm identificadores de tarefas. As mesmas tarefas pertencem ao mesmo fold em todos os checkpoints e métodos. Não se refaz o particionamento para obter uma distribuição mais favorável de rótulos depois dos resultados.

Utiliza-se avaliação externa com cinco folds agrupados e validação interna com três folds agrupados dentro de cada treino externo. A camada e `C` são selecionados internamente. O risco de sobreajuste ao critério de seleção motiva a separação entre seleção e avaliação @cawley2010. Todos os passos dependentes dos dados, inclusive padronização e construção de direções, são repetidos no treino correto.

O conjunto candidato de regularização é $C in {0.001,0.01,0.1,1,10}$. No endpoint científico de 32 tokens, a posição não é selecionada. Para cada modelo, o critério interno principal é log loss médio por tarefa. Dentro de uma faixa de um erro-padrão do melhor valor interno, usa-se maior regularização e depois camada mais rasa. O erro-padrão para essa regra é uma heurística de seleção, não um intervalo de confiança do resultado final.

O objeto avaliado pela validação externa é o procedimento que escolhe, treina e aplica um modelo usando apenas tarefas de treino. As camadas selecionadas podem variar entre folds. Não se deve apresentar a melhor camada encontrada posteriormente em todo o banco como se tivesse sido definida antecipadamente.

== Predições fora da amostra e calibração interna

Cada candidato confirmatório recebe uma previsão de um modelo que não treinou em sua tarefa. A concatenação dessas previsões permite calcular métricas sobre o conjunto externo completo. As escolhas continuam restritas ao interior de cada treino externo.

Para calibrar e selecionar a política, obtêm-se scores internos fora da amostra. Quando a seleção de hiperparâmetros participa desse processo, ela também precisa ocorrer sem o fold interno cujos scores estão sendo produzidos, ou os scores devem ser tratados apenas como dados de desenvolvimento com viés de seleção. Uma implementação com cross-fitting de desenvolvimento registra essa distinção explicitamente.

Calibrador e limiar podem ser ajustados nos dados de desenvolvimento desde que todo o sistema seja avaliado depois em tarefas externas intocadas. O recall obtido nesse conjunto de ajuste é uma meta empírica, sujeita a otimismo por seleção. Para reivindicar uma certificação de recall, seria necessário um conjunto independente de certificação e um procedimento de inferência adequado à dependência entre tentativas; o presente desenho não faz essa reivindicação.

== Métricas probabilísticas e de ranking

As métricas principais de probabilidade são log loss e Brier, com peso igual por tarefa:

$ "LogLoss"=1/m sum_i 1/n_i sum_j ell(S_(i j),hat(p)_(i j)), $
$ "Brier"=1/m sum_i 1/n_i sum_j (hat(p)_(i j)-S_(i j))^2. $

Para estabilidade numérica, probabilidades usadas em logaritmos são limitadas a $[epsilon,1-epsilon]$, com $epsilon$ documentado e idêntico entre métodos. A AUC agregada é reportada como diagnóstico entre candidatos, acompanhada da WT-AUC. O resumo de precisão-revocação utiliza average precision (AP), com implementação declarada; a denominação genérica PR-AUC não deve esconder a diferença entre AP e integração trapezoidal.

Curvas de calibração descrevem a relação entre probabilidade prevista e frequência observada. O número de bins e a regra de particionamento são fixados. ECE é secundário porque depende desses bins e pode ser instável na amostra pequena. A análise não se limita a acurácia com limiar 0,5, que pode ser pouco informativa para a política de alta retenção.

== Bootstrap por tarefa e alcance dos intervalos

Para estimar incerteza, reamostram-se tarefas inteiras com reposição, preservando todas as suas tentativas, scores e resultados de políticas. Usam-se os mesmos índices reamostrados para comparar os métodos. Em cada réplica, recomputam-se as métricas e as diferenças. O número planejado é 2.000 réplicas com seed fixada, produzindo intervalos percentílicos de 95%.

Na WT-AUC, reamostrar apenas as tarefas mistas estima a incerteza condicional a esse subconjunto; nas métricas de resolução e custo, reamostra-se todo o conjunto confirmatório. A opção e seu denominador serão explicitados. Réplicas sem classes necessárias não recebem valor zero artificialmente; são contadas como indefinidas.

O bootstrap dos scores externos já calculados trata os ajustes como fixos. Como os modelos dos folds compartilham dados de treino, esses intervalos são aproximações para a variabilidade entre tarefas e não capturam integralmente a instabilidade de retreinar todo o procedimento. Uma análise adicional com repetição agrupada do processo pode avaliar essa componente, mas precisa ser dimensionada antes de olhar os resultados. O texto final deverá dizer se a incerteza reportada é condicional aos modelos ajustados.

== Permutações: hipótese nula e validade

Permutações avaliam se o alinhamento entre features e rótulos é compatível com uma hipótese nula de intercambialidade @ojala2010. Embaralhar rótulos dentro da tarefa mantém seu número de acertos e destrói o alinhamento entre prefixos e desfechos. Esse é um controle relevante para sinal intratarefa.

Entretanto, tal embaralhamento também destrói a relação entre os observáveis $B$ e o resultado. Por isso, ele não é um teste específico da hipótese “hidden states não acrescentam nada a B”. Aplicar esse procedimento ao delta H+B menos B e chamá-lo de teste de poder incremental seria uma interpretação incorreta. A análise incremental principal usa deltas pareados fora da amostra, intervalos por tarefa e perdas probabilísticas.

Há também um cuidado com os folds. Permutar todos os rótulos depois de concatenar scores OOF e tratar todos esses scores como externos a todos os rótulos ignora que o rótulo de uma tarefa pode ter sido usado para treinar scores de outros folds. Uma alternativa computacionalmente simples é o teste por fold externo:

+ Fixar o modelo e as previsões do fold $k$, sem ajustes usando seus rótulos.
+ Permutar os rótulos apenas dentro de cada tarefa ativa desse fold.
+ Calcular a WT-AUC de H+B em cada permutação, usando 10.000 repetições.
+ Obter $p_k=(1+sum_b bold(1){M_b>=M_"obs"})/(10001)$, com $M$ igual à WT-AUC.
+ Combinar conservadoramente por $p_"global"=min(1,5 min_k p_k)$.

A correção de Bonferroni permanece válida sem independência entre os cinco p-valores, desde que cada teste individual seja válido. Folds sem tarefas mistas recebem $p_k=1$. Esse teste verifica ausência global de associação intratarefa no sentido definido; a combinação pode ter potência baixa. Uma alternativa é permutar rótulos dentro das tarefas e repetir integralmente treinamento e seleção a cada permutação, com custo computacional maior. A escolha deve ser congelada, sem selecionar a estratégia que produza menor p-valor.

A validade ainda pressupõe que, sob a hipótese nula, os rótulos sejam intercambiáveis entre tentativas elegíveis do mesmo problema. Amostragem com configurações distintas ou tentativas dependentes de feedback do verificador exigiria restringir os blocos de permutação ou redesenhar o teste. Este projeto usa tentativas sem realimentação entre gerações.

== Efeitos fixos como análise complementar

Uma regressão com intercepto por tarefa formaliza o controle da informação constante no prompt:

$ "logit" P(S_(i j)=1) = alpha_i + beta_B^top B_(i j t)+beta_z z_(i j t). $

A estimação logística condicional pode eliminar os interceptos ao condicionar no número de sucessos de cada tarefa. Tarefas sem variação de rótulo não identificam $beta_z$. Colunas constantes dentro da tarefa, como comprimento do prompt, são absorvidas pelo efeito fixo e não precisam ser estimadas separadamente.

Essa regressão é associativa e complementar. Interceptos estimados com os rótulos das tarefas externas não constituem um preditor válido para tarefas novas. Portanto, não se usa sua log loss ajustada como melhoria fora da amostra. A comparação preditiva vem dos modelos B e H+B avaliados externamente. A incerteza de um coeficiente calculado sobre scores aprendidos deve reconhecer que esses scores também foram estimados; um erro-padrão convencional não representa automaticamente toda a cadeia de aprendizado.

== Controles negativos e diagnósticos

O controle determinístico de `prompt_last` deve ter WT-AUC igual a 0,5 quando há empates exatos. Uma direção aleatória normalizada avalia a facilidade de obter associação aparente em alta dimensão. Um probe treinado com rótulos permutados dentro das tarefas do treino verifica se a separação depende do alinhamento real entre representação e resultado.

A direção invertida não é um controle independente contra associação: para um score sem outras mudanças, $"AUC"(-z)=1-"AUC"(z)$ com a convenção de empates. Ela serve para verificar sinal e implementação. Também não se permite usar o comprimento final como controle de uma política precoce. Indicadores de formatação já presentes no prefixo podem ser diagnósticos de artefatos; seu ajuste deve respeitar os splits.

Um desempenho não significativo de um controle não demonstra automaticamente a validade do modelo principal. Os controles se somam a uma avaliação explícita de vazamento, amostra, hipótese nula e tamanho de efeito.

== Multiplicidade e critérios de evidência

A posição de 32 tokens e a comparação H+B contra B definem o endpoint principal. O resultado positivo de especificidade incremental requer $Delta_W>0$, intervalo pareado de 95% inteiramente acima de zero e $Delta_L<0$ fora da amostra. O intervalo de $Delta_L$ também é reportado; se ele incluir zero, a evidência probabilística é qualificada como incerta. O teste de permutação apoia a análise de existência de sinal intratarefa, sem funcionar como teste do ganho condicional.

Curvas por posição, comparações de camadas, tipos de erro, direções geométricas e subgrupos são análises secundárias. Se forem submetidas a testes formais múltiplos, a família de comparações e a correção serão declaradas. Não se escolhe retrospectivamente outro checkpoint para substituir um resultado desfavorável em 32 tokens.

Não rejeitar uma hipótese nula não demonstra equivalência. Um intervalo amplo pode conter tanto ausência de efeito quanto um efeito relevante. Com poucas tarefas mistas, a conclusão apropriada pode ser “inconclusivo neste orçamento”, e não “o modelo não contém o sinal”.

= Política de interrupção precoce e replay

== Regra de decisão e casos de término

A política utiliza um único checkpoint $t^*$ por ajuste, evitando a multiplicação de decisões dentro da mesma tentativa. Se o candidato permanece ativo nesse ponto e seu score é inferior ao limiar $q$, a geração é interrompida. Caso contrário, continua normalmente até terminar. Não há steering nem alteração da distribuição dos tokens nos candidatos mantidos.

O algoritmo por tarefa é:

+ Iniciar a tentativa seguinte com sua seed pré-atribuída.
+ Gerar até o término natural, o checkpoint ou o limite, conforme o que ocorrer primeiro.
+ Se houve término, extrair o programa, verificar e parar a tarefa se houver sucesso.
+ Se a geração segue ativa no checkpoint, calcular os preditores e o score.
+ Interromper se o score for menor que $q$; caso contrário, completar e verificar.
+ Repetir até um sucesso verificado ou até atingir $N$ tentativas iniciadas.

Uma tentativa abortada consome orçamento de tentativa. Um candidato que termina exatamente em $t^*$ vai ao verificador antes da decisão de aborto. A regra com $q$ abaixo de todos os scores representa “não interromper” e deve sempre estar incluída na seleção.

== Seleção do checkpoint e do limiar

Dentro do treino externo, avaliam-se $t in {8,16,32,64}$ e limiares derivados exclusivamente dos scores de desenvolvimento. O modelo de cada posição é ajustado sem usar o teste externo. A métrica de retenção relevante para o gate é

$ hat(r)_"ativo" (t,q)=
(sum_(i,j) S_(i j) bold(1){T_(i j)>t} bold(1){hat(p)_(i j t)>=q}) /
(sum_(i,j) S_(i j) bold(1){T_(i j)>t}). $

Essa taxa evita que muitos sucessos curtos, nunca expostos ao gate, escondam perdas entre candidatos efetivamente avaliados. Reporta-se também a retenção global, contando sucessos que terminaram antes do checkpoint como mantidos. Se não há sucessos ativos suficientes para avaliar uma configuração, ela não é calibrável e a opção conservadora é não interromper.

A meta de desenvolvimento é $hat(r)_"ativo">=0.97$. Todos os pares checkpoint-limiar que atendem à meta são avaliados por replay de custo médio por tarefa. Escolhe-se o menor custo; empates numéricos são resolvidos por maior resolução no desenvolvimento, checkpoint mais cedo e menor limiar. Não se escolhe automaticamente o maior limiar admissível, pois abortar sucessos pode aumentar o custo até solução.

Para evitar impor um limiar pior que o baseline, a opção sem aborto participa da mesma seleção. O protocolo registra uma tolerância numérica para empates e uma grade finita de limiares, incluindo os valores nos quais a decisão muda. A política e seu calibrador ficam congelados antes da avaliação externa.

== Meta empírica de recall e generalização

A taxa de 97% é uma meta de ajuste. O recall no teste deve ser medido, e pode ser menor. Dependência dentro da tarefa, seleção do melhor limiar e mudança de distribuição impedem atribuir automaticamente uma garantia binomial simples à taxa agregada.

#cite(<ruan2026>, form: "prose") distinguem ajuste e certificação independente em seu contexto de agentes. Esta proposta adota apenas uma restrição empírica de seleção e uma avaliação externa; não reproduz a garantia formal daquele método. Além disso, retenção de trajetórias corretas não é igual a preservação da resolução das tarefas. Perder a única solução de um problema tem efeito diferente de perder uma entre várias soluções corretas.

== Replay com candidatos pareados

O banco guarda a completion completa de cada tentativa. Para simular aborto, o replay consulta apenas features disponíveis no checkpoint e usa o resultado completo exclusivamente para avaliar o contrafactual de retenção: saber se o candidato descartado passaria. A política não acessa esse rótulo para decidir.

Defina $b_(i j)=bold(1){T_(i j)>t^*,hat(p)_(i j t^*)<q}$. O custo de uma tentativa iniciada no replay é

$ d_(i j)^pi = b_(i j)t^*+(1-b_(i j))T_(i j). $

Se $K_i^pi$ é a primeira tentativa não abortada que passa, então

$ C_i^pi (N)=sum_(j=1)^N d_(i j)^pi bold(1){K_i^pi>=j}, $
$ A_i^pi (N)=bold(1){K_i^pi<=N}. $

É essencial somar apenas tentativas efetivamente iniciadas. O rótulo de um candidato abortado pode ser usado para calcular false-abort na análise, mas não para parar o replay como se a tarefa tivesse sido resolvida.

A equivalência com a execução online exige que abortar uma tentativa não modifique a identidade das seguintes. Seeds independentes, ordem fixa e configuração determinística fornecem essa propriedade no desenho. Batching e diferenças numéricas podem impedir igualdade bit a bit; por isso, uma execução online pequena deve conferir decisões, contagem de tokens e comportamento de término.

== Comparadores operacionais

O experimento utiliza o mesmo banco e a mesma ordem para quatro famílias: geração fixa de N candidatos completos; geração sequencial com verificador; interrupção com observáveis B; e interrupção com H+B. Os orçamentos reportados são 1, 5, 10 e 20, com $N=20$ como comparação principal.

Cada política adaptativa recebe o mesmo espaço de checkpoints, método de calibração e regra de seleção. Uma ablação secundária pode fixar o mesmo checkpoint para B e H+B, isolando a contribuição do score. A comparação principal entre procedimentos permite que cada um selecione sua configuração dentro do mesmo orçamento de desenvolvimento.

O baseline fixo e o baseline com verificador têm a mesma cobertura potencial no mesmo conjunto de candidatos, mas custos distintos quando há sucessos precoces. O comparador decisivo da economia é o baseline com verificador, porque ele já evita gerar tentativas depois de resolver a tarefa.

== Não inferioridade e economia

Definam-se as diferenças médias pareadas

$ Delta_A=1/m sum_i (A_i^pi-A_i^V), quad Delta_C=1/m sum_i (C_i^pi-C_i^V). $

A margem planejada de não inferioridade é 0,03 em proporção, isto é, 3 pontos percentuais. Exige-se que o limite inferior do intervalo de 95% de $Delta_A$ seja maior que $-0.03$, e que o limite superior do intervalo de $Delta_C$ seja menor que zero. Usar um intervalo bilateral de 95% para essas condições é uma escolha conservadora em relação a um teste unilateral de 5%.

A economia relativa é $1-overline(C)^pi/overline(C)^V$, com intervalo pareado. A margem de resolução não deriva automaticamente do recall de 97%; são critérios distintos. Com aproximadamente 132 tarefas, a perda de uma tarefa equivale a cerca de 0,76 ponto percentual, o que limita a resolução estatística de uma margem estreita. A discussão de potência deve considerar tarefas e discordâncias pareadas, não apenas o número total de trajetórias.

A comparação contra a política observável usa as mesmas métricas. Economizar em relação ao verificador, mas não superar B, sustenta o valor da interrupção naquele regime, sem demonstrar que os estados internos foram necessários para o ganho.

== Medição online e amortização

A execução online mede tempo após aquecimento, sincronização apropriada da GPU e condições comparáveis de carga e batch. Reporta geração, extração/score, verificação e total. A política não deve ser creditada por economia de FLOPs ou energia sem medição correspondente.

O custo de pesquisa inclui gerar o banco completo, extrair ativações, treinar e calibrar. Esses custos não pertencem ao custo marginal por tarefa da política já treinada, mas precisam ser informados. Se o custo de preparação for $F$ e a economia média por uso for $s>0$ na mesma unidade, o ponto de amortização idealizado é $F/s$ utilizações. Sem estimativas compatíveis, não se afirma retorno econômico total.

= Plano dos experimentos e critérios de conclusão

== Experimento 1: previsibilidade temporal

O primeiro experimento produz scores externos em cada checkpoint e compara B0, B1, P, B, H e H+B nos conjuntos em que suas features estão disponíveis. No ponto inicial, não existem estatísticas de completion; os comparadores apropriados são o prior, observáveis do prompt e seu estado interno. Nos pontos seguintes, os modelos são avaliados sobre as mesmas trajetórias elegíveis de cada ponto.

A figura temporal principal apresenta AUC agregada e WT-AUC em painéis separados, com os respectivos intervalos. Log loss, Brier, AP e calibração completam a análise. Os desempenhos no último token são referências tardias, não limites superiores matemáticos: uma representação final ou um probe linear pode inclusive desempenhar pior que uma representação anterior.

A interpretação observa se o ganho aparece antes de grande parte do programa ser produzida e quantas trajetórias ainda estão ativas. Um score forte em 64 tokens pode ter pouca utilidade se quase todos os candidatos já terminam nessa posição. Da mesma forma, AUC alta antes de gerar qualquer token pode refletir sobretudo heterogeneidade entre tarefas.

== Experimento 2: especificidade e ganho incremental

O experimento central utiliza o checkpoint fixo de 32 tokens. São avaliados WT-AUC de B, H e H+B, delta H+B menos B, perdas probabilísticas e distribuição dos deltas por tarefa. As previsões são produzidas por validação externa agrupada, e camada e regularização são escolhidas apenas internamente.

O teste de permutação intratarefa examina a presença de associação entre score e desfecho. O bootstrap pareado e a comparação de perdas examinam a vantagem operacional da classe H+B sobre B. A regressão com efeitos fixos é secundária e recebe interpretação associativa.

Exemplos individuais podem ilustrar o fenômeno, mas devem ser escolhidos por regra explícita: por exemplo, tarefas próximas da mediana do ganho, acompanhadas de exemplos de falha. Selecionar apenas as trajetórias visualmente mais impressionantes criaria uma narrativa enviesada. A conclusão central depende das estatísticas do conjunto, não de um caso.

== Experimento 3: utilidade operacional

O terceiro experimento seleciona a política nos dados internos e avalia uma única versão congelada por fold externo. O replay produz a resolução, os custos e a retenção externa para cada comparador. Uma curva de sensibilidade pode avaliar uma grade de metas de recall previamente definida no desenvolvimento; cada ponto externo continua associado a uma escolha feita sem os rótulos externos.

As diferenças pareadas sustentam ou não a não inferioridade e a economia. A decisão não é baseada exclusivamente no ponto com maior economia observado na curva externa. Se a configuração selecionada não melhora o custo, o resultado permanece negativo para a política proposta, ainda que alguma configuração escolhida depois pareça favorável.

Uma validação online em tarefas predefinidas compara decisões e custos reais. O rótulo “economia de tokens em replay” deve permanecer até existir medição online suficiente para afirmar redução de latência. A ausência de aceleração apesar de economia de tokens é um resultado informativo sobre overhead e implementação.

== Robustez em MBPP+

A robustez externa utiliza MBPP+ na implementação e versão fixadas do EvalPlus. O teste principal é de transferência: após concluir a análise principal, treinar uma versão final com os dados de desenvolvimento permitidos de HumanEval+, escolher hiperparâmetros e calibrador sem MBPP+ e congelar tudo antes de avaliar o benchmark externo.

A mesma variante do modelo, convenção de extração e construção das features é mantida. O template pode precisar representar a estrutura de uma tarefa MBPP+, mas a transformação deve ser definida sem observar seus resultados. Se houver subconjunto por restrição de custo, seus IDs são escolhidos deterministicamente antes da geração ou da inspeção de rótulos.

Um experimento que retreina probes ou recalibra limiares com rótulos de MBPP+ responde a outra pergunta: replicação do fenômeno dentro de outro benchmark. Ele pode ser útil, mas deve ser nomeado separadamente. Não pode substituir silenciosamente uma transferência que falhou. Em particular, a meta de recall ajustada em HumanEval+ não é garantida após mudança de distribuição.

== Critérios de encerramento e resultados inconclusivos

O trabalho está completo quando as perguntas são respondidas com os dados efetivamente disponíveis, mesmo que a resposta seja negativa ou inconclusiva. Não depende de confirmar todas as hipóteses.

#grade(
 [*Padrão observado*], [*Interpretação permitida*],
 [Sinal precoce não detectado], [Os preditores e o orçamento testados não fornecem evidência robusta; não se conclui inexistência absoluta de informação.],
 [AUC agregada alta; WT-AUC baixa], [O desempenho pode ser dominado por variação entre tarefas. Se os intervalos forem amplos, a especificidade permanece inconclusiva.],
 [WT-AUC alta; sem ganho sobre B], [Existe discriminação intratarefa, mas não foi demonstrada vantagem incremental dos estados sobre os observáveis escolhidos.],
 [Ganho incremental; sem economia], [O sinal é útil para previsão, mas insuficiente ou tardio para a política operacional avaliada.],
 [Ganho incremental e economia], [Evidência favorável ao encadeamento Q1–Q3 nas condições testadas, com a resolução e a incerteza reportadas.],
)

Uma taxa baixa de tarefas mistas reduz a potência. Nesse caso, não se expande repetidamente o banco até obter significância sem uma regra de parada previamente definida. Um novo orçamento é uma emenda prospectiva ou uma nova análise exploratória.

= Resultados: estrutura para incorporação dos experimentos

#pendente[Este capítulo define a apresentação dos resultados. Não contém medições do modelo. Os campos “A medir” devem ser substituídos por saídas reproduzíveis dos scripts, com versão, denominadores e intervalos. Os números de planejamento mencionados anteriormente não são tamanhos de amostra já observados.]

== Execução efetiva, amostra e qualidade dos dados

A primeira seção empírica deverá registrar configuração final do piloto, versões, data de execução, hardware e banco definitivo. Deve conciliar tarefas planejadas, efetivamente geradas, verificadas, incluídas e excluídas. Contagens de interrupções de infraestrutura não serão misturadas a programas incorretos.

#figure(
 table(columns: (2fr, 1fr, 1.8fr),
  [*Indicador*], [*Valor*], [*Denominador/critério*],
  [Tarefas de desenvolvimento], [A medir], [IDs reservados e exposição prévia],
  [Tarefas confirmatórias], [A medir], [Após exclusões documentadas],
  [Trajetórias verificadas], [A medir], [Com rótulo válido],
  [Aprovações funcionais], [A medir], [Suíte HumanEval+],
  [Tarefas mistas], [A medir], [Banco completo e cada risco],
  [Truncamentos], [A medir], [Critério de término],
  [Tokens por tentativa], [A medir], [Média, mediana e quantis],
 ),
 caption: [Composição do banco e qualidade dos dados. Tabela preparada para preenchimento.],
)

Uma tabela ou painel adicional apresentará os conjuntos ativos em 8, 16, 32 e 64 tokens: número de trajetórias, prevalência, tarefas e tarefas mistas. A discussão deverá explicar se a população muda substancialmente ao longo dos checkpoints e se isso afeta a comparação temporal.

== Previsibilidade ao longo da geração

A Figura 1 empírica deverá resumir o desenho executado: banco de tarefas, múltiplas tentativas, extração de prefixos, verificação e validação agrupada. A Figura 2 mostrará as curvas temporais de previsão, distinguindo AUC agregada e WT-AUC. Seus eixos e legendas devem indicar se os intervalos condicionam nos modelos ajustados e qual conjunto sob risco é usado.

O texto deverá responder quando aparece ganho sobre B, sua intensidade e estabilidade entre folds. Resultados de `prompt_last` serão interpretados como sinal entre tarefas. O desempenho final será explicitamente identificado como referência tardia e inacessível à decisão precoce.

#figure(
 table(columns: (1fr, 1fr, 1fr, 1fr, 1.2fr),
  [*Posição*], [*AUC B*], [*AUC H*], [*AUC H+B*], [*WT-AUC H+B*],
  [0], [Não se aplica], [A medir], [A medir], [Controle: 0,5],
  [8], [A medir], [A medir], [A medir], [A medir],
  [16], [A medir], [A medir], [A medir], [A medir],
  [32], [A medir], [A medir], [A medir], [A medir],
  [64], [A medir], [A medir], [A medir], [A medir],
 ),
 caption: [Estrutura da comparação temporal. No ponto zero, H designa o estado inicial e H+B apenas pode incorporar observáveis do prompt. Valores deverão incluir intervalos e denominadores.],
)

== Discriminação intratarefa e valor incremental

Esta é a seção central do capítulo. Deve apresentar WT-AUC dos três modelos no ponto de 32 tokens, o delta pareado, o intervalo e as perdas probabilísticas. A Figura 3 deverá mostrar tanto o efeito agregado quanto a heterogeneidade entre tarefas, sem transformar cada par de trajetórias em observação independente.

Os resultados de permutação serão reportados com hipótese nula, blocos e método de combinação. O texto deve distinguir “associação dentro da tarefa” e “ganho sobre B”. A regressão de efeitos fixos, se estimável, incluirá quantidade de tarefas contribuintes, convergência e a natureza complementar da análise.

#figure(
 table(columns: (2fr, 1.2fr, 1.8fr),
  [*Estimando em 32 tokens*], [*Estimativa*], [*Incerteza*],
  [WT-AUC de B], [A medir], [IC por tarefa],
  [WT-AUC de H], [A medir], [IC por tarefa],
  [WT-AUC de H+B], [A medir], [IC por tarefa],
  [$Delta_W$], [A medir], [IC pareado por tarefa],
  [$Delta_L$], [A medir], [IC pareado por tarefa],
  [Associação por permutação], [A medir], [P-valor e regra de combinação],
 ),
 caption: [Resultado científico principal. O p-valor de associação não substitui o intervalo do ganho incremental.],
)

Se o efeito estiver concentrado em poucas tarefas, o texto deverá reconhecer essa heterogeneidade. Se o intervalo incluir zero, deverá indicar quais magnitudes permanecem compatíveis com os dados. Resultados exploratórios em outros checkpoints não serão usados para reescrever retrospectivamente a hipótese principal.

== Política adaptativa e economia

A seção deve informar quais checkpoints e limiares foram selecionados nos folds internos e como o recall de desenvolvimento se compara ao externo. O false-abort será calculado tanto entre candidatos ativos corretos quanto no conjunto global de corretos, explicitando os denominadores.

#figure(
 table(columns: (1.8fr, 1fr, 1.3fr, 1.3fr),
  [*Política em N=20*], [*Resolução*], [*Tokens/tarefa*], [*Retenção ativa*],
  [N fixo], [A medir], [A medir], [Não aborta],
  [Verificador], [A medir], [A medir], [Não aborta],
  [Observáveis B], [A medir], [A medir], [A medir],
  [Latente H+B], [A medir], [A medir], [A medir],
 ),
 caption: [Comparação operacional principal sobre os mesmos candidatos. Relatar ICs e diferenças pareadas em tabela complementar.],
)

A Figura 4 deverá relacionar retenção de corretos e economia nas configurações previamente definidas; a Figura 5 mostrará custo versus resolução. O ponto selecionado em desenvolvimento deve ser identificado. Se uma política economiza apenas por ultrapassar a margem de perda de resolução, isso não satisfaz o objetivo principal.

Devem ser reportados também tokens por solução agregada, tentativas iniciadas, candidatos completos e tempos online. Um resultado favorável apenas em replay será descrito como tal. A contabilidade inclui as tarefas não resolvidas e todos os candidatos gerados até o limite.

== Transferência para MBPP+

Apresentar tamanho e regra do subconjunto, configuração congelada de origem e grau de mudança na distribuição de comprimentos e aprovações. Reportar as mesmas métricas principais com intervalos, distinguindo transferência sem ajuste de eventual replicação com ajuste local.

Uma tabela final pode comparar HumanEval+ e MBPP+ quanto a tarefas mistas, $Delta_W$, $Delta_L$, perda de resolução e economia. Ausência de ganho externo limita o alcance do resultado principal; não o apaga nem deve ser ocultada por uma recalibração não declarada.

== Falhas, auditoria e emendas

Listar falhas de infraestrutura, inconsistências corrigidas, alterações de parser, reextrações e mudanças de protocolo. Uma correção que altere rótulos ou features deve produzir uma nova versão do banco e a reexecução das análises dependentes.

Incluir o relatório dos testes de integridade, manifestos e localização das saídas de análise. O texto deve dizer quais experimentos foram prospectivos e quais foram motivados por observações posteriores. Essa distinção é mais informativa do que apresentar toda análise realizada como previamente planejada.

= Discussão e limitações

== Significado dos padrões encontrados

A discussão será preenchida a partir dos resultados, seguindo Q1–Q3. Uma melhora da AUC agregada que não se repete dentro das tarefas é compatível com reconhecimento de dificuldade. Uma melhora dentro da tarefa que não supera B indica que observáveis simples podem carregar a informação operacional relevante. Um ganho de H+B sem economia mostra que decodabilidade e utilidade de interrupção não são equivalentes.

Mesmo quando as três etapas forem favoráveis, a conclusão permanece condicionada ao modelo, ao regime de amostragem, à classe de probes, aos benchmarks e ao acesso a verificação funcional. A formulação “o modelo sabe que vai errar” deve ser substituída por uma descrição probabilística da evidência medida.

== Validade interna

Riscos principais incluem vazamento por tarefas repetidas, ajuste do processamento após inspeção do teste, uso de tokens futuros, descompasso entre geração e extração, escolhas múltiplas e calibração otimista. O desenho agrupado, o checkpoint confirmatório e os testes causais de prefixo reduzem esses riscos, mas precisam estar implementados e auditados, não apenas declarados.

O sinal pode refletir estilo, formatação, prefixos degenerados ou sintomas de erro sintático. Isso ainda pode ter utilidade para prever corretude funcional, mas não comprova compreensão semântica profunda do programa. Análises por classe de erro ajudam a delimitar a interpretação, desde que permaneçam secundárias.

== Validade de mensuração e do verificador

Passar em uma suíte finita não equivale a satisfazer a especificação em todo domínio de entradas. Aumento de cobertura de testes reduz algumas falhas de mensuração, sem eliminá-las. Timeouts dependem de hardware e orçamento, e o parser pode afetar o rótulo. A qualidade da variável resposta precisa ser discutida junto da qualidade do probe.

O acesso a testes completos para parar a busca favorece o cenário com verificador. A comparação mede eficiência condicional a esse acesso; não estabelece o mesmo ganho em cenários sem um oráculo funcional confiável.

== Validade externa e contaminação

Um único modelo pequeno e dois benchmarks de funções curtas não representam a variedade de repositórios, dependências, linguagens e tarefas de engenharia de software. Repetição de HumanEval durante desenvolvimento pode comprometer a novidade das tarefas para o pesquisador; presença de exemplos no pré-treinamento compromete inferências sobre problemas inéditos. Nenhum split de probes remove contaminação no pré-treinamento.

A mudança de benchmark pode alterar a prevalência, os prefixos e a confiabilidade do score. Isso afeta tanto AUC quanto calibração e retenção. Uma política de alta retenção precisa ser reavaliada quando o regime muda, sem atribuir garantias por analogia.

== Precisão estatística e escopo computacional

A amostra relevante para generalização contém tarefas, não milhares de observações independentes. Poucas tarefas mistas e poucos sucessos em certas posições tornam intervalos largos e limiares instáveis. Mais tentativas por tarefa estimam melhor sua variabilidade, mas não substituem automaticamente mais tarefas distintas.

Probes lineares podem falhar mesmo quando existe informação acessível a modelos não lineares. Portanto, um resultado negativo delimita a classe de leitura avaliada. Expansões para classificadores mais complexos precisam de orçamento e validação próprios; não são necessárias para produzir um resultado conclusivo sobre o desenho delimitado.

== Trabalhos futuros

Extensões naturais incluem avaliação com benchmarks menos sujeitos a contaminação, verificadores incompletos separados dos testes finais, reinvestimento da economia em tentativas adicionais e repetição em outros modelos. Uma comparação com representações textuais mais fortes poderia aprofundar a interpretação de ganho sobre observáveis simples.

Activation Steering constitui uma pergunta posterior: se uma direção preditiva também permite modificar a distribuição de sucesso. Essa pergunta exigiria intervenções, controles e desenho causal próprios. Não se infere a resposta a partir da existência de um probe preciso.

= Considerações finais

A fundamentação apresentada identifica uma lacuna experimental específica: diferenciar sinais que reconhecem a dificuldade do problema daqueles que distinguem o futuro de trajetórias estocásticas do mesmo problema. Para investigá-la, o trabalho combina geração verificável, representações de prefixos, baselines observáveis, validação por tarefa e uma política de interrupção avaliada por custo e resolução.

Nesta versão, a contribuição efetivamente concluída é a formulação do problema e do desenho de medição. Não foram estabelecidos ganho preditivo, economia ou generalização do modelo. A conclusão empírica deverá ser escrita depois dos experimentos, respondendo separadamente quando o sinal aparece, se acrescenta discriminação intratarefa e se produz benefício operacional.


#pagebreak()
#bibliography("referencias.bib", title: "Referências", style: "associacao-brasileira-de-normas-tecnicas")
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Ajustes necessários ao protocolo experimental

Este apêndice registra diferenças metodológicas em relação ao `RESEARCH_PROTOCOL.md` versão 0.1 produzido no planejamento anterior. O arquivo não foi alterado automaticamente. Os ajustes devem ser incorporados como uma nova versão antes da execução confirmatória, com data, justificativa e indicação dos resultados já conhecidos. Se resultados afetados já foram observados, a mudança não pode ser apresentada retroativamente como pré-registro.

#grade(
 [*Ponto*], [*Especificação adotada neste documento*],
 [Permutação e incremento], [Permutação intratarefa testa associação; ganho sobre B é avaliado por diferenças preditivas externas e intervalos pareados.],
 [Dependência entre folds], [Teste por fold com combinação conservadora, ou reexecução integral da pipeline sob permutação; evitar validade exata presumida ao permutar labels do pool OOF fixo.],
 [Término no checkpoint], [Checar término antes do aborto. Para continuação, risco ativo $T>t$.],
 [Escolha do limiar], [Avaliar o custo de todos os pares admissíveis; maior limiar não é necessariamente menor custo. Incluir a opção sem aborto.],
 [Retenção de corretos], [Meta empírica de 97% entre corretos ativos, com retenção global também reportada; não é garantia externa.],
 [Calibração], [Separar scores internos de ajuste e avaliação externa; reconhecer viés de seleção e custo de obter OOF honesto.],
 [Efeitos fixos], [Análise associativa complementar; não estimar interceptos com rótulos externos e chamar a perda ajustada de previsão fora da amostra.],
 [Truncamento], [Guardar motivo de término; definir corretude pela suíte. “Passa e não truncou” seria outro endpoint, explicitamente nomeado.],
 [Falhas de execução], [Separar timeout do programa de erro da infraestrutura de verificação.],
 [Controles e referência tardia], [Direção invertida é diagnóstico de sinal; estado final não é limite superior garantido.],
 [Piloto], [Quantificar ou remover a regra subjetiva de aumento material de truncamento; registrar tarefas previamente expostas.],
 [MBPP+], [Transferência congelada é distinta de replicação com retreino ou recalibração local.],
)

Os valores de planejamento preservados são: um modelo principal, HumanEval+, 32 tarefas de desenvolvimento, 20 tentativas definitivas por tarefa, posição confirmatória de 32 tokens, cinco folds externos, margem de 3 pontos percentuais e uma extensão em MBPP+. São decisões de desenho, não achados experimentais.

= Reprodutibilidade e critérios de integridade

== Organização dos artefatos

O repositório deve distinguir configuração, banco de candidatos, ativações, previsões, decisões da política e resultados agregados. Cada tabela e figura final deve ser produzida por script a partir de artefatos versionados. Arquivos grandes podem ficar fora do Git, mas seus esquemas, hashes, configurações e comandos permanecem registrados.

#grade(
 [*Artefato*], [*Conteúdo e finalidade*],
 [`RESEARCH_PROTOCOL.md`], [Perguntas, endpoints, splits, seleção, critérios de decisão e emendas.],
 [Manifesto da execução], [Commit, ambiente, revisões, seeds, tempos e hashes.],
 [Candidate Bank], [Candidatos completos, rótulos funcionais e dados de geração.],
 [Ativações e observáveis], [Features indexadas por tarefa, tentativa, camada e checkpoint.],
 [Predições externas], [Probabilidades, scores, fold, elegibilidade e identidade do ajuste.],
 [Replay], [Cada decisão, custo cobrado, parada, retenção e resolução por tarefa.],
 [`RESULTS.md`], [Tabelas consolidadas, comandos, limitações e localização dos artefatos.],
)

== Invariantes verificáveis

+ Nenhuma tarefa pode ocupar treino e teste do mesmo ajuste, inclusive na seleção interna.
+ Tarefas de desenvolvimento não aparecem no teste confirmatório.
+ Alterar ou ocultar labels externos não pode alterar parâmetros, calibração ou limiar daquele ajuste.
+ O estado no offset $t$ corresponde ao token correto e passa no teste de prefixo causal.
+ Padronização, direções e seleção utilizam exclusivamente o conjunto permitido.
+ Seeds são determinísticas, únicas na chave definida e independentes do término de tentativas anteriores.
+ O replay não consulta o rótulo futuro para decidir aborto, nem contabiliza candidatos depois da parada.
+ Geração concluída em $t$ é verificada, e não tratada como aborto.
+ Custo de candidato abortado é $t$ tokens gerados, com trabalho adicional de score medido separadamente.
+ A soma dos custos por decisão coincide com os totais por tarefa e por política.
+ Candidatos sem verificação válida são identificados, sem conversão silenciosa em erro do modelo.
+ Resultados de robustez preservam a distinção entre transferência congelada e ajuste local.

Parte dessas propriedades é verificável por testes de isolamento: por exemplo, modificar labels externos e confirmar que os objetos treinados permanecem idênticos. Nenhum teste automatizado consegue, por si só, demonstrar que o pesquisador não inspecionou resultados ao tomar decisões. Por isso, testes se complementam com arquivos de configuração congelados e um histórico de alterações.

== Ordem de execução

Primeiro, conciliar este documento com o protocolo e registrar suas emendas. Em seguida, implementar captura temporal e isolamento dos splits; executar o piloto; congelar geração e ambiente; construir o banco; gerar predições externas; concluir Q1 e Q2; selecionar e avaliar a política de Q3; executar a robustez; e preencher o capítulo de resultados a partir das saídas.

O protocolo serve para limitar decisões, e o manuscrito serve para justificar conceitos, hipóteses e interpretações. Ambos devem concordar sobre o que foi executado. Se divergirem, a versão final relata a divergência e sua data, em vez de ajustar silenciosamente a narrativa ao resultado.
