#import "../lib.typ": *

= Fundamentação matemática e computacional <chap-fundamentacao>

O capítulo desenvolve o argumento em três níveis. Primeiro define a informação disponível durante a geração e a corretude de uma resposta. Depois separa dificuldade da tarefa, discriminação intratarefa e capacidade do probe. Por fim, formula o custo de interromper e de iniciar novas tentativas. As mesmas definições serão usadas nas duas etapas experimentais; os procedimentos de ajuste e os estimandos primários de cada protocolo serão distinguidos na metodologia.

== Formulação probabilística e notação

=== Tarefas, candidatos e configuração

Seja $cal(X)$ o espaço de tarefas e seja $x_i in cal(X)$ o prompt da tarefa $i$, incluindo especificação, assinatura e template fixado. O modelo tem parâmetros $theta$ congelados. Uma configuração $c$ reúne temperatura, top-p, limite de tokens, regras de parada, precisão e versão do tokenizer. A tentativa $j$ produz a sequência aleatória $Y_(i j)=(Y_(i j 1), dots, Y_(i j T_(i j)))$, de comprimento $T_(i j)$.

Distinguem-se a distribuição original do modelo, $p_theta$, e a distribuição efetivamente usada pelo decodificador, $q_(theta,c)$. A lei da sequência amostrada é

$ q_(theta,c) (y | x_i) = product_(u=1)^(T) q_(theta,c) (y_u | x_i,y_(<u)). $

O comprimento é determinado por término natural ou limite computacional. A distribuição da sequência incorpora as regras de parada. Tratar todos os candidatos como tendo comprimento fixo esconderia uma dimensão essencial do custo e da elegibilidade para interrupção.

A notação omite um índice de modelo para simplificar as expressões. Na comparação entre Qwen e DeepSeek, as distribuições de geração, os vocabulários e as dimensões dos estados são próprios de cada gerador. O índice de tarefa pode ser compartilhado; tokens e coordenadas latentes não são tratados como intercambiáveis.

#grade(
 caption: [Notação do modelo de geração, corretude e custo sequencial.],
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

=== Corretude operacional

Um verificador $V$ recebe a tarefa e o programa extraído da completion. Sua saída define

$ S_(i j) = V(x_i,Y_(i j)) in {0,1}, quad pi_i = P(S_(i j)=1 | x_i,c). $

O verificador é mantido fixo em versão, testes, ambiente e limites de execução. Falhas de sintaxe, execução, asserção e extração são classificadas como insucesso. Um timeout do programa é uma falha sob o orçamento especificado. Uma interrupção da infraestrutura de avaliação é um erro de medição e deve ser reexecutada ou marcada como dado indisponível, sem virar automaticamente um rótulo negativo.

Truncamento no limite de geração é registrado como atributo do candidato. O programa extraído é avaliado pela mesma regra funcional: um candidato que terminou por limite pode, excepcionalmente, já conter uma solução válida. Caso o protocolo deseje definir sucesso como “passa e não truncou”, deve nomear essa variável separadamente e não confundi-la com corretude funcional. Esta versão adota a aprovação no verificador como rótulo principal.

=== Informação disponível durante a geração

Seja $cal(F)_(i j t)$ a informação observável até o prefixo de comprimento $t$: prompt, tokens amostrados, distribuições de decodificação calculadas e estados acessíveis até esse instante. O resultado final é observado apenas depois de completar e testar o programa. O alvo ideal de um preditor precoce é

$ p^*_(i j t) = P(S_(i j)=1 | cal(F)_(i j t)). $

Antes do término, o mesmo prefixo pode admitir continuações corretas e incorretas. Portanto, “trajetória com baixa chance de sucesso” não significa “trajetória inevitavelmente errada”. Dois candidatos que compartilham exatamente o mesmo prefixo podem receber a mesma previsão e posteriormente obter rótulos diferentes.

Pela propriedade da esperança condicional, a sequência ideal $p^*_(i j t)$ satisfaz, antes ou após absorção no término,

$ E[p^*_(i j,t+1) | cal(F)_(i j t)] = p^*_(i j t). $

Essa identidade é uma propriedade do preditor probabilístico ideal. Não implica que o probe ajustado seja uma martingala, que a confiança cresça monotonamente em cada trajetória ou que a AUC estimada aumente a cada checkpoint.

== Transformers, espaço latente e extração causal

=== Representações e atenção causal

Um Transformer representa tokens por vetores e aplica sucessivos blocos de atenção e transformações não lineares @vaswani2017. Em notação esquemática, para uma matriz de estados $H$, uma cabeça de atenção calcula

$ Q=H W_Q, quad K=H W_K, quad U=H W_U, $
$ "Attn"(H)= "softmax"((Q K^top)/sqrt(d_k)+M) U. $

A máscara causal $M$ impede que uma posição consulte tokens posteriores. Conexões residuais, normalizações e transformações feed-forward produzem estados contextualizados. O espaço latente é o espaço vetorial dessas representações, com dimensão determinada pela arquitetura. Não é um espaço de significados com coordenadas semanticamente nomeadas por definição.

O estado $h_(i j ell t) in RR^d$ é uma função do prompt e do prefixo processado. Para um modelo determinístico em modo de avaliação,

$ h_(i j ell t) = f_(theta,ell) (x_i,Y_(i j,1:t)). $

A aleatoriedade da representação decorre principalmente da amostragem de tokens; pesos e procedimento de extração permanecem constantes. A investigação testa a acessibilidade do sinal à classe de decodificadores escolhida. Na segunda etapa, logística e MLP recebem a mesma representação latente; muda a função estatística que procura ler o sinal, e não o gerador de código.

=== Convenção temporal e alinhamento

O checkpoint $t=0$ corresponde ao último token do prompt. Para $t>0$, o estado é extraído após processar o token $t$ da completion, e não o estado que foi usado para amostrar esse token. Em um decodificador autoregressivo, os logits usados para amostrar $Y_t$ dependem do prefixo que termina em $Y_(t-1)$; o estado de $Y_t$ participa da predição seguinte. Confundir essas posições produz um deslocamento de um token e invalida a interpretação temporal.

Um pipeline que para imediatamente depois de amostrar $Y_t$ pode ainda não ter calculado o seu estado. É necessário processá-lo, sem amostrar $Y_(t+1)$, para decidir a interrupção. Esse forward adicional ou antecipado deve ser incluído na latência e no custo de operação. Contar $t$ tokens gerados continua correto, mas não contabiliza sozinho todo o trabalho do Transformer.

=== Propriedade do estado anterior à amostragem

#caixa("Proposição 1 — Ausência de discriminação antes da divergência")[Com prompt, template, pesos e execução determinística fixados, o estado do último token do prompt é igual entre tentativas da mesma tarefa. Qualquer score determinístico calculado apenas desse estado também é constante dentro da tarefa.]

*Demonstração.* Para todo $j$, $h_(i j ell 0)=f_(theta,ell) (x_i)$. Logo, $z_(i j 0)=g(f_(theta,ell) (x_i))$ independe de $j$. Cada comparação entre uma tentativa correta e uma incorreta da tarefa resulta em empate. Uma AUC que atribui peso $1/2$ aos empates vale $1/2$ dentro dessa tarefa.

Esse resultado não impede que `prompt_last` apresente boa AUC entre tarefas. Ele o torna um baseline de dificuldade e um controle de integridade: diferenças relevantes entre repetições do mesmo prompt precisam ser explicadas por mudanças de entrada, dropout, extração incorreta ou não determinismo numérico. Não devem ser interpretadas como previsão da seed futura.

=== Teste de ausência de informação futura

Para cada camada e posição selecionadas, compare a extração usando apenas o prefixo e a extração na mesma posição de uma passagem sobre a sequência completa:

$ h_(ell,t) (x,y_(1:t)) approx h_(ell,t) (x,y_(1:T)). $

A igualdade é esperada matematicamente sob atenção causal. Diferenças numéricas podem decorrer de precisão e kernels. O teste deve fixar tolerância absoluta e relativa, modo de avaliação, backend e convenção de normalização. Uma falha exige diagnóstico antes de usar as ativações. Este é um teste de integridade da causalidade temporal da arquitetura, não uma demonstração de que uma direção latente causa corretude.

=== Estatísticas observáveis do prefixo

Se $q_u$ é a distribuição efetivamente amostrada na posição $u$, a NLL média até $t$ é

$ "NLL"_t=-1/t sum_(u=1)^t log q_u (Y_u). $

Ela aumenta quando os tokens escolhidos recebem menor probabilidade do próprio gerador. A entropia de uma distribuição é

$ "Ent"(q_u)=-sum_(v in cal(V)) q_u (v) log q_u (v), $

em que $cal(V)$ é o vocabulário. Ela descreve dispersão entre as alternativas, e não a corretude da resposta que será completada. A margem entre os dois maiores logits descreve outra forma de concentração. O vetor B utiliza resumos dessas grandezas até o checkpoint, sem acesso à continuação.

Uma resposta frequente e incorreta pode ter NLL baixa, assim como um programa correto pouco usual pode ter NLL maior. Esses resumos precisam ser avaliados empiricamente como preditores de corretude. Também não são medidas de confiança diretamente comparáveis entre vocabulários diferentes: a segunda etapa ajusta B separadamente em cada gerador.

=== Alcance informacional do estado interno

Se $H=f_theta (X,P)$, em que $P$ é o prefixo completo e o modelo é fixado, então $H$ não acrescenta informação aleatória quando $(X,P)$ já é integralmente conhecido. Em notação de informação mútua,

$ I(S;H | X,P)=0. $

A utilidade do estado está em representar o prefixo de modo que certas propriedades se tornem acessíveis a um preditor limitado. Assim, um ganho sobre NLL e entropia sustenta a afirmação de ganho sobre essas estatísticas, e não sobre todo método possível baseado no texto. Essa restrição impede transformar um resultado de probing em uma alegação excessiva sobre introspecção.

== Dificuldade da tarefa e sinal específico da trajetória

=== Heterogeneidade entre problemas

A probabilidade $pi_i$ varia com a tarefa e a configuração. Em um banco com $n_i$ tentativas, a proporção $hat(pi)_i=sum_j S_(i j)/n_i$ é uma estimativa descritiva dessa probabilidade. Uma tarefa sem sucessos observados não possui necessariamente probabilidade verdadeira zero; a amostra pode simplesmente não ter encontrado sua região de soluções corretas.

Para descrição do banco, definem-se três categorias mutuamente exclusivas: todas as tentativas corretas, resultados mistos e todas as tentativas incorretas. Esses nomes se referem aos resultados observados. Não autorizam afirmar que o modelo sempre acertará ou nunca resolverá o problema.

O número de tentativas influencia a classificação. Sob amostragem independente e $n$ tentativas, a probabilidade de observar resultados mistos é

$ P("mista" | pi_i,n)=1-pi_i^n-(1-pi_i)^n. $

Assim, a análise entre tarefas mistas representa principalmente problemas com variabilidade detectável nesse orçamento. Não é uma amostra aleatória de todos os problemas de programação.

=== Decomposição da associação

Para um score $Z$ e o indicador $S$, a lei da covariância total fornece

$ "Cov"(Z,S) = "Cov"(E[Z|X],E[S|X]) + E["Cov"(Z,S|X)]. $

O primeiro termo captura a associação entre tarefas; o segundo, a associação dentro da tarefa. A decomposição é conceitual e não uma decomposição da AUC. Ela mostra por que uma associação agregada pode ser explicada inteiramente pela dificuldade do prompt.

Como exemplo estritamente ilustrativo, considere duas tarefas: uma com 90% de acertos e outra com 10%. Um score constante igual a 0,9 para a primeira e 0,1 para a segunda pode ordenar bem exemplos agregados. Entretanto, não distingue qual tentativa acertará dentro de nenhuma das tarefas. Nenhum desempenho numérico desse exemplo é um resultado do modelo estudado.

=== Discriminação dentro da tarefa

Sejam $cal(P)_i$ e $cal(N)_i$ os conjuntos de tentativas corretas e incorretas elegíveis da tarefa $i$. Define-se a AUC intratarefa por

$ "AUC"_i = 1/(|cal(P)_i| |cal(N)_i|) sum_(j in cal(P)_i) sum_(k in cal(N)_i) psi(z_(i j)-z_(i k)), $

em que $psi(a)$ vale 1 se $a>0$, $1/2$ se $a=0$ e 0 se $a<0$. A média por tarefas mistas é

$ "WT-AUC"=1/(abs(cal(M))) sum_(i in cal(M)) "AUC"_i. $

Cada tarefa informativa recebe o mesmo peso. Uma média por todos os pares de candidatos daria maior peso a tarefas com mais pares positivo-negativo; seria outro estimando. A unidade principal de inferência é a tarefa, mesmo que milhares de pares sejam construídos a partir dela.

Tarefas com apenas uma classe não têm AUC intratarefa definida. Elas continuam nas métricas probabilísticas e operacionais. A exclusão da WT-AUC deve ser acompanhada dos respectivos denominadores, para que um resultado elevado em poucas tarefas não pareça representar todo o benchmark.

=== Informação incremental e hipótese operacional

Seja $B_(i j t)$ um vetor fixado de características observáveis do prefixo. Compare um modelo que usa apenas $B$ com outro que combina $B$ e a ativação. Os estimandos centrais são

$ Delta_W = "WT-AUC"_(H+B) - "WT-AUC"_B, $
$ Delta_L = "LogLoss"_(H+B) - "LogLoss"_B. $

Valores positivos de $Delta_W$ e negativos de $Delta_L$ indicam ganho, mas os dois critérios avaliam propriedades diferentes: ordenação dentro da tarefa e qualidade de previsão probabilística. Um ganho robusto sustenta a utilidade incremental da representação para a classe de modelos comparada. Não demonstra independência condicional universal nem que todos os fatores semânticos ou estilísticos foram controlados.

A hipótese nula de ausência de qualquer sinal dentro da tarefa, $S$ independente de $Z$ dado $X$, difere da hipótese de ausência de informação adicional sobre $B$, isto é, $S$ independente de $Z$ dado $(X,B)$. Essa distinção orienta a interpretação dos testes estatísticos posteriores.

=== Perdas probabilísticas e calibração

Para um resultado $S in {0,1}$ e uma previsão $p$ entre zero e um, as duas perdas utilizadas são

$ ell_"log" (S,p)=-S log p-(1-S)log(1-p), quad ell_"Brier" (S,p)=(S-p)^2. $

A log loss penaliza fortemente um erro ao qual se atribuiu probabilidade extrema. O Brier mede a distância quadrática entre probabilidade e resultado. Ao calcular a média, o estudo dá o mesmo peso total a cada tarefa contribuinte e divide esse peso entre suas tentativas ativas. Tarefas com somente acertos ou somente erros podem entrar nessas perdas, embora não identifiquem AUC intratarefa.

Calibração pergunta se respostas às quais se atribui probabilidade p acertam com frequência aproximadamente p na população considerada. A calibração Platt ajusta uma transformação logística do score bruto, $sigma(a z+b)$, usando dados de desenvolvimento adequados. Com inclinação positiva, essa transformação preserva a ordenação, mas muda as probabilidades e as decisões em limiares numéricos fixos. Isso explica como uma métrica probabilística pode mudar sem melhora de AUC.

== Capacidade do decodificador de corretude

=== Da representação à probabilidade

Seja $v$ o vetor fornecido ao probe: apenas $B$ ou a concatenação de $B$ com a representação de $H$. A regressão logística constrói um score afim e o transforma em probabilidade pela função sigmoide:

$ z_"lin" (v)=w^top v+b, quad sigma(z)=1/(1+exp(-z)). $

A fronteira entre scores altos e baixos é um hiperplano nesse espaço de features. Uma relação pode estar presente nos estados internos e ainda assim não ser bem aproximada por essa fronteira, sobretudo com regularização e amostra finita. Esse é um limite do procedimento de leitura, não uma demonstração de que o estado não contém informação relevante.

Um MLP de uma camada oculta usa uma transformação intermediária:

$ z_"MLP" (v)=w_2^top phi(W_1 v+b_1)+b_2, quad phi(a)=max(0,a). $

A ReLU é aplicada por coordenada. Ao combinar essas unidades, o classificador pode representar regiões e interações que uma função afim das features originais não representa diretamente. Essa flexibilidade também aumenta a possibilidade de ajustar ruído. Por isso, largura, regularização, parada do treinamento e seeds precisam ser definidos e selecionados sem acesso aos rótulos do teste externo.

Na segunda etapa, cada classe é comparada consigo mesma:

$ Delta_"linear"="WT-AUC"(H+B,"linear")-"WT-AUC"(B,"linear"), $
$ Delta_"MLP"="WT-AUC"(H+B,"MLP")-"WT-AUC"(B,"MLP"), $
$ Gamma=Delta_"MLP"-Delta_"linear". $

$Gamma$ pergunta se a mudança de classe altera o incremento atribuído a $H$. Um $Gamma$ positivo pode significar apenas que uma desvantagem ficou menor: se $Delta_"linear"=-0.06$ e $Delta_"MLP"=-0.01$, então $Gamma=0.05$, embora nenhum dos incrementos seja positivo. Os números desse exemplo são ilustrativos. O contraste absoluto precisa ser examinado junto à interação.

A correspondência entre os probes significa mesma profundidade, larguras candidatas e regras de seleção dentro da classe. Não significa igualdade exata do número de parâmetros: acrescentar features a $H+B$ aumenta a primeira matriz de pesos. Assim, os resultados comparam procedimentos controlados e não fornecem uma decomposição causal perfeita entre quantidade de parâmetros e conteúdo da representação.

=== Redução de dimensão e risco de vazamento

Os estados têm centenas ou milhares de coordenadas. No PCA, uma base de componentes é estimada a partir da covariância das features de treinamento. A projeção em 64 componentes reduz a dimensão antes do ajuste do probe; o whitening reescala as coordenadas projetadas segundo suas variâncias estimadas. Essas operações não usam diretamente o rótulo, mas aprendem características da distribuição dos exemplos. Logo, também precisam ser ajustadas somente no treino de cada partição.

A mesma restrição vale para imputação e padronização. Ajustar o PCA uma única vez com todas as trajetórias e depois separar folds permitiria que a transformação visse as tarefas pontuadas. O protocolo evita esse vazamento repetindo o pré-processamento dentro dos treinos. Na aplicação externa, utiliza a transformação já congelada da origem, sem recalcular componentes com dados MBPP+.

A hipótese experimental continua sendo uma comparação de predição, e não uma estimativa direta de informação mútua. Em particular, a ausência de melhora de um probe regularizado não implica $I(S;H|B)=0$. Também não elimina a possibilidade de um baseline textual mais forte capturar o mesmo padrão que o probe encontra no estado.

== Amostragem repetida, seleção e custo

=== Cobertura e pass\@k

Para $N$ candidatos, a cobertura da tarefa é

$ A_i (N)=bold(1){exists j <= N : S_(i j)=1}. $

Sob tentativas independentes e identicamente distribuídas condicionalmente à tarefa,

$ P(A_i (N)=1|x_i)=1-(1-pi_i)^N. $

O ganho de passar de $N-1$ para $N$ tentativas é $pi_i (1-pi_i)^(N-1)$, que decresce quando $0<pi_i<1$. Essa expressão não deve ser aplicada após substituir $pi_i$ por uma taxa global sem considerar heterogeneidade: em geral, a média das coberturas por tarefa difere da cobertura calculada com a média das probabilidades.

Com $n$ candidatos e $c_i$ aprovados, o estimador usual de pass\@k é @chen2021codex

$ hat("pass@k")_i = 1 - binom(n-c_i,k)/binom(n,k), quad k <= n. $

Esse estimador descreve a probabilidade de incluir um sucesso em um subconjunto de tamanho $k$. Já a resolução observada dos primeiros $k$ candidatos é $bold(1){sum_(j=1)^k S_(i j)>0}$. Ambos são úteis, mas não são números intercambiáveis. O replay operacional preserva uma ordem de tentativas e utiliza o segundo objeto.

=== Terminologia: Best-of-N e verificador

“Best-of-N” frequentemente descreve gerar candidatos e escolher o melhor segundo um score. Neste documento, o baseline operacional é mais precisamente *amostragem repetida com verificador e parada no primeiro sucesso*. Ele é denominado BoN com verificador por continuidade com o projeto, com essa definição explícita.

A suíte completa de testes é acessada pela política apenas quando um candidato termina. Esse acesso representa um cenário de verificação forte: o programa não recebe os testes ocultos no prompt, mas o controlador pode utilizá-los para decidir a parada. Isso difere da seleção em produção, na qual os testes disponíveis podem ser incompletos. As conclusões operacionais devem ser limitadas a esse cenário, ou acompanhadas futuramente de uma separação entre testes de seleção e de avaliação final.

=== Custo até sucesso ou esgotamento

Defina $K_i=min{j:S_(i j)=1}$, adotando $K_i=infinity$ quando não há sucesso no banco. O custo do baseline até o primeiro sucesso ou limite $N$ é

$ C_i^V (N)=sum_(j=1)^N T_(i j) bold(1){K_i>=j}. $

Tarefas não resolvidas contribuem com o custo de todas as tentativas iniciadas. Excluir essas tarefas do custo médio favoreceria políticas que simplesmente desistem de problemas difíceis.

A métrica primária de custo é tokens de completion por tarefa, $overline(C)=sum_i C_i/m$. Uma métrica secundária é o custo agregado por tarefa resolvida,

$ C_"por solução" = (sum_i C_i)/(sum_i A_i), $

que fica indefinido se nenhuma tarefa for resolvida. Ela inclui no numerador o custo das tarefas sem sucesso. Não é a média do custo apenas entre tarefas que tiveram sucesso.

=== Retenção de candidatos e efeito sobre resolução

Considere uma política que mantém um candidato com indicador $R_(i j)$, com $R=1$ para candidatos que terminaram naturalmente antes da decisão. O sucesso útil é $S_(i j) R_(i j)$. Para $pi_i>0$, defina $r_i=P(R=1|S=1,x_i)$, e defina $a_i=P(R=1|x_i)$. Se $pi_i=0$, a retenção condicionada ao sucesso não é identificável, mas a probabilidade de sucesso útil é zero sem precisar defini-la.

Sob tentativas independentes e a mesma regra por tentativa,

$ P(S R=1|x_i)=pi_i r_i, $
$ P(A_i^pi (N)=1|x_i)=1-(1-pi_i r_i)^N. $

A precisão entre candidatos mantidos é $P(S=1|R=1,x_i)=pi_i r_i/a_i$, quando $a_i>0$. Essa precisão pode crescer mesmo quando a cobertura com orçamento fixo diminui. Portanto, “maior fração de acertos entre os mantidos” não basta como evidência de melhoria do sistema.

#caixa("Proposição 2 — Limite da interrupção no mesmo banco")[Quando a política usa exatamente os mesmos $N$ candidatos potenciais e só pode descartá-los, seu conjunto de sucessos é subconjunto do conjunto de sucessos do baseline. Logo, $A_i^pi (N)<=A_i^V (N)$ para cada tarefa.]

*Demonstração.* Se um sucesso é encontrado pela política, o candidato correspondente está no banco original e também passa no verificador do baseline. A política não cria novos candidatos além do limite e não altera o conteúdo dos mantidos. Assim, não pode resolver uma tarefa que nenhum dos $N$ candidatos originais resolvia.

O ganho esperado nesse regime é reduzir custo com perda pequena ou nula de resolução. Para superar cobertura em um mesmo orçamento de tokens, seria necessário reinvestir a economia em novas tentativas, com um banco maior e uma comparação própria. Essa extensão não integra o experimento principal.

=== Condição econômica para interromper

Uma política pode economizar tokens em um candidato e aumentar o custo total da tarefa ao eliminar um sucesso precoce, forçando outras tentativas. Por isso, o custo do replay não é necessariamente monotônico no limiar de interrupção.

Sob tentativas iid por tarefa, seja $D$ o custo de uma tentativa sob a política e $s=pi_i r_i$ sua probabilidade de sucesso útil. Como iniciar uma tentativa depende dos resultados anteriores, independentes da tentativa atual,

$ E[C_i^pi (N)|x_i]=E[D|x_i] sum_(j=0)^(N-1)(1-s)^j. $

Para $s>0$, a soma vale $(1-(1-s)^N)/s$; para $s=0$, vale $N$. Essa expressão mostra que a redução do custo por tentativa deve compensar a possível redução na probabilidade de sucesso por tentativa. A análise empírica usa os caminhos efetivos do banco e não depende de estimar $pi_i$ para substituir nessa fórmula.

=== Derivação, limite ilimitado e retenção ativa

Para explicitar as hipóteses, condicione em $x_i$ e suponha que os pares
$(D_j,S_j R_j)$ sejam identicamente distribuídos e independentes entre
tentativas. Permite-se dependência entre custo e sucesso da *mesma* tentativa.
Se $I_j$ indica que as tentativas anteriores não produziram sucesso mantido,
então $I_j$ é independente de $D_j$, e

$ E[D_j I_j | x_i]=E[D|x_i] P(I_j=1|x_i)
 =d_i (1-pi_i r_i)^(j-1). $

Somar essas identidades demonstra a expressão anterior sem impor que
candidatos corretos e incorretos tenham o mesmo comprimento. A hipótese falha
se o controlador adapta prompts, seeds, amostragem ou sua regra com base
nas tentativas anteriores. O replay preserva ordem e configuração, mas o
banco finito não prova que as tentativas de uma aplicação futura sejam iid.

Se $pi_i>0$, $r_i>0$ e os custos médios são finitos, o limite com tentativas
ilimitadas é

$ E[C_i^pi (infinity)|x_i]=d_i/(pi_i r_i), quad
 E[C_i^V (infinity)|x_i]=tau_i/pi_i, quad tau_i=E[T|x_i]. $

Logo, a interrupção reduz o custo esperado nesse limite precisamente quando

$ d_i/tau_i < r_i. $

Trata-se de uma consequência algébrica conhecida de uma soma geométrica,
não de um teorema inédito. Se $pi_i r_i=0$, o tempo até sucesso não é
finito e não se aplica essa comparação de razões. Nenhum sucesso em vinte
tentativas tampouco demonstra que a probabilidade verdadeira seja zero.

Para orçamento finito, escreva $G_N (s)=sum_(j=0)^(N-1)(1-s)^j$. A condição
de menor custo é $d_i/tau_i < G_N (pi_i)/G_N (pi_i r_i)$, enquanto a cobertura
é $1-(1-pi_i r_i)^N$. Quando $N=1$, economizar em candidatos é suficiente
para reduzir custo; ao aumentar $N$, os sucessos perdidos pesam mais. Menor
custo finito pode refletir abandono de soluções e precisa ser examinado junto
à resolução, sem interpretação automática de eficiência.

A retenção $r_i$ dessas equações é *global*, incluindo sucessos terminados
antes do gate. Se $a_i^+=P(T>t|S=1,x_i)$ e $r_i^"ativo"$ é a retenção
entre corretos ativos, então

$ r_i=1-a_i^+(1-r_i^"ativo"). $

Portanto, substituir indiscriminadamente recall ativo por $r_i$ altera o
modelo econômico. A análise empírica explicita os dois denominadores.
Se $a_i^+=0$, o recall ativo não é identificável e os corretos terminam antes
do gate, de modo que $r_i=1$; a identidade anterior se aplica diretamente
quando $a_i^+>0$.

Finalmente, sob heterogeneidade entre tarefas, o custo populacional é
$E_X [d_X G_N (pi_X r_X)]$. Em geral ele não coincide com
$E_X [d_X] G_N (E_X [pi_X] E_X [r_X])$. A média agregada de acertos não pode
substituir as probabilidades condicionais nessa expressão não linear.
Overheads de prefill, gate e verificação devem ser acrescidos ao custo por
tentativa na mesma unidade, preservando sua dependência com o caminho.
Um tempo de probe não é uma quantidade de tokens sem hipótese de conversão
explicitada e validada; as medições deste estudo mantêm essas unidades separadas.

=== Oracle retrospectivo e alcance do limite econômico

O oracle idealizado conhece $S_(i j)$, interrompe todos os incorretos ativos
em um checkpoint fixo $t$ e mantém todos os corretos. Usa a mesma ordem e
para no primeiro sucesso. Em cada tarefa, seu custo é no máximo o do verificador
e sua resolução é idêntica. Entre políticas no mesmo checkpoint que preservam
*todos* os candidatos corretos, ele minimiza o custo nesse caminho: nenhuma
delas pode cobrar menos que $t$ por um incorreto ainda ativo ou menos que o
comprimento completo do primeiro sucesso mantido.

Esse limite é condicional às restrições descritas. Uma política que descarta
sucessos pode, em certos caminhos, custar menos que esse oracle e perder
cobertura, ou alcançar depois um sucesso mais curto. O oracle não é limite
inferior universal para toda política de inferência nem é implementável online.
Sua utilidade é separar o espaço econômico disponível da capacidade real de
identificar candidatos descartáveis sem conhecer seus rótulos finais.

=== Tokens, latência e fronteira custo–resolução

Tokens gerados são uma medida auditável de trabalho de decodificação, mas não equivalem a FLOPs, energia ou tempo. Prefill, comprimento do contexto, cache, extração de estados, cálculo de entropia, transferência GPU–CPU, inferência do probe e verificação têm custos próprios. Os resultados devem distinguir economia de tokens e aceleração medida em execução.

Uma política domina outra no plano descritivo se tem resolução ao menos igual e custo não maior, com uma desigualdade estrita. Na presença de incerteza amostral, a comparação deve reportar diferenças pareadas e intervalos. A fronteira apresentada corresponde às políticas e limiares previamente definidos; não representa a fronteira ótima global de todos os algoritmos possíveis.

== Decisão sequencial sob orçamento finito <sec-bellman>

=== Objetivo e estado

O replay anterior avalia regras fixadas. Para entender quais grandezas uma regra ideal precisaria considerar, pode-se formular um problema de decisão sequencial. Seja $lambda$ a penalidade por encerrar sem uma solução verificada. O objetivo é minimizar o custo esperado de geração, aquisição dos sinais e verificação, acrescido dessa penalidade, sob um orçamento finito $b_0$.

Custos e penalidade devem estar na mesma unidade, por exemplo milissegundos ou uma unidade monetária definida. Se forem usados tokens, o overhead do gate e do verificador exige uma conversão explícita, ou precisa ser mantido fora desse objetivo de tokens. Somar milissegundos e tokens sem conversão não define uma função de custo. Custos já pagos são afundados: importam para o total da política, mas não são cobrados novamente na comparação entre ações futuras.

O estado de uma tentativa ativa é $(k,b,h,z)$: $k$ novas tentativas ainda disponíveis *além da atual*, orçamento restante $b$, prefixo observado $h$ e crença $z$ sobre a tarefa. A crença resume o que resultados anteriores informam sobre sua dificuldade. O score de um probe só poderia substituir esse estado completo se fosse uma estatística suficiente para as distribuições de sucesso e custo futuro; essa suficiência não é assumida nos experimentos.

As ações são continuar a tentativa, abortá-la para considerar outra, ou encerrar sem solução. Há ainda a decisão de iniciar uma tentativa quando nenhuma está ativa. Um sucesso verificado encerra o processo com custo futuro zero. Sem novas tentativas, ainda pode ser ótimo completar a atual.

=== Recursão de Bellman

Denote por $J_k (b,z)$ o menor custo esperado sem tentativa ativa e por $F_k (b,h,z)$ o custo no checkpoint após obter seu score. Suponha inicialmente custos determinísticos por transição e aquisição obrigatória dos sinais nos checkpoints previstos. Se iniciar uma tentativa custa $a$ até o primeiro checkpoint, incluindo o primeiro gate, então

$ J_0 (b,z)=lambda, quad
J_k (b,z)=min(lambda,a+E[F_(k-1) (b-a,H_0,z)]). $

O segundo ramo só é admissível para $k>=1$ e $b>=a$. A esperança considera o prefixo inicial aleatório $H_0$. No checkpoint, se $d(h)$ é o custo de gerar até o próximo evento,

$ F_k (b,h,z)=min(lambda,J_k (b,z),d(h)+E[Phi_k (b-d(h),H',Z')]). $

O custo do gate atual já foi pago e não aparece apenas no ramo de continuar. Se o próximo evento for outro checkpoint, $Phi$ inclui o custo do próximo gate e retorna a $F$. Se for o término, inclui o custo $v$ do verificador e retorna zero em caso de sucesso ou $J_k$ com orçamento e crença atualizados em caso de falha. Ações que excedem o recurso disponível são removidas. Se não houver recurso para uma verificação ou aquisição obrigatória, o modelo deve prever o encerramento sem solução. Quando adquirir o próprio sinal for opcional, é necessário acrescentar uma ação anterior à observação que pague esse custo.

Explicitamente, escrevendo $u=b-d(h)$, o ramo de outro checkpoint custa $g'+F_k (u-g',H',Z')$, admissível se $u>=g'$, em que $g'$ é o custo do próximo sinal. No término, o ramo custa $v+(1-S')J_k (u-v,Z')$, admissível se $u>=v$; $S'$ é o resultado da verificação. Assim, geração, aquisição e verificação reduzem o orçamento no ponto em que são pagas. O ramo de sucesso tem custo futuro zero depois de pagar $v$.

Os limites de tentativas e tokens asseguram horizonte finito. A recursão escolhe o ramo admissível de menor custo. Sua interpretação exige transições conhecidas ou crenças suficientes, custos aditivos e uma regra de verificação bem especificada. Independência entre tentativas condicionalmente à tarefa não implica independência marginal: fracassos anteriores podem atualizar $z$. O caso sintético testado simplifica esse ponto ao usar probabilidade de sucesso conhecida e tentativas iid.

=== Quando a regra se reduz a um limiar

Considere uma decisão única, sem novos checkpoints dentro da tentativa. Sejam $p_t$ sua probabilidade condicional de sucesso, $c_t$ o custo restante e $V_k$ o custo ótimo de fallback, incluindo a penalidade por não resolver. Mantém-se a convenção de $k$ novas tentativas além da atual. Suponha que falhar após continuar e abortar agora conduzam *ao mesmo estado futuro*. Então

$ Q_"continuar"=c_t+(1-p_t)V_k, quad Q_"abortar"=V_k. $

Subtraindo os custos, continuar é preferível ou empatado precisamente quando

$ c_t-p_t V_k<=0 quad <==> quad p_t V_k>=c_t. $

$p_t$ é adimensional; $c_t$ e $V_k$ são custos na mesma unidade. $V_k$ não é a probabilidade de sucesso de outra tentativa. Se $V_k>0$, o limiar simplificado é $c_t/V_k$; se $c_t>V_k$, nenhuma probabilidade admissível torna continuar vantajoso. Sem novas tentativas, $V_0=lambda$. Se $V_k=0$, apenas $c_t=0$ permite empate.

A hipótese de fallback idêntico é forte. Com orçamento estrito, completar a tentativa consome recurso e altera o estado após fracasso. Nesse caso, a comparação envolve $J_k (b-c_t,z_"falha")$ e $J_k (b,z_"aborto")$, que não são geralmente iguais. Custos aleatórios correlacionados ao sucesso também exigem uma esperança conjunta; substituir tudo por médias não preserva automaticamente uma restrição estrita de orçamento.

=== Conexão com a política empírica

Os gates estudados usam limiares escolhidos para uma meta de retenção na origem. Não estimam as funções $J$ e $F$, nem implementam a solução ótima dessa recursão. A formulação explica por que um limiar ótimo poderia depender da tarefa, do comprimento restante, do orçamento e do número de tentativas. Também mostra por que boa ordenação por AUC não garante decisões econômicas perto do limiar.

O problema pertence à tradição de seleção de computações por custo e valor esperado @russell1991 @callaway2018 e se relaciona à parada na amostragem de respostas completas @wan2025beacon. A contribuição aqui é explicitar essas condições para a interrupção de prefixos. O solver sintético foi confrontado com enumeração exata de políticas iid pequenas; essa validação é interna ao modelo e não demonstra ganho econômico dos probes nos benchmarks.
