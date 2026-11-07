#import "../lib.typ": *

= Metodologia

A implementação revisada segue o protocolo v0.2, de 12 de outubro de 2026.
As regras a seguir definem o experimento principal executado. Resultados e
auditorias aparecem no capítulo empírico. O registro histórico mantém a aprovação
do orientador pendente para v0.2; a revisão da extensão foi depois informada pelo
estudante. Execução autorizada e validação técnica não equivalem a pré-registro
independente. A reanálise histórica e as extensões posteriores são identificadas
separadamente e não alteram os endpoints definidos antes do banco principal.

== Desenho experimental e construção dos dados

=== Modelo e ambiente

O modelo principal utilizado é Qwen2.5-Coder-1.5B-Instruct, integrante de uma família especializada em código @hui2024. Os manifestos registram identificador completo, revisão imutável dos pesos, tokenizer, template de chat, precisão e backend. A escolha da variante Instruct é específica; não deve ser substituída silenciosamente pela variante base.

A viabilidade na RTX 3050 foi medida no piloto e nas sessões de coleta, com batch 1, BF16 nos pesos e projeções FP32. Não foi usada quantização. Geração e extração preservam a mesma condição numérica. Alterar precisão, quantização ou backend exigiria outra condição experimental, com validação e proveniência próprias.

=== Benchmark principal e unidade de generalização

HumanEval+ é o benchmark principal, avaliado pelo EvalPlus. O snapshot utilizado contém 164 tarefas, com identificadores e hash nos manifestos. A avaliação de generalização separa tarefas, não apenas tentativas. Problemas semelhantes identificados antes dos resultados precisariam compartilhar um grupo; o desenho efetivamente usado agrupa por task_id e não certifica ausência de similaridade semântica entre tarefas.

O acesso prévio a algumas tarefas para desenvolvimento não desaparece com a geração de novas seeds. Tarefas usadas para decisões de método devem ser registradas como desenvolvimento. O particionamento verifica a generalização a prompts retidos dentro do benchmark, não certifica ausência de contaminação do benchmark no pré-treinamento do modelo.

=== Piloto e congelamento

O planejamento reserva 32 tarefas, selecionadas deterministicamente pelo hash de `protocol-v0.1-pilot:` concatenado a `task_id`, para desenvolvimento. As demais 132 tarefas constituem o conjunto da avaliação principal. As 32 tarefas não entram nos folds de teste. O histórico de geração prévia em 26 das outras tarefas é declarado; o estudo não presume que essas tarefas fossem desconhecidas do pesquisador. Nenhuma exclusão é feita com base no desfecho observado.

O piloto testa temperaturas 0,6, 0,8 e 1,0, com top-p igual a 0,95, dez tentativas por tarefa e limite inicial de 512 tokens. A finalidade é verificar execução, quantidade de tarefas mistas, taxas de truncamento e custo. AUC de probes não deve ser usada para escolher a temperatura. A regra proposta privilegia o número de tarefas mistas, com taxa agregada de aprovação entre 10% e 90%; empates são resolvidos por taxa mais próxima de 50%, menor comprimento médio e menor temperatura.

O aumento material foi fixado em cinco pontos percentuais acima da menor taxa combinada de sintaxe inválida, extração vazia ou truncamento entre temperaturas. A regra previa revisão anterior à análise principal caso nenhuma configuração fosse elegível. A seleção efetiva usou o piloto completo e escolheu temperatura 1,0, sem usar AUC dos probes.

=== Banco de candidatos

A configuração definitiva utilizou 20 tentativas para cada tarefa principal. Com 132 tarefas, o banco completo contém 2.640 trajetórias. As ativações e estatísticas foram extraídas desse mesmo banco. Gerar bancos distintos para cada método acrescentaria variação de candidatos à comparação dos métodos.

Cada tentativa tem seed própria, derivada de uma função hash documentada do protocolo, versão do benchmark, tarefa e índice da tentativa, convertida para o intervalo aceito pelo gerador. Deve-se verificar colisões na conversão. A seed é reinicializada antes de cada tentativa, de modo que interromper uma amostra anterior não altere o fluxo aleatório da seguinte.

A independência estatística entre amostras é uma aproximação do desenho de seeds e do decodificador. A reprodutibilidade bit a bit depende também do hardware e das bibliotecas. O banco arquivado, com hashes, é a referência empírica; uma seed isolada não substitui a preservação dos candidatos.

=== Contrato dos registros

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

=== Verificação funcional e limites de execução

O código gerado é avaliado pelo checker oficial em subprocessos POSIX/WSL, com seus limites de tempo e memória e o mecanismo reliability_guard. Esse mecanismo desabilita algumas operações, mas o próprio EvalPlus esclarece que não constitui um sandbox de segurança. A implementação utilizada não comprova isolamento de rede ou ausência de credenciais do ambiente; essas propriedades não são reivindicadas. Reexecuções preservam os mesmos limites do checker, e diferenças de ambiente ou flutuações precisam ser diagnosticadas como possíveis alterações de mensuração.

O parser de completion para código é determinístico e definido antes da avaliação. Regras para remover cercas Markdown, tratar prefixos e extrair funções não podem variar após a inspeção de exemplos que falharam. Alterar o parser muda o experimento e exige versionar os rótulos.

== Preditores precoces e modelos estatísticos

=== Checkpoints e conjunto sob risco

Foram avaliados os checkpoints 0, 8, 16, 32 e 64 tokens; a representação final é uma referência tardia. O endpoint principal predefinido é 32 tokens. As outras posições descrevem a evolução temporal e participam da escolha interna da política operacional.

A ordem da decisão é: primeiro verificar se a geração terminou; se terminou, encaminhar ao verificador; caso contrário, avaliar a regra de interrupção. Assim, para uma decisão depois de $t$ tokens, o conjunto sob risco é

$ cal(R)_t={(i,j):T_(i j)>t}. $

O uso da desigualdade estrita evita classificar como abortada uma completion que terminou exatamente no checkpoint. Um estado existente em $T=t$ pode ser usado em uma análise descritiva separada, mas já não corresponde à mesma decisão de continuação. Gerações curtas não são preenchidas com zeros nem recebem uma ativação futura fictícia.

As curvas em diferentes checkpoints condicionam em conjuntos diferentes. Deve-se reportar, em cada ponto, candidatos ativos, tarefas representadas, tarefas mistas e prevalência. Uma análise descritiva adicional sobre o conjunto comum que permanece ativo até 64 tokens ajuda a distinguir mudança de população e mudança de informação; ela representa essa população restrita, e não todo o benchmark.

=== Camadas e convenção do estado

As quatro profundidades candidatas correspondem aos blocos mais próximos de 25%, 50%, 75% e 100% da profundidade. Os índices são calculados uma vez, registrados e deduplicados se necessário. O manifesto especifica se o vetor corresponde à saída residual do bloco antes ou depois de normalização. O embedding inicial não conta como bloco Transformer.

A captura reproduz o estado disponível na inferência incremental. Salvar todas as ativações de todos os tokens não é requisito: foram preservados os vetores selecionados, tokens e metadados, reduzindo o armazenamento. O banco completo permite avaliar os desfechos potenciais; na validação online, a política obteve seus estados no checkpoint e abortou sem executar a continuação descartada.

=== Estatísticas observáveis

Sejam $a_(u,v)$ os logits do token $v$ no passo $u$. A distribuição bruta é $p_u (v)="softmax"(a_u)_v$. O decodificador aplica temperatura, seleção do núcleo e renormalização para produzir $q_u$. A probabilidade bruta e a efetivamente amostrada são objetos diferentes.

Para a distribuição de amostragem, definem-se

$ "NLL"_u=-log q_u (Y_u), quad "Ent"_u=-sum_(v:q_u (v)>0) q_u (v) log q_u (v). $

Até $t$, as características principais incluem média de NLL, NLL do último token, média de entropia, entropia máxima e média da diferença entre os dois maiores logits. Para esta última, usam-se os logits finitos antes do top-p, após a escala de temperatura, evitando uma margem infinita quando o filtro deixa apenas um token. A base do logaritmo é natural e a unidade é nat.

A logprobabilidade do prefixo é $sum_(u=1)^t log q_u (Y_u)$. Em um checkpoint fixo, ela é uma transformação linear da NLL média; incluir ambas não cria uma fonte independente de informação. O comprimento final, o tempo total futuro e o resultado dos testes nunca entram em features precoces. O número de tokens $t$ também é constante em uma análise de checkpoint fixo.

Comprimento do prompt e da assinatura são controles de tarefa. Se a assinatura não puder ser extraída com regra confiável, seu indicador de ausência é registrado. Estatísticas equivalentes da distribuição bruta podem ser uma análise de sensibilidade previamente definida, porque entropia após top-p depende do filtro de amostragem.

=== Baselines e modelo principal

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

=== Probe linear e função objetivo

Para um vetor de features $a$, o probe define

$ z=b+w^top a, quad hat(p)=sigma(z)=1/(1+exp(-z)). $

Se a tarefa $i$ contribui com $n_i$ tentativas elegíveis, cada tentativa recebe peso proporcional a $1/n_i$. Uma forma da função objetivo é

$ J(b,w)=1/m_"tr" sum_(i in cal(I)_"tr") 1/n_i sum_j ell(S_(i j),sigma(b+w^top a_(i j))) + lambda/2 norm(w)_2^2, $

com $ell(s,p)=-s log p-(1-s) log(1-p)$. O intercepto não é penalizado. A relação exata entre $lambda$ e o parâmetro `C` depende da normalização usada pela implementação; a configuração preserva a definição da biblioteca, sem assumir que qualquer convenção equivale numericamente a $lambda=1/C$.

Padronização é ajustada apenas no conjunto de treinamento de cada split. Uma direção por diferença de médias,

$ v=hat(mu)_1^"tr"-hat(mu)_0^"tr", $

pode ser um diagnóstico geométrico secundário. A direção e sua orientação são construídas exclusivamente com rótulos de treino. Separação visual em projeções escolhidas depois de olhar todo o banco não constitui evidência confirmatória.

=== Calibração e interpretação dos scores

Uma transformação logística de calibração, $hat(p)_"cal"=sigma(a z+b)$, pode ser ajustada com previsões de desenvolvimento produzidas fora da amostra. Calibração trata a relação entre score e frequência observada de sucesso; é distinta de discriminação @guo2017. Uma transformação estritamente crescente preserva a ordenação e a AUC.

A interrupção por limiar necessita principalmente de uma boa ordenação e de um limiar validado. Probabilidades bem calibradas facilitam interpretação e comparações, mas não são condição matemática para obter economia. O desempenho probabilístico foi reportado sem apresentar calibração como garantia de retenção de corretos em tarefas novas.

== Validação, inferência estatística e controles

=== Separação por tarefa e avaliação aninhada

As tentativas de um problema compartilham especificação e estrutura de solução. Dividi-las aleatoriamente entre treino e teste permite que o treinamento se beneficie de características do próprio problema avaliado. A separação exigida é

$ cal(I)_"tr" ∩ cal(I)_"te" = emptyset, $

em que os conjuntos contêm identificadores de tarefas. As mesmas tarefas pertencem ao mesmo fold em todos os checkpoints e métodos. Não se refaz o particionamento para obter uma distribuição mais favorável de rótulos depois dos resultados.

Utiliza-se avaliação externa com cinco folds agrupados e validação interna com três folds agrupados dentro de cada treino externo. A camada e `C` são selecionados internamente. O risco de sobreajuste ao critério de seleção motiva a separação entre seleção e avaliação @cawley2010. Todos os passos dependentes dos dados, inclusive padronização e construção de direções, são repetidos no treino correto.

O conjunto candidato de regularização é $C in {0.001,0.01,0.1,1,10}$. No endpoint científico de 32 tokens, a posição não é selecionada. Para cada modelo, o critério interno principal é log loss médio por tarefa. Dentro de uma faixa de um erro-padrão do melhor valor interno, usa-se maior regularização e depois camada mais rasa. O erro-padrão para essa regra é uma heurística de seleção, não um intervalo de confiança do resultado final.

O objeto avaliado pela validação externa é o procedimento que escolhe, treina e aplica um modelo usando apenas tarefas de treino. As camadas selecionadas podem variar entre folds. Não se deve apresentar a melhor camada encontrada posteriormente em todo o banco como se tivesse sido definida antecipadamente.

=== Predições fora da amostra e calibração interna

Cada candidato principal recebe uma previsão de um modelo que não treinou em sua tarefa. A concatenação dessas previsões permite calcular métricas sobre o conjunto externo completo. As escolhas continuam restritas ao interior de cada treino externo.

Para calibrar e selecionar a política, obtêm-se scores internos fora da amostra. Quando a seleção de hiperparâmetros participa desse processo, ela também precisa ocorrer sem o fold interno cujos scores estão sendo produzidos, ou os scores devem ser tratados apenas como dados de desenvolvimento com viés de seleção. A implementação v0.2 produz scores por cross-fitting da seleção inteira: cada partição pontuada é excluída também da escolha de camada e regularização. O calibrador é ajustado nesses scores; calibrador e limiar continuam sendo ajustes de desenvolvimento avaliados integralmente em tarefas externas.

Calibrador e limiar podem ser ajustados nos dados de desenvolvimento desde que todo o sistema seja avaliado depois em tarefas externas intocadas. O recall obtido nesse conjunto de ajuste é uma meta empírica, sujeita a otimismo por seleção. Para reivindicar uma certificação de recall, seria necessário um conjunto independente de certificação e um procedimento de inferência adequado à dependência entre tentativas; o presente desenho não faz essa reivindicação.

=== Métricas probabilísticas e de ranking

As métricas principais de probabilidade são log loss e Brier, com peso igual por tarefa:

$ "LogLoss"=1/m sum_i 1/n_i sum_j ell(S_(i j),hat(p)_(i j)), $
$ "Brier"=1/m sum_i 1/n_i sum_j (hat(p)_(i j)-S_(i j))^2. $

Para estabilidade numérica, probabilidades usadas em logaritmos são limitadas a $[epsilon,1-epsilon]$, com $epsilon$ documentado e idêntico entre métodos. A AUC agregada é reportada como diagnóstico entre candidatos, acompanhada da WT-AUC. O resumo de precisão-revocação utiliza average precision (AP), com implementação declarada; a denominação genérica PR-AUC não deve esconder a diferença entre AP e integração trapezoidal.

Curvas de calibração descrevem a relação entre probabilidade prevista e frequência observada. O número de bins e a regra de particionamento são fixados. ECE é secundário porque depende desses bins e pode ser instável na amostra pequena. A análise não se limita a acurácia com limiar 0,5, que pode ser pouco informativa para a política de alta retenção.

=== Bootstrap por tarefa e alcance dos intervalos

Para estimar incerteza, reamostram-se tarefas inteiras com reposição, preservando todas as suas tentativas, scores e resultados de políticas. Usam-se os mesmos índices reamostrados para comparar os métodos. Em cada réplica, recomputam-se as métricas e as diferenças. O número utilizado é 2.000 réplicas com seed fixada, produzindo intervalos percentílicos de 95%.

Na WT-AUC, reamostrar apenas as tarefas mistas estima a incerteza condicional a esse subconjunto; nas métricas de resolução e custo, reamostra-se todo o conjunto principal. A opção e seu denominador são explicitados. Réplicas sem classes necessárias não recebem valor zero artificialmente; são contadas como indefinidas.

O bootstrap dos scores externos já calculados trata os ajustes como fixos. Como os modelos dos folds compartilham dados de treino, esses intervalos são aproximações para a variabilidade entre tarefas e não capturam integralmente a instabilidade de retreinar todo o procedimento. A incerteza reportada é condicional aos modelos ajustados. Uma análise de repetição agrupada de todo o ajuste exigiria dimensionamento próprio e não foi realizada.

=== Permutações: hipótese nula e validade

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

=== Efeitos fixos como análise complementar

A regressão abaixo é uma formulação conceitual complementar; não foi ajustada
como novo estimador neste estudo. A análise intratarefa efetivamente executada
usa WT-AUC e os contrastes pareados definidos no protocolo.

Uma regressão com intercepto por tarefa formaliza o controle da informação constante no prompt:

$ "logit" P(S_(i j)=1) = alpha_i + beta_B^top B_(i j t)+beta_z z_(i j t). $

A estimação logística condicional pode eliminar os interceptos ao condicionar no número de sucessos de cada tarefa. Tarefas sem variação de rótulo não identificam $beta_z$. Colunas constantes dentro da tarefa, como comprimento do prompt, são absorvidas pelo efeito fixo e não precisam ser estimadas separadamente.

Essa regressão é associativa e complementar. Interceptos estimados com os rótulos das tarefas externas não constituem um preditor válido para tarefas novas. Portanto, não se usa sua log loss ajustada como melhoria fora da amostra. A comparação preditiva vem dos modelos B e H+B avaliados externamente. A incerteza de um coeficiente calculado sobre scores aprendidos deve reconhecer que esses scores também foram estimados; um erro-padrão convencional não representa automaticamente toda a cadeia de aprendizado.

=== Controles negativos e diagnósticos

O controle determinístico de `prompt_last` deve ter WT-AUC igual a 0,5 quando há empates exatos. Uma direção aleatória normalizada avalia a facilidade de obter associação aparente em alta dimensão. Um probe treinado com rótulos permutados dentro das tarefas do treino verifica se a separação depende do alinhamento real entre representação e resultado.

A direção invertida não é um controle independente contra associação: para um score sem outras mudanças, $"AUC"(-z)=1-"AUC"(z)$ com a convenção de empates. Ela serve para verificar sinal e implementação. Também não se permite usar o comprimento final como controle de uma política precoce. Indicadores de formatação já presentes no prefixo podem ser diagnósticos de artefatos; seu ajuste deve respeitar os splits.

Um desempenho não significativo de um controle não demonstra automaticamente a validade do modelo principal. Os controles se somam a uma avaliação explícita de vazamento, amostra, hipótese nula e tamanho de efeito.

=== Multiplicidade e critérios de evidência

A posição de 32 tokens e a comparação H+B contra B definem o endpoint principal. O resultado positivo de especificidade incremental requer $Delta_W>0$, intervalo pareado de 95% inteiramente acima de zero e $Delta_L<0$ fora da amostra. O intervalo de $Delta_L$ também é reportado; se ele incluir zero, a evidência probabilística é qualificada como incerta. O teste de permutação apoia a análise de existência de sinal intratarefa, sem funcionar como teste do ganho condicional.

Curvas por posição, comparações de camadas, tipos de erro, direções geométricas e subgrupos permaneceram secundários. Seus intervalos pontuais não têm cobertura simultânea e não foram usados para selecionar uma confirmação posterior. A correção Bonferroni descrita na permutação combina os folds daquele teste; não cobre toda a extensão exploratória. Não se escolheu outro checkpoint para substituir o resultado em 32 tokens.

Não rejeitar uma hipótese nula não demonstra equivalência. Um intervalo amplo pode conter tanto ausência de efeito quanto um efeito relevante. Com poucas tarefas mistas, a conclusão apropriada pode ser “inconclusivo neste orçamento”, e não “o modelo não contém o sinal”.

== Política de interrupção precoce e replay

=== Regra de decisão e casos de término

A política utiliza um único checkpoint $t^*$ por ajuste, evitando a multiplicação de decisões dentro da mesma tentativa. Se o candidato permanece ativo nesse ponto e seu score é inferior ao limiar $q$, a geração é interrompida. Caso contrário, continua normalmente até terminar. Não há steering nem alteração da distribuição dos tokens nos candidatos mantidos.

O algoritmo por tarefa é:

+ Iniciar a tentativa seguinte com sua seed pré-atribuída.
+ Gerar até o término natural, o checkpoint ou o limite, conforme o que ocorrer primeiro.
+ Se houve término, extrair o programa, verificar e parar a tarefa se houver sucesso.
+ Se a geração segue ativa no checkpoint, calcular os preditores e o score.
+ Interromper se o score for menor que $q$; caso contrário, completar e verificar.
+ Repetir até um sucesso verificado ou até atingir $N$ tentativas iniciadas.

Uma tentativa abortada consome orçamento de tentativa. Um candidato que termina exatamente em $t^*$ vai ao verificador antes da decisão de aborto. A regra com $q$ abaixo de todos os scores representa “não interromper” e deve sempre estar incluída na seleção.

=== Seleção do checkpoint e do limiar

Dentro do treino externo, avaliam-se $t in {8,16,32,64}$ e limiares derivados exclusivamente dos scores de desenvolvimento. O modelo de cada posição é ajustado sem usar o teste externo. A métrica de retenção relevante para o gate é

$ hat(r)_"ativo" (t,q)=
(sum_(i,j) S_(i j) bold(1){T_(i j)>t} bold(1){hat(p)_(i j t)>=q}) /
(sum_(i,j) S_(i j) bold(1){T_(i j)>t}). $

Essa taxa evita que muitos sucessos curtos, nunca expostos ao gate, escondam perdas entre candidatos efetivamente avaliados. Reporta-se também a retenção global, contando sucessos que terminaram antes do checkpoint como mantidos. Se não há sucessos ativos suficientes para avaliar uma configuração, ela não é calibrável e a opção conservadora é não interromper.

A meta de desenvolvimento é $hat(r)_"ativo">=0.97$. Todos os pares checkpoint-limiar que atendem à meta são avaliados por replay de custo médio por tarefa. Escolhe-se o menor custo; empates numéricos são resolvidos por maior resolução no desenvolvimento, checkpoint mais cedo e menor limiar. Não se escolhe automaticamente o maior limiar admissível, pois abortar sucessos pode aumentar o custo até solução.

Para evitar impor um limiar pior que o baseline, a opção sem aborto participa da mesma seleção. O protocolo registra uma tolerância numérica para empates e uma grade finita de limiares, incluindo os valores nos quais a decisão muda. A política e seu calibrador ficam congelados antes da avaliação externa.

=== Meta empírica de recall e generalização

A taxa de 97% é uma meta de ajuste. O recall no teste deve ser medido, e pode ser menor. Dependência dentro da tarefa, seleção do melhor limiar e mudança de distribuição impedem atribuir automaticamente uma garantia binomial simples à taxa agregada.

#cite(<ruan2026>, form: "prose") distinguem ajuste e certificação independente em seu contexto de agentes. Esta proposta adota apenas uma restrição empírica de seleção e uma avaliação externa; não reproduz a garantia formal daquele método. Além disso, retenção de trajetórias corretas não é igual a preservação da resolução das tarefas. Perder a única solução de um problema tem efeito diferente de perder uma entre várias soluções corretas.

=== Replay com candidatos pareados

O banco guarda a completion completa de cada tentativa. Para simular aborto, o replay consulta apenas features disponíveis no checkpoint e usa o resultado completo exclusivamente para avaliar o contrafactual de retenção: saber se o candidato descartado passaria. A política não acessa esse rótulo para decidir.

Defina $b_(i j)=bold(1){T_(i j)>t^*,hat(p)_(i j t^*)<q}$. O custo de uma tentativa iniciada no replay é

$ d_(i j)^pi = b_(i j)t^*+(1-b_(i j))T_(i j). $

Se $K_i^pi$ é a primeira tentativa não abortada que passa, então

$ C_i^pi (N)=sum_(j=1)^N d_(i j)^pi bold(1){K_i^pi>=j}, $
$ A_i^pi (N)=bold(1){K_i^pi<=N}. $

É essencial somar apenas tentativas efetivamente iniciadas. O rótulo de um candidato abortado pode ser usado para calcular false-abort na análise, mas não para parar o replay como se a tarefa tivesse sido resolvida.

A equivalência com a execução online exige que abortar uma tentativa não modifique a identidade das seguintes. Seeds independentes, ordem fixa e configuração determinística fornecem essa propriedade no desenho. Batching e diferenças numéricas podem impedir igualdade bit a bit; por isso, uma execução online pequena deve conferir decisões, contagem de tokens e comportamento de término.

=== Comparadores operacionais

O experimento utiliza o mesmo banco e a mesma ordem para quatro famílias: geração fixa de N candidatos completos; geração sequencial com verificador; interrupção com observáveis B; e interrupção com H+B. Os orçamentos reportados são 1, 5, 10 e 20, com $N=20$ como comparação principal.

Cada política adaptativa recebe o mesmo espaço de checkpoints, método de calibração e regra de seleção. Uma ablação secundária pode fixar o mesmo checkpoint para B e H+B, isolando a contribuição do score. A comparação principal entre procedimentos permite que cada um selecione sua configuração dentro do mesmo orçamento de desenvolvimento.

O baseline fixo e o baseline com verificador têm a mesma cobertura potencial no mesmo conjunto de candidatos, mas custos distintos quando há sucessos precoces. O comparador decisivo da economia é o baseline com verificador, porque ele já evita gerar tentativas depois de resolver a tarefa.

=== Não inferioridade e economia

Definam-se as diferenças médias pareadas

$ Delta_A=1/m sum_i (A_i^pi-A_i^V), quad Delta_C=1/m sum_i (C_i^pi-C_i^V). $

A margem planejada de não inferioridade é 0,03 em proporção, isto é, 3 pontos percentuais. Exige-se que o limite inferior do intervalo de 95% de $Delta_A$ seja maior que $-0.03$, e que o limite superior do intervalo de $Delta_C$ seja menor que zero. Usar um intervalo bilateral de 95% para essas condições é uma escolha conservadora em relação a um teste unilateral de 5%.

A economia relativa é $1-overline(C)^pi/overline(C)^V$, com intervalo pareado. A margem de resolução não deriva automaticamente do recall de 97%; são critérios distintos. Com aproximadamente 132 tarefas, a perda de uma tarefa equivale a cerca de 0,76 ponto percentual, o que limita a resolução estatística de uma margem estreita. A discussão de potência deve considerar tarefas e discordâncias pareadas, não apenas o número total de trajetórias.

A comparação contra a política observável usa as mesmas métricas. Economizar em relação ao verificador, mas não superar B, sustenta o valor da interrupção naquele regime, sem demonstrar que os estados internos foram necessários para o ganho.

=== Medição online e amortização

A execução online mede tempo após aquecimento, sincronização apropriada da GPU e condições comparáveis de carga e batch. Reporta geração, extração/score, verificação e total. A política não deve ser creditada por economia de FLOPs ou energia sem medição correspondente.

O custo de pesquisa inclui gerar o banco completo, extrair ativações, treinar e calibrar. Esses custos não pertencem ao custo marginal por tarefa da política já treinada, mas precisam ser informados. Se o custo de preparação for $F$ e a economia média por uso for $s>0$ na mesma unidade, o ponto de amortização idealizado é $F/s$ utilizações. Sem estimativas compatíveis, não se afirma retorno econômico total.

== Ambiente efetivamente validado

O dispositivo é uma NVIDIA GeForce RTX 3050 6GB Laptop GPU. A geração usa pesos
em bfloat16, projeções lineares calculadas em float32, atenção eager e uma tentativa
por vez, com TF32 desativado. O armazenamento dos vetores selecionados usa float16.
Essa combinação separa precisão dos pesos, computação e armazenamento; descrevê-la
apenas como “inferência FP16” seria inexato. As tentativas técnicas rejeitadas não
forneceram rótulos ao banco. Versões e hashes estão nos manifestos.

== Transferência externa implementada

O fluxo secundário prepara modelos, calibradores e políticas exclusivamente com
HumanEval+, congela o bundle e então permite coletar MBPP+. Foram avaliadas 32
tarefas selecionadas por hash, com 20 tentativas cada. A aplicação não ajusta
parâmetros com rótulos do benchmark alvo. Recalibração ou retreino em MBPP+
constituiriam outro experimento. Os resultados da transferência congelada são reportados no capítulo empírico,
sem retreino, recalibração ou seleção nos rótulos alvo.
