# Defesa: roteiro e perguntas difíceis

## Roteiro de 18–20 minutos

| Slide | Tempo | Mensagem que precisa ficar clara |
|---|---:|---|
| 1. Pergunta | 1 min | Prever corretude, acrescentar informação e economizar são problemas distintos. |
| 2. Tarefa versus trajetória | 1,5 min | Um prompt difícil não identifica qual amostra do mesmo prompt terá sucesso. |
| 3. Desenho e história | 1,5 min | Piloto separado, 132 tarefas principais, 32 externas; exposição e histórico de autorização divulgados. |
| 4. Validação | 2 min | Agrupamento, seleção interna e cross-fitting de toda a seleção para calibração. |
| 5. Resultado original | 2 min | H+B não supera B; log loss aumenta. Resultado do algoritmo e regime delimitados. |
| 6. Ablações | 2 min | NLL/margem têm ranking útil; PCA/residual reduzem a desvantagem, sem confirmar superioridade. |
| 7. Transferência | 2 min | Limiar de alta retenção no desenvolvimento não é garantia; distinguir três avaliações. |
| 8. Candidatos versus tarefas | 1,5 min | 90 corretos descartados, duas tarefas perdidas, denominadores diferentes. |
| 9. Economia | 2 min | E[C]=E[D] soma geométrica; retenção e overhead precisam entrar. |
| 10. Oracle | 1,5 min | Espaço econômico idealizado existe, mas exige prever descartes sem rótulos finais. |
| 11. Limites e reprodução | 1 min | Um modelo, tarefas conhecidas, intervalos condicionais, tempos descritivos. |
| 12. Avaliação independente e conclusão | 1 min | 120 tarefas novas; endpoint de retenção e suas limitações, sem substituir Q2/H3. |

Antes de apresentar: conferir o PDF final, ensaiar a derivação sem ler, reproduzir
um exemplo de replay à mão e saber localizar cada número no JSON ou CSV correspondente.
O objetivo é defender a evidência, não prometer nota, novidade universal ou mecanismo causal.

## Perguntas e respostas

### 1. Por que escolher WT-AUC?

Ela compara candidatos corretos e incorretos da mesma tarefa, atribui meio ponto
ao empate e depois dá peso igual às tarefas mistas. Responde se o score distingue
trajetórias além de simplesmente ordenar dificuldade de prompts. Sua população é
condicional às tarefas que apresentam ambos os resultados no risco e orçamento usados.

### 2. Por que AUC agregada é insuficiente?

Um score constante por prompt pode separar tarefas fáceis e difíceis e ter AUC
agregada alta, mantendo WT-AUC igual a 0,5. Ambas são reportadas porque respondem
perguntas diferentes; uma não é uma versão universalmente melhor da outra.

### 3. O que acontece com tarefas só corretas ou só incorretas?

Sua AUC intratarefa não é identificada; não atribuímos 0,5 artificialmente.
Elas continuam nas perdas, calibração e resolução. O número de mistas muda com
o checkpoint e com a quantidade de tentativas, e é sempre divulgado.

### 4. Por que H+B pode perder para B se contém todas as features de B?

A inclusão das variáveis amplia a classe representacional, mas não garante menor
risco fora da amostra de um estimador regularizado e selecionado com dados finitos.
Dimensão, penalização conjunta e calibração mudam o ajuste. PCA e residual reduziram
a desvantagem observada, sem identificar uma causa única ou demonstrar ganho sobre B.

### 5. Por que regressão logística?

Ela fornece uma leitura linear controlável, probabilidades e uma função de perda
própria, com regularização explícita. A pergunta é acessibilidade a essa classe,
não uma busca por qualquer decodificador capaz de extrair toda informação possível.
Classificadores mais complexos exigiriam outro orçamento e validação.

### 6. Por que validação aninhada?

Escolher C ou camada pelo teste favorece resultados ocasionais. Os folds internos
escolhem essas opções no treino externo; o teste externo avalia o procedimento
completo. O calibrador recebe scores cujo próprio procedimento de seleção foi cross-fitted.

### 7. Como evitou leakage?

Separando tarefas, refazendo imputation, escala e PCA no treino de cada split e
confinando seleção e calibração ao desenvolvimento. O offset residual também
exclui o rótulo pontuado na seleção e calibração do primeiro estágio. Auditoria de
IDs e testes sustentam isso, mas não provam ausência de adaptação humana ou contaminação de pré-treino.

### 8. Por que não dividir tentativas aleatoriamente?

O mesmo prompt apareceria nos dois lados, permitindo reconhecer a tarefa e
produzindo uma avaliação inadequada para generalização a tarefas excluídas do treino.
As vinte tentativas do mesmo problema permanecem juntas no fold externo.

### 9. Por que bootstrap por tarefa?

Tentativas compartilham prompt e dificuldade. Reamostrar candidatos como se fossem
independentes subestimaria a variabilidade entre tarefas. Formamos contribuições
pareadas por tarefa ou reamostramos suas contagens conjuntamente para razões de retenção.

### 10. O bootstrap cobre a incerteza de treinamento inteira?

Não. Mantém modelos e escores OOF fixos e não refaz sua seleção/ajuste em cada réplica.
Folds compartilham dados de treino. Os intervalos são condicionais e sua limitação
é explícita; uma nova avaliação independente com scorers congelados é uma extensão distinta.

### 11. Como interpretar o intervalo −0,099 a 0,006?

Ele contém zero e uma faixa de efeitos sob as suposições do procedimento. Não
demonstra equivalência nem ausência universal de sinal. O efeito pontual é negativo
e o limite positivo é pequeno nesta análise; a perda também piora no contraste principal.

### 12. Qual foi exatamente o critério original Q2?

ΔWT-AUC positivo com IC95% inteiramente positivo, mais redução pontual de log loss.
O intervalo da perda qualifica sua incerteza. Uma tabela complementar anterior foi
corrigida porque exigia IC de perda inteiramente negativo; a regra do protocolo e o veredito original foram preservados.

### 13. A permutação testa informação adicional de H sobre B?

Não. A permutação intratarefa com scorer fixo testa associação sob intercambialidade
de rótulos. A comparação incremental depende dos contrastes pareados contra B.
O procedimento por fold e Bonferroni não transforma associação em um teste condicional de incremento.

### 14. Por que o resultado negativo ainda é útil?

Ele delimita uma hipótese operacional: esse procedimento com esse modelo, dados
e baseline não demonstrou vantagem. As ablações mostram dependência da parametrização;
o diagnóstico do gate mostra limitações de transportar recall empírico, e o oracle
separa oportunidade econômica de capacidade preditiva efetivamente disponível.

### 15. Qual é a contribuição original, sem exagerar?

Uma avaliação integrada, rastreável e delimitada que separa dificuldade e trajetória,
ganho sobre observáveis e utilidade sequencial, aprofundando um resultado negativo
com controles de leitura, transporte e limite econômico. Não reivindico inventar
probing, PCA, interrupção por estados internos, bootstrap ou uma nova soma geométrica.

### 16. PCA provou que a dimensão era a causa?

Não. PCA altera dimensão, coordenadas e penalização efetiva; whitening também
muda a parametrização. Sua melhora contra o H+B original é evidência de sensibilidade
do procedimento, mas não isola uma causa única nem comprova superioridade sobre B.

### 17. Como funciona a correção residual?

Usa logit(p_B) como offset fixo e ajusta uma correção linear penalizada em H.
Para treinar essa correção, cada p_B é produzido sem seu rótulo, inclusive na
seleção e calibração de B. A seleção residual repete essa separação em cada treino interno.
As probabilidades usam a ligação logística nativa, sem um segundo Platt.

### 18. Correlação baixa entre H e B implica informação nova?

Não. Um score pode ser pouco correlacionado e ainda carregar ruído sem utilidade
para o rótulo. Correlação, redundância representacional e valor incremental
preditivo são conceitos distintos. O teste operacional continua sendo generalização pareada sobre corretude.

### 19. Como explicar a perda de retenção em MBPP+?

Os corretos no alvo recebem muitos scores abaixo do limiar de origem; o ranking
H+B em t=8 também é fraco. Há diferenças de distribuição, prevalência e comprimento,
mas origem OOF, desenvolvimento e modelo completo no alvo diferem como procedimentos.
Não identifiquei causalmente um único efeito de domínio ou de calibração.

### 20. Por que 97,87% na origem não é comparável diretamente a 75,41%?

O primeiro número agrega políticas e checkpoints escolhidos em folds externos;
o segundo é um único bundle ajustado em toda a origem, com gate em 8 tokens.
O diagnóstico explicita essas avaliações e a proposta nova compara B/H+B no mesmo checkpoint.

### 21. Retenção de 75,41% significa perder 24,59% das tarefas?

Não. É uma proporção de candidatos corretos ativos. Uma tarefa com vários sucessos
pode continuar resolvida depois de alguns descartes; perder o único sucesso é diferente.
No banco MBPP+ observado, duas tarefas perdem resolução: Mbpp/166 e Mbpp/259.

### 22. Por que 90/366 e 23/48 descrevem perdas diferentes?

A primeira razão usa todos os corretos ativos potenciais do banco. A segunda
condiciona nos corretos ativos efetivamente iniciados até sucesso ou esgotamento.
O próprio gate modifica quais tentativas seguintes são iniciadas, mudando a população.

### 23. Pode escolher outro limiar depois de ver MBPP+?

Pode apresentar uma curva retrospectiva, identificada como diagnóstico. Escolher
um ponto com esses rótulos e chamá-lo de transferência congelada seria incorreto.
A validação proposta preserva limiares ajustados apenas em HumanEval+.

### 24. Como deriva o custo esperado?

Condicionado à tarefa, a tentativa j começa se as anteriores não tiveram sucesso
mantido, com probabilidade (1−pr)^(j−1). O custo da tentativa atual é independente
desse evento quando os pares custo/sucesso são iid entre tentativas. Logo,
E[C(N)]=E[D] soma de j=0 até N−1 de (1−pr)^j. Custo e sucesso da mesma tentativa podem depender.

### 25. Qual a condição econômica com orçamento ilimitado?

Para p>0, r>0 e custos médios finitos: gate custa E[D]/(pr) e verificador E[T]/p.
O gate reduz custo precisamente quando E[D]/E[T]<r. r é retenção global dos
corretos, incluindo os que terminam antes do gate; não se substitui automaticamente por recall ativo.

### 26. O que muda para N finito?

Com G_N(s)=soma geométrica, a condição de custo é E[D]/E[T]<G_N(p)/G_N(pr),
e a cobertura do gate é 1−(1−pr)^N. Menor custo pode refletir abandono de sucessos,
por isso resolução e custo precisam ser avaliados juntos. As fórmulas condicionam na tarefa; médias agregadas não eliminam heterogeneidade.

### 27. O oracle é um limite inferior absoluto?

Não. Ele minimiza custo no checkpoint fixo e classe que preserva todos os
candidatos corretos na mesma ordem. Um gate que elimina sucessos pode custar
menos em alguns caminhos, perdendo cobertura ou encontrando depois sucesso mais curto.
O oracle conhece os rótulos finais e não é implementável online.

### 28. Economizar tokens demonstra menor latência?

Não. Prefill, processamento do token no gate, transferência GPU–CPU, probe e
verificação têm custos diferentes. O benchmark técnico separa componentes e mostra
igualdade numérica, mas seus tempos são descritivos e não demonstram aceleração da política ou economia de energia.

### 29. O que mudaria com modelos maiores ou outra GPU?

Não é determinado por este estudo. Representações, precisão, baseline, custo e
calibração podem mudar. A GPU detectada é RTX3050 Laptop 6GB; nenhuma medição
RTX4060 ou modelo maior foi fabricada. Seria necessário novo preflight e desenho próprio.

### 30. O que faria diferente no próximo estudo?

Reservaria tarefas independentes antes de explorar, separaria seleção e avaliação
de políticas, faria planejamento por tarefas e discordâncias e definiria o papel
dos testes funcionais na produção. A proposta atual testa transporte em 120
tarefas adicionais, com piloto técnico disjunto, sem escolher o melhor classificador exploratório.

### 31. O que os testes automatizados não demonstram?

Não demonstram ausência de contaminação do pré-treinamento, de decisões humanas
adaptadas aos dados, de similaridade semântica entre tasks ou de toda variação numérica
futura. Verificam contratos específicos: IDs, sufixos, estatística, contabilidade,
cross-fitting, aprovação e persistência. Seu escopo precisa ser explicitado.

### 32. O verificador é uma prova formal ou um sandbox completo?

Não é nenhum dos dois. Aprovação é relativa às suites e limites fixados.
O reliability_guard do EvalPlus não constitui um sandbox de segurança; o
manuscrito foi corrigido para descrever os subprocessos e limites efetivos,
sem prometer isolamento de rede e credenciais que não foi demonstrado.

### 33. Quais condições precederam a nova coleta?

Aprovação do estudante e revisão do orientador sobre o protocolo concreto,
com hash e snapshot da versão aprovada, além dos gates técnicos. O planejamento
estima 4,44 horas locais, ou 6,66 com margem de 50%, para 400 tentativas técnicas
e 2.400 principais. A aprovação explícita e a revisão informada pelo estudante foram registradas antes do congelamento e das novas trajetórias.

### 34. A probabilidade verdadeira de sucesso deve crescer a cada token?

Não. Para a filtração dos tokens, a probabilidade condicional verdadeira de um
mesmo evento final é um martingale pela propriedade da torre. Ela pode subir ou
cair numa trajetória; o valor esperado condicional é que se conserva. Um probe
ajustado não é necessariamente esse processo. Comparar AUCs de populações T>t
diferentes tampouco testa a propriedade de martingale. Uma ação de aborto muda
o evento operacional de sucesso; não permite aplicar esse argumento sem cuidado.

### 35. A covariância total decompõe a AUC?

A identidade separa covariância entre médias condicionais por tarefa e covariância
média dentro da tarefa. Ela explica por que dificuldade pode induzir associação
agregada. AUC é uma probabilidade de ordenação, não uma covariância; sua decomposição
não é essa identidade. WT-AUC define outro estimando: média das AUCs de tarefas
mistas, com empates valendo meio e igual peso por tarefa.

### 36. Como um hidden state determinístico poderia adicionar informação?

Com pesos fixos e prefixo completo conhecido, o estado é uma função determinística;
a informação mútua condicional adicional é zero. O baseline B usa um resumo finito
do prefixo e uma classe restrita de leitores. Ganho preditivo de H sobre esse B
pode existir por perda de informação no resumo ou por acessibilidade ao leitor.
Não significa criar informação nova em relação ao prefixo completo.

### 37. O bootstrap OOF mede toda a incerteza de treinamento?

Não. Reamostrar tarefas com seus escores OOF fixos preserva dependência entre
tentativas, mas condiciona nos scorers ajustados e não refaz todas as seleções.
Treinos sobrepostos também impedem tratar folds como réplicas independentes.
Na nova reserva, o gate está fixo antes das tarefas e o bootstrap por tarefa
responde à incerteza do desempenho desse gate, sob intercambiabilidade dos clusters.

### 38. Como a multiplicidade foi tratada?

O endpoint original e o único endpoint da nova reserva mantêm seus critérios.
Os demais checkpoints, famílias, curvas e exclusões são exploratórios, com
intervalos pontuais. Não se escolhe o melhor resultado deles como confirmação.
A Bonferroni da permutação combina os cinco testes por fold especificados;
ela não corrige automaticamente todas as comparações da extensão.

### 39. O que demonstra não inferioridade na resolução?

O protocolo usa margem absoluta de três pontos percentuais, definida previamente.
O limite inferior do intervalo pareado precisa satisfazer a regra de não
inferioridade em relação ao comparador. Um ponto estimado próximo de zero ou
um intervalo que inclui zero não basta. O benefício conjunto ainda exige
redução de custo contra verificador e B, suporte a H2 e validação online.

### 40. Seeds diferentes tornam as tarefas independentes?

Seeds reproduzem fluxos de amostragem e a auditoria detecta colisões de identidade.
Elas não removem similaridade entre problemas, contaminação de pré-treino nem
dependência induzida por um modelo comum. Tentativas ficam agrupadas por tarefa
na inferência estatística. Semelhança textual foi auditada sem observar novos
rótulos; agrupá-la posteriormente é uma sensibilidade, não alteração da reserva.

### 41. PCA e residual identificaram a causa do resultado negativo?

Não. Melhoram a estimativa pontual do procedimento testado e mostram sensibilidade
à leitura. PCA muda representação, escala e restrição efetiva; o residual ancora
B e muda a regularização. Seus intervalos contra B incluem zero. Dimensão,
otimização, ruído e ajuste probabilístico permanecem explicações compatíveis;
não há experimento que isole uma causa única nem confirmação do residual.

### 42. As 120 tarefas são um terceiro benchmark ou uma reserva sem qualquer exposição?

São novas tarefas do mesmo MBPP+, sem geração ou avaliação local anterior segundo
os artefatos auditados, não um terceiro benchmark. O desenho e IDs precedem a
coleta aprovada. A inspeção textual detalhada foi feita depois do início e
identificou dois pares semelhantes; não se excluíram tarefas nem se ajustaram
gates. Ausência de exposição local não certifica ausência no pré-treinamento.

<!-- independent-completed -->
### 43. Qual foi a nova validação independente?

Foram 20 tarefas técnicas disjuntas (400 tentativas) e 120 tarefas principais
MBPP+ sem resultados locais anteriores (2.400 tentativas), reservadas por hash.
O estudante aprovou e informou revisão do orientador antes do congelamento.
B e H+B de origem foram comparados no mesmo t=8, sem ajuste no alvo.
O único endpoint confirmatório é a diferença de razões de contagens de corretos
mantidos/ativos, H+B menos B: -0.16359060402684564; IC95% [-0.2402298061170319, -0.0942741120201171]. Critério direcional: **satisfeito**.
São 10.000 reamostragens pareadas por tarefa, seed 20261017; o suporte exige
limite superior do IC bilateral de 95% abaixo de zero. O piloto técnico não
entra nessas contagens. Tarefas sem acertos continuam no conjunto de clusters.

### 44. Esse resultado muda as hipóteses originais ou prova uma causa de domain shift?

Não muda Q2/H3 originais. O endpoint novo compara retenção de dois gates
específicos já congelados. Não testa superioridade universal de representações,
ganho incremental após novos ajustes, energia ou latência. Mesmo quando o critério
de retenção é satisfeito, a diferença não identifica uma causa de domain shift.
Há um único modelo e uma nova reserva MBPP+; o bootstrap pressupõe tarefas
intercambiáveis e a ausência de exposição local não certifica ausência no pré-treino.
As métricas de custo, resolução, calibração e outros orçamentos são secundárias.
