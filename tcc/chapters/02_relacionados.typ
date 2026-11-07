#import "../lib.typ": *

= Trabalhos relacionados e posicionamento

== Amostragem repetida e verificação funcional

#cite(<chen2021codex>, form: "prose") formulam a avaliação de programas gerados por meio de testes e apresentam um estimador de pass\@k baseado em múltiplos candidatos. #cite(<brown2024>, form: "prose") estudam o aumento de cobertura obtido com mais amostras e destacam a relevância da seleção e verificação. Esses trabalhos motivam a separação entre a capacidade de gerar ao menos uma solução correta e a capacidade de identificá-la.

A qualidade do verificador também importa. #cite(<liu2023evalplus>, form: "prose") apresentam EvalPlus e a ampliação dos testes de HumanEval para HumanEval+, mostrando que testes mais completos revelam falhas que os testes originais não identificam. Neste projeto, a variável resposta é passar em uma versão fixada dessa suíte. Isso fornece um critério reproduzível, mas não uma prova de correção para todas as entradas possíveis.

== Representações internas e estimação de corretude

#cite(<kadavath2022>, form: "prose") diferenciam a avaliação de uma resposta específica da estimativa de que o modelo saberá responder uma pergunta. Essa distinção é próxima da separação entre nível da tarefa e nível da trajetória adotada aqui. #cite(<azaria2023>, form: "prose") estudam classificadores sobre ativações internas para identificar veracidade de afirmações, comparando-os também a medidas baseadas na probabilidade da sequência. A abordagem de Representation Engineering organiza a investigação de fenômenos de alto nível em padrões distribuídos das ativações @zou2023.

Em geração de código, #cite(<dicicco2026>, form: "prose") reporta decodificação linear de corretude a partir do estado do último token do prompt em Qwen3-4B-Instruct-2507 sobre tarefas de LiveCodeBench. O resultado demonstra a relevância de estados anteriores à geração para discriminação entre tarefas. Por construção, porém, um estado determinístico do mesmo prompt é compartilhado pelas suas diferentes tentativas. Essa observação motiva uma avaliação adicional entre trajetórias do mesmo problema.

#cite(<klassert2026>, form: "prose") investiga estimação de incerteza introspectiva em geração de código, com características estáticas e dinâmicas e avaliações no nível da resposta e da linha. O resumo do trabalho reporta sinal de corretude no nível da resposta, sem ganhos consistentes das estratégias dinâmicas mais elaboradas. Isso reforça a opção por um probe simples como experimento principal; não justifica assumir que uma dinâmica temporal mais complexa será superior.

#cite(<rathnasuriya2026>, form: "prose") examinam sobreconfiança em quatro modelos
de código e três benchmarks de execução. Confiança e entropia oferecem
discriminação parcial e dependente do modelo; seleção por incerteza não melhora
consistentemente a acurácia do conjunto aceito. A análise latente é exploratória.
Esse antecedente recente aproxima a literatura da comparação entre observáveis
e estados internos, mas não estabelece o estimando intratarefa nem o custo
sequencial dos gates específicos avaliados neste TCC. Resultados de modelos de
7–8 bilhões de parâmetros não são transferidos numericamente aos dois SLMs de 1,3 e 1,5 bilhão avaliados aqui.

== Duas famílias de geradores de código

Qwen2.5-Coder @hui2024 e DeepSeek-Coder @guo2024deepseek são famílias de modelos especializados em programação. Seus relatórios fundamentam a escolha de geradores de pesos públicos e portes próximos, mas não respondem à pergunta deste estudo: obter bom desempenho funcional em um benchmark é diferente de oferecer um estado precoce que permita ordenar as próprias tentativas corretas e incorretas. Aqui se utilizam as variantes de aproximadamente 1,5 e 1,3 bilhão de parâmetros especificadas na metodologia.

Arquitetura, tokenizer e template fazem parte do regime de geração. Não se procura alinhar coordenadas latentes ou comparar entropias como se os vocabulários fossem os mesmos. Cada gerador tem seu próprio contraste entre $H+B$ e $B$. As tarefas externas comuns tornam a comparação mais informativa, mas não igualam automaticamente as populações que permanecem ativas em 32 tokens.

== Dificuldade e interrupção adaptativa

#cite(<zhu2025>, form: "prose") estimam dificuldade percebida pelo modelo a partir de representações iniciais e aplicam essa informação a estratégias de inferência adaptativa. #cite(<ruan2026>, form: "prose") estudam interrupção de episódios de agentes a partir de probes em estados internos, com controle de retenção e certificação em dados independentes. Seu cenário, unidade temporal e procedimento de garantia diferem de uma regra empírica ajustada em prefixos de código.

#let stbon-ok = true
#if stbon-ok [O método ST-BoN @stbon2025 utiliza estados internos em uma estratégia de truncamento e seleção de candidatos. Esse antecedente impede apresentar o uso de hidden states para reduzir o custo de Best-of-N como novidade isolada. A comparação do presente estudo concentra-se na corretude funcional supervisionada, na discriminação dentro da tarefa e na comparação controlada com observáveis.]

== Complexidade do probe e transporte de probabilidades

#cite(<hewitt2019>, form: "prose") mostram que o desempenho de um probe pode
refletir sua capacidade de aprender a tarefa, motivando controles e restrições
de complexidade. Seus controles linguísticos diferem da permutação de rótulos
intratarefa utilizada aqui. As ablações com PCA, penalização e offset residual
avaliam a sensibilidade à parametrização; não identificam, sozinhas, um mecanismo
semântico ou a ausência de informação acessível a outros decodificadores.

#cite(<ovadia2019>, form: "prose") avaliam incerteza sob mudanças de distribuição
e documentam limitações do transporte da calibração. Isso motiva avaliar o gate
congelado por suas decisões, além do ranking. O presente estudo distingue
escores OOF da origem, escores de desenvolvimento usados na seleção do gate e
aplicação de um modelo ajustado em toda a origem ao novo benchmark. Essa
distinção impede atribuir toda diferença a um único efeito de domínio.

Na comparação entre classes de probes, uma rede não linear pode expressar interações que a logística não representa diretamente, mas também acrescenta parâmetros e variabilidade de ajuste. Esse aumento de capacidade precisa ser oferecido ao baseline $B$. Por isso, a segunda etapa cruza logística e MLP com $B$ e $H+B$, mantendo a redução de dimensão latente e as regras de validação correspondentes. O desenho examina acessibilidade a classes restritas, sem presumir superioridade do MLP ou identificar, por si só, um mecanismo semântico.

== Interrupção como decisão sequencial

A escolha de continuar uma computação tem antecedentes em metarraciocínio. #cite(<russell1991>, form: "prose") formulam a avaliação de ações computacionais por sua utilidade esperada; #cite(<callaway2018>, form: "prose") estudam como aprender a selecionar computações considerando custo e valor esperado. No contexto de modelos de linguagem, #cite(<wan2025beacon>, form: "prose") formulam uma regra de parada para a busca por respostas completas. Esses antecedentes impedem apresentar o controle de inferência por programação dinâmica como novidade isolada.

O objeto específico deste estudo é a decisão sobre uma tentativa parcialmente gerada, com possibilidade de abandoná-la, iniciar outra ou encerrar a busca. A recursão de Bellman apresentada adiante organiza essas escolhas e explicita as hipóteses sob as quais um limiar de probabilidade poderia ser ótimo. Ela não transforma os gates empíricos em políticas ótimas ou certificadas.

== Contribuição delimitada e limites da reivindicação

O trabalho não reivindica ter sido o primeiro a prever corretude com hidden states, o primeiro a estimar dificuldade nem o primeiro a interromper inferência com sinais internos. A contribuição é uma avaliação integrada e reproduzível, em um regime delimitado de geração de código, de três aspectos: separação entre tarefa e trajetória; poder incremental sobre baselines observáveis especificados; e utilidade em uma política sequencial de interrupção. A extensão exploratória aprofunda o resultado negativo com controles de parametrização, diagnóstico dos sucessos descartados e um limite econômico contrafactual. A derivação do custo por série geométrica e os métodos de PCA e validação não são apresentados como novidades teóricas.

As referências são identificadas por fontes primárias, com versão explícita quando fixada na bibliografia. Resultados resumidos não substituem a leitura integral dos métodos antes da comparação final. Em especial, alegações de ausência de trabalhos idênticos exigem revisão bibliográfica mais ampla; não são inferidas de uma busca limitada.

== Matriz de posicionamento

#set text(size: 10pt)
#set par(leading: 0.34em)
#figure(kind: "quadro", supplement: [Quadro],
 caption: [Posicionamento sintético em relação aos antecedentes discutidos na revisão bibliográfica.])[
#table(columns: (1.1fr, 1.4fr, 1.5fr),
 table.header([*Antecedente*], [*Objeto principal*], [*Relação com este estudo*]),
 [Di Cicco], [Corretude da primeira tentativa a partir do prompt], [Motiva separar dificuldade e trajetória],
 [Klassert], [Incerteza introspectiva em respostas e linhas de código], [Motiva o probe simples e limites da generalização],
 [Rathnasuriya e Yang], [Sobreconfiança, seleção e sinais latentes exploratórios], [Exige separar ranking, calibração, retenção e custo sequencial],
 [Zhu et al.], [Dificuldade percebida e alocação de inferência], [Comparação no nível da tarefa],
 [Ruan et al.], [Abortos em episódios de agentes], [Retenção certificada difere de meta empírica],
 [Wang et al. / ST-BoN], [Consistência interna entre amostras e truncamento], [Uso de estados internos para economia já possui antecedentes],
 [Este estudo], [Duas famílias de SLMs, logística/MLP e replay pareado], [Incremento sobre observáveis, transporte das decisões e limites da utilidade econômica],
)
]
#set text(size: 12pt)
#set par(leading: 0.84em)
