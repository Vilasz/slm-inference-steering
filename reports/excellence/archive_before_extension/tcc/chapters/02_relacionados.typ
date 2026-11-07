#import "../lib.typ": *

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

As referências são identificadas por fontes primárias, com versão explícita quando fixada na bibliografia. Resultados resumidos não substituem a leitura integral dos métodos antes da comparação final. Em especial, alegações de ausência de trabalhos idênticos exigem revisão bibliográfica mais ampla; não são inferidas de uma busca limitada.

== Matriz de posicionamento

#set text(size: 9pt)
#table(columns: (1.1fr, 1.4fr, 1.5fr),
 table.header([*Antecedente*], [*Objeto principal*], [*Relação com este estudo*]),
 [Di Cicco], [Corretude da primeira tentativa a partir do prompt], [Motiva separar dificuldade e trajetória],
 [Klassert], [Incerteza introspectiva em respostas e linhas de código], [Motiva o probe simples e limites da generalização],
 [Zhu et al.], [Dificuldade percebida e alocação de inferência], [Comparação no nível da tarefa],
 [Ruan et al.], [Abortos em episódios de agentes], [Retenção certificada difere de meta empírica],
 [Wang et al. / ST-BoN], [Consistência interna entre amostras e truncamento], [Uso de estados internos para economia já possui antecedentes],
 [Este estudo], [Trajetórias da mesma tarefa, observáveis e replay pareado], [Avaliação delimitada; evidência precoce ainda a produzir],
)
#set text(size: 11pt)
