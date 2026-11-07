#import "../lib.typ": *

= Discussão e limitações

#include "../generated/discussion_findings.typ"

== Validade interna

As partições agrupadas, escolhas internas, reconstrução dos scores e testes de prefixo foram auditados. Esses controles sustentam contratos específicos de isolamento e temporalidade. Não demonstram ausência de adaptação humana ao teste, de toda similaridade entre tarefas ou de contaminação no pré-treinamento. As escolhas múltiplas posteriores permanecem exploratórias, e a retenção de desenvolvimento não foi certificada independentemente.

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
