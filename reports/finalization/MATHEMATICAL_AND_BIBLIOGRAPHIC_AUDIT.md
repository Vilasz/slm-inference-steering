# Revisão matemática e bibliográfica

## Formulação e escopo

| Objeto | Conferência realizada | Resultado |
|---|---|---|
| WT-AUC | Contagem independente de pares positivo-negativo e peso 1/2 para empates; média uniforme por tarefa mista | Contrastes e IC reproduzidos; homogêneas não recebem AUC artificial |
| Covariância total | Condicionamento por tarefa X; separação de Cov(E[Z|X],E[S|X]) e E[Cov(Z,S|X)] | Identidade correta; texto não a apresenta como decomposição da AUC |
| Custo até sucesso | Custo da tentativa j multiplicado pelo indicador de chegar a j | Soma geométrica válida sob pares custo/desfecho iid entre tentativas, condicionalmente à tarefa; permite dependência custo/sucesso na mesma tentativa |
| Retenção | Separação entre retenção global e entre corretos ativos | r = 1 − a⁺(1 − r_ativo); o caso sem corretos ativos é explicitado |
| Oracle | Mesma ordem e checkpoint; mantém todos os corretos e interrompe incorretos ativos | Limite pathwise dentro dessa classe; não é limite universal de políticas que descartam sucessos |
| Bellman | Estado, tentativas restantes, custos, orçamento e fronteiras | Atualizações de recurso explicitadas no manuscrito; k mantém uma única convenção |
| Limiar pV ≥ c | Comparação c+(1−p)V com V | V é custo esperado de fallback, não probabilidade; redução exige o mesmo fallback após aborto e fracasso |

O solver sintético congelado foi confrontado com enumeração fechada de
políticas iid em 216 combinações finitas. Outras 27 comparações conferem a
regra simplificada. Os 231 testes específicos da extensão e recursos passaram.
Os registros estão em `statistical_verification.json` e `tests.xml`.

No manuscrito, J_k representa ausência de tentativa ativa, e F_k uma tentativa
ativa com k novas tentativas restantes. Iniciar reduz k; continuar e abortar
a atual não reduzem novamente esse índice. A formulação simplificada agora
usa V_k, mantendo a convenção. O documento conceitual histórico permanece
intacto e declara sua mudança local de indexação para V_(k−1).

O ramo de adquirir outro sinal é g′+F_k(u−g′,h′,z′), após a geração que deixa
u de orçamento; o terminal é v+(1−S′)J_k(u−v,z′). Ambos exigem orçamento
admissível. Sem novas tentativas ainda pode ser útil continuar a atual, com
fallback V_0=λ. Com V=0, somente custo restante zero permite empate.

As identidades e o princípio de programação dinâmica têm antecedentes. Nenhum
resultado desta revisão é apresentado como teorema novo ou prova de otimalidade
dos gates empíricos.

## Fontes bibliográficas conferidas

1. **Russell e Wefald (1991), Principles of Metareasoning.** Autoria, título,
   periódico *Artificial Intelligence*, volume 49, números 1–3, páginas 361–395
   e DOI foram conferidos no registro depositado pelo editor. A referência foi
   acrescentada à bibliografia ativa e citada nos capítulos 2 e 3.
   [Registro Crossref](https://api.crossref.org/works/10.1016%2F0004-3702%2891%2990015-C),
   [fonte do autor](https://aima.cs.berkeley.edu/~russell/research-bo.html).
   A abertura direta da página integral do editor retornou HTTP 403; não se
   alega ter lido o artigo integral nesta rodada.
2. **Callaway et al. (2018), Learning to Select Computations.** Conferidos título,
   cinco autores e publicação no UAI 2018 no PDF dos anais. O manuscrito já
   continha uma entrada coerente, com endereço da versão arXiv especificada.
   [PDF oficial do UAI](https://auai.org/uai2018/proceedings/papers/269.pdf).
3. **Wan et al. (2025), BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling.**
   Conferidos título, sete autores, ano e versão v1. Mantido como preprint, sem
   atribuir periódico ou aceitação que não constam da versão consultada.
   [arXiv v1](https://arxiv.org/abs/2510.15945v1).
4. **Guo et al. (2024), DeepSeek-Coder: When the Large Language Model Meets Programming —
   The Rise of Code Intelligence.** Conferidos título, treze autores e versão v2.
   É o relatório primário da família original, não o de DeepSeek-Coder-V2.
   [arXiv v2](https://arxiv.org/abs/2401.14196v2).

## Correção da saída de anais

A inspeção do PDF encontrou prefixos “In:” vazios nas referências de anais.
O estilo ABNT usa `eventtitle` para habilitar a apresentação do evento e do
`booktitle`; acrescentar o campo resolveu a saída sem trocar o estilo ou
reclassificar os trabalhos. A condição foi reproduzida em um documento mínimo
e confirmada na [definição do estilo](https://github.com/citation-style-language/styles/blob/master/associacao-brasileira-de-normas-tecnicas.csl).

Além de Callaway, os eventos de Hewitt/Liang, Ovadia e Guo foram identificados
pelos anais originais: [ACL Anthology](https://aclanthology.org/D19-1275/),
[NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
e [PMLR](https://proceedings.mlr.press/v70/guo17a.html).
O parser bibliográfico não foi alterado. O PDF final não contém prefixos de
conferência vazios, e essa condição integra a auditoria automatizada.

Consulta: 07/11/2026. A compilação final verifica a resolução de todas as
citações e referências cruzadas. Esta rodada verificou a identificação das
quatro fontes prioritárias e os metadados de anais afetados; não representa uma nova revisão sistemática de
toda a literatura nem uma releitura integral de todos os artigos citados.
