# Guia científico atual do TCC

Atualização: 17/10/2026. A avaliação v0.2 está concluída: 960 trajetórias de piloto,
2.640 principais em 132 tarefas e 640 de transferência MBPP+ em 32 tarefas.
O endpoint de 32 tokens tem 2.158 candidatos ativos e 92 tarefas mistas.

O contraste H+B menos B em WT-AUC é −0,0470, IC95% [−0,0994; 0,0056];
em log loss, +0,0379, IC95% [0,0067; 0,0706]. Os critérios de ganho incremental
e de benefício operacional conjunto não foram satisfeitos no desenho original.
O gate H+B reteve 276/366 corretos ativos em MBPP+ (75,41%; IC95% [61,87%; 87,74%]).

## Escopo e história

O [protocolo v0.2](../RESEARCH_PROTOCOL.md) e os bancos originais permanecem
preservados. Trata-se de execução de desenho fixo autorizada pelo estudante,
com exposição histórica divulgada. Aprovação do orientador e pré-registro
independente não são presumidos a partir da execução. A versão recebida deste
próprio guia está arquivada em `reports/excellence/archive_before_extension/docs/`.

## Aprimoramento científico

A [auditoria científica](../reports/excellence/scientific_audit.md) reconstrói os
resultados e classifica as limitações. O [registro de progresso](../reports/excellence/PROGRESS.md)
acompanha ablações de observáveis, PCA, correção residual, transporte do gate,
oracle econômico, medição técnica, revisão do manuscrito e materiais de defesa.

Novas análises sobre os dados examinados são exploratórias. A proposta de
avaliação independente em `protocols/EXCELLENCE_EXTENSION_v1.md` será submetida
à aprovação do estudante e revisão do orientador antes do congelamento e coleta.

## Escrita e defesa

Separe associação agregada, ordenação intratarefa, calibração, retenção de
candidatos e resolução das tarefas. Q2 exige IC positivo de ΔWT-AUC e redução
pontual de log loss; o IC da perda qualifica sua incerteza. Permutação de
associação não testa incremento condicional. Não rejeição não demonstra equivalência.
Intervalos exploratórios pontuais não substituem os critérios originais.

Não atribua à dimensionalidade ou ao domínio uma causa identificada sem desenho
que a isole. Não converta tokens em latência, energia ou FLOPs. O oracle é
retrospectivo e conhece rótulos finais. Equações econômicas condicionam na tarefa
e distinguem retenção global de retenção ativa. Heterogeneidade não desaparece
por substituir parâmetros de tarefas por médias agregadas.

O [notebook](../notebooks/final_research_summary.ipynb), [HTML](../reports/final_review/notebook.html)
e [PDF](../tcc/build/manuscrito.pdf) devem ser regenerados e auditados após alterações
relevantes. Cada número deve ser rastreável a uma tabela, código e insumo verificáveis.
