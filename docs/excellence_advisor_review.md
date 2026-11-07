# Solicitação de revisão do orientador — rascunho

Assunto sugerido: TCC — resultados auditados e proposta independente de transporte de gates

Professor Dário,

Preparei uma versão do TCC com reprodução dos resultados originais, auditoria
de partições e rótulos e aprofundamento exploratório. O resultado principal
continua negativo: H+B não supera os observáveis no critério definido. PCA e
uma correção residual com cross-fitting reduzem a desvantagem, sem estabelecer
superioridade sobre B. A transferência revela descartes concentrados e perda
de resolução; o oracle quantifica a oportunidade econômica idealizada.

Gostaria de sua revisão do manuscrito e da proposta `EXCELLENCE_EXTENSION_v1.md`
antes de qualquer congelamento científico novo. A proposta compara gates
congelados B/H+B no mesmo checkpoint, em 120 tarefas adicionais MBPP+, com
20 tentativas por tarefa e piloto técnico disjunto de 20 tarefas. O endpoint
é diferença de retenções sobre a mesma população de corretos ativos, com
bootstrap pareado por tarefa. Nenhum limiar é ajustado no alvo.

As estimativas na GPU conectada, RTX3050 Laptop 6GB, são 4,44 horas esperadas
ou 6,66 com margem. Uma RTX4060 exigiria medição e revisão do ambiente antes
de misturar trajetórias. A aprovação e os hashes da versão revista ficarão
registrados; a publicação atual não presume aprovação ou pré-registro independente.

Pontos para revisão:

1. Delimitação da contribuição e interpretação do resultado negativo.
2. Cross-fitting completo do offset, papel da calibração e limites da análise de capacidade.
3. População, endpoint único, amostra de 120 tarefas e precisão planejada.
4. Hipóteses do custo esperado, classe em que o oracle é um limite e separação de latência.
5. Normas de apresentação vigentes na EMAP e elementos formais necessários à submissão.

Materiais: `tcc/build/manuscrito.pdf`, `reports/excellence/scientific_audit.md`,
`docs/protocols/EXCELLENCE_EXTENSION_v1.md`, research summary de duas páginas,
defesa de doze slides e roteiro com33 perguntas.

Atenciosamente,
João Felipe

## Verificação formal antes da entrega

A página oficial de [Matemática Aplicada da EMAP](https://emap.fgv.br/curso/matematica-aplicada)
reúne documentos pedagógicos e aponta o repositório na seção TCC. Não foi
confirmado ali um modelo público específico de apresentação vigente para esta
submissão. Foi localizado um [manual geral do Sistema de Bibliotecas FGV](https://biblioteca.fgv.br/sites/default/files/2024-11/normalizacao-de-trabalhos-academicos-manual-fgv_1.pdf),
mas sua aplicabilidade e versão devem ser conferidas com a orientação do curso.
Não se adotaram regulamentos de outras escolas da FGV como regras da EMAP.

O manuscrito de pesquisa mantém A4, Times New Roman e referências APA da versão
recebida, com tabelas e fontes editáveis. Isso não é certificação de conformidade
formal para depósito. Folha de aprovação, composição da banca, assinaturas e
eventual ficha catalográfica dependem de dados institucionais reais e não foram
fabricadas. A etapa final de orientação deve definir esses elementos e o formato
exigido antes da entrega em dezembro.
