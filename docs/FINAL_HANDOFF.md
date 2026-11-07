# Revisão pelo autor, orientação e entrega

O protocolo adicional já tem aprovação do estudante e revisão do orientador
informada por ele. Esta etapa trata da versão integrada do TCC, sem presumir
aprovação da banca ou assinatura do orientador.

## Ordem de leitura para a revisão final

1. `reports/excellence/final_quality_report.md`: verificações, pendências externas
   e afirmações que não podem ser certificadas.
2. `tcc/build/manuscrito.pdf`: resumo, introdução 1.4, resultados originais,
   diagnósticos, validação independente, discussão e conclusão.
3. `reports/excellence/final_scientific_audit.md` e
   `reports/excellence/independent_v1/README.md`: resultados reconstruídos,
   denominadores e alcance do novo endpoint.
4. `reports/excellence/operational_v1/README.md`: tempo completo, cobertura,
   repetições, componentes e limites da medição descritiva.
5. `reports/excellence/final_related_work_matrix.md` e fontes primárias; conferir
   pessoalmente as comparações e a redação de contribuição própria.
6. `docs/FINAL_REPRODUCIBILITY.md`: demonstrar uma reprodução estatística sem
   iniciar coleta na GPU.

## Defesa simulada de 18–20 minutos

Use os doze slides e o roteiro de `docs/excellence_defense.md`. Reserve perguntas
para depois dos vinte minutos. O estudante deve conseguir, sem ler:

- Definir WT-AUC, população ativa e quais tarefas são informativas para cada
  estimando; explicar por que um prompt determinístico empata suas tentativas.
- Desenhar os folds externos, internos e de calibração e mostrar onde o próprio
  rótulo é excluído, inclusive no primeiro estágio do residual.
- Explicar o resultado original negativo e por que não significa ausência
  universal de informação. Localizar no artefato o contraste e seus intervalos.
- Derivar custo finito por indicadores de início: a tentativa j só começa se
  as j−1 anteriores não tiveram sucesso mantido. A independência necessária é
  entre a tentativa atual e o passado, não entre custo e sucesso da mesma tentativa.
- Distinguir retenção ativa/global, candidatos/tarefas e ponto estimado/garantia.
- Percorrer manualmente uma sequência com sucesso na primeira tentativa, na
  última e sem sucesso, mostrando tokens e cobertura de verificador, gate e oracle.
- Explicar o único critério da reserva nova e por que ele não confirma PCA,
  residual, causa de domain shift nem benefício operacional conjunto.
- Ler um tempo de execução junto com cobertura e orçamento, sem extrapolar
  de tokens para segundos, energia ou outra GPU.

Registre no ensaio tempo por slide, respostas não sustentadas e referências que
precisam ser relidas. Uma dúvida não deve ser preenchida com uma explicação
causal sem identificação. Corrija o texto ou qualifique a conclusão, preservando
os dados e critérios. O roteiro é material de preparação, não um ensaio que o
agente possa declarar realizado em nome do autor.

## Calendário restante

O prazo informado é aproximadamente meados de dezembro de 2026; a data oficial
precisa ser confirmada com o curso.

| Período | Prioridade |
|---|---|
| Outubro, após integração | Leitura integral pelo autor e orientador; resolver questões científicas/textuais com os artefatos já existentes |
| Novembro | Ajustes de redação e formato, referências e repetições de defesa; confirmar regras administrativas e depósito |
| 1–14 de dezembro | Reservar prioritariamente para revisão, orientação e defesa; nenhuma coleta de alto risco |
| Antes do prazo oficial | Conferir PDF definitivo, documentos autênticos, autorizações e recibo de depósito |

Ficam com o curso/biblioteca a confirmação do modelo, a necessidade e emissão
de ficha catalográfica e a folha de aprovação com dados reais da defesa.
Não foram criados documentos, datas, assinaturas ou mensagens em nome de terceiros.
