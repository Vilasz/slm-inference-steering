# Limitations

Limites atuais:

- Runs completas dependem de GPU, tempo e cache/download de modelos.
- HumanEval+ e MBPP dependem da disponibilidade e schema do dataset remoto.
- Steering causal so deve ser interpretado depois de extrair ativacoes reais e
  validar controles.
- Smoke tests validam infraestrutura, nao qualidade de modelo.
- Comparacoes pareadas so sao geradas quando ha JSONLs com tarefas sobrepostas.
- CI padrao nao instala CUDA nem roda modelo grande.

Roadmap tecnico:

- ampliar adapters para mais benchmarks;
- calibrar `latent_adaptive` com dados de validação;
- adicionar mais testes pareados por familia de metodo;
- publicar tabelas finais com seeds multiplas.
