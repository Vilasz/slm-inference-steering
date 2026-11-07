# Overview

> Documento exploratório anterior ao protocolo. O fluxo principal atual está em
> [research_workflow.md](research_workflow.md) e é regido por
> [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md).

Este repositorio investiga inference-time compute para pequenos modelos de
codigo. A hipotese principal e que ativacoes internas carregam sinal latente de
corretude/dificuldade, e que esse sinal pode deslocar a fronteira custo-acuracia
por duas vias:

- activation steering causal;
- Best-of-N adaptativo com politicas calibradas por custo.

O projeto e organizado como artefato experimental reproduzivel:

- `src/slm_steering`: biblioteca reutilizavel;
- `scripts`: entrypoints de experimento;
- `configs`: pipelines declarativas;
- `tests`: testes CPU/smoke;
- `runs`: saidas experimentais locais;
- `reports`: tabelas e figuras consolidadas.

Smoke tests nao baixam modelo grande. Runs completas normalmente exigem GPU,
datasets externos e tempo de execucao.
