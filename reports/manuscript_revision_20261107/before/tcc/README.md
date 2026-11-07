# Manuscrito

Abra [build/manuscrito.pdf](build/manuscrito.pdf) para leitura. O fonte principal
é [main.typ](main.typ), com capítulos editáveis em `chapters/` e bibliografia
em `referencias.bib`. O manuscrito recebido foi preservado em `archive/`.

As seções em `generated/` são produzidas por `scripts/build_manuscript.py` a
partir de `reports/final_review/evidence.json`. Elas contêm resultados observados,
com os mesmos valores do notebook. Atualize esse manifesto antes de gerar uma
nova versão e revise explicitamente a interpretação quando mudar de análise.

```powershell
.venv/Scripts/python.exe scripts/build_research_delivery.py
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
```

Para atualizar metadados bibliográficos pelas fontes primárias listadas:
`python scripts/fetch_tcc_references.py`. Revise as versões antes de substituir
uma bibliografia já utilizada no texto. As páginas primárias e os metadados
consultados estão em `reference_sources.json`.

A edição usa A4, Times New Roman 12 pt e bibliografia ABNT. Margens, hierarquia,
paginação e pré-textuais seguem o manual geral FGV consultado; o alcance e as
exigências institucionais externas estão em `reports/excellence/formatting_review.md`.
O desenho original foi
executado, incluindo HumanEval+ e transferência MBPP+. A extensão exploratória
está em `generated/excellence.typ`; sua síntese e fontes auditáveis estão em
`reports/excellence/`. A avaliação independente foi aprovada pelo estudante,
que informou revisão do orientador, e congelada antes das novas trajetórias.
O estado efetivo, os logs e os resultados completos ficam em
`reports/excellence/independent_v1/`; não se inferem resultados de bancos parciais.

Após alterações nas análises, execute `python scripts/run_excellence.py report`
antes de reconstruir a evidência. Os capítulos autorais estão em `chapters/`;
as seções geradas utilizam números rastreáveis aos artefatos. O novo coletor
exige aprovação vinculada aos hashes do protocolo e dos fontes.
