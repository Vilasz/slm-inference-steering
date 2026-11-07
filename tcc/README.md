# Manuscrito integrado

O PDF atual é [build/manuscrito.pdf](build/manuscrito.pdf). Esta edição apresenta
as duas etapas como uma investigação única: Qwen e validações anteriores;
depois comparação prospectiva Qwen/DeepSeek, logística/MLP e consequências
operacionais. Resumo, introdução, fundamentação, método, resultados, discussão
e conclusão descrevem o mesmo escopo, preservando a cronologia científica.

```powershell
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
.venv/Scripts/python.exe scripts/audit_integrated_manuscript.py
```

Esse comando **não inicia coleta, avaliação funcional ou ajuste de modelos**.
Com a extensão final concluída, utiliza
[render_integrated_manuscript.py](../scripts/render_integrated_manuscript.py),
que lê os resultados das duas etapas e verifica os hashes científicos
congelados antes de renderizar tabelas e figuras.

Os capítulos autorais ativos são os incluídos em [main.typ](main.typ).
O texto das seções numéricas fica em `templates/*.typ.tpl`; as tabelas e os
valores são preenchidos em `generated/`. Para revisar a prosa dessas seções,
edite o template e execute o comando acima. Não edite apenas o arquivo gerado,
pois ele será reconstruído. A bibliografia reúne `referencias.bib` e
`referencias_extensao.bib` em uma única lista no PDF.

No workspace de pesquisa, a edição anterior de 75 páginas e os fontes foram preservados em
`reports/manuscript_revision_20261107/before/`. O manifesto editorial e as
figuras novas ficam no mesmo diretório, fora de `before/`. Os backups não fazem
parte da cópia compacta para o GitHub. No workspace, os arquivos antigos
`generated/resultados.typ`, `generated/excellence.typ`,
`chapters/06_discussao.typ` e `chapters/09_robustez_modelos.typ` são registros
da apresentação anterior e não são incluídos na edição atual.

O [relatório final](../reports/finalization/FINAL_REVIEW.md)
descreve a integração e as verificações. O registro de validação está em
`reports/manuscript_revision_20261107/validation.json`.

O publicador experimental congelado `run_final_extension.py publish` foi usado
para a edição anterior. Para esta redação integrada, use o comando de
manuscrito acima, não esse publicador histórico. Os scripts e resultados
científicos permanecem inalterados.

O PDF mantém A4, Times New Roman 12 pt, margens e bibliografia ABNT da edição
anterior. A revisão editorial não cria aprovação do orientador ou pré-registro
independente, nem transforma análises exploratórias ou secundárias em
confirmação da hipótese primária.
