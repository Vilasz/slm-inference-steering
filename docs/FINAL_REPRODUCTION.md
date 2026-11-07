# Reprodução da versão final

## Escopo reproduzível sem modelos

A entrega pública permite reconstruir os contrastes, intervalos, interação,
calibração, poder e replay da extensão final e recompilar o manuscrito. Usa
scores já salvos e um CSV com os metadados das quatro mil tentativas externas.
Não carrega pesos Qwen/DeepSeek, não executa código gerado e não treina probes.

Os protocolos históricos, resultados originais e bundles leves dos probes são
preservados para proveniência. Os manifests exigem alguns bancos históricos
JSONL, que estão incluídos na cópia curada. Os registros completos e ativações
da extensão nova continuam no workspace de pesquisa; sua ausência da entrega
compacta é explícita. O catálogo de hashes dos registros originais acompanha
o CSV derivado.

## Ambiente

Foi utilizado Python 3.12. As dependências da reconstrução e publicação estão
em `requirements-final.txt`; esse arquivo não instala PyTorch nem geradores.
A compilação com a identidade visual original requer Times New Roman instalada.
A fonte não é redistribuída no repositório. Uma substituição tipográfica pode
mudar as quebras de página e emitir avisos; o PDF final fornecido usa a fonte
verificada no Windows.

Um ambiente separado com as 27 dependências fixadas foi instalado e verificado
durante a entrega, sem pacotes de modelos. A execução a partir de uma cópia
extraída do Git também foi conferida. Os registros datados ficam em
`reports/finalization/dependency_verification.json` e
`reports/finalization/publication_verification.json`.

Na raiz do projeto, em PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-final.txt
.venv/Scripts/python.exe scripts/finalization_audit.py
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
.venv/Scripts/python.exe scripts/audit_integrated_manuscript.py
.venv/Scripts/python.exe -m pytest -q tests/test_final_extension_resources.py
```

Em Linux/macOS, crie o ambiente com `python3.12 -m venv .venv` e substitua
`.venv/Scripts/python.exe` por `.venv/bin/python` nos comandos restantes.

## O que cada comando verifica

| Comando | Insumo e resultado |
|---|---|
| `finalization_audit.py` | Confere hashes científicos, identidades e labels salvos; reconstrói AUC por pares, perdas, bootstrap pareado, Γ, calibração, 540 cenários de poder, 5.600 linhas de replay, 56 sínteses e casos sintéticos de decisão. Escreve a auditoria e CSVs derivados em `reports/finalization/`. |
| `build_manuscript.py --compile` | Preenche os cinco templates numéricos com os resultados concluídos, gera figuras e recompila `tcc/build/manuscrito.pdf`. Registra os hashes em `reports/manuscript_revision_20261107/manifest.json`. |
| `audit_integrated_manuscript.py` | Confere fontes e histórico congelados, proveniência editorial, citações/referências pelo compilador, valores centrais no PDF e largura do conteúdo. |
| Testes de recursos | Verificam contabilização de intervalos de suspensão sem sobreposição e preservação dos tempos brutos. |

Execute a auditoria estatística antes da construção quando atualizar seus
relatórios: a renderização lê a síntese auditada para documentar abortos reais.
Os timestamps dos relatórios e manifests são metadados de execução e mudam
em uma reprodução. Os resultados numéricos devem permanecer dentro da
tolerância absoluta de 1e−11 usada pela auditoria.

As 231 verificações específicas executadas na máquina de pesquisa estão em
`reports/finalization/tests.xml`. Parte desses testes históricos também usa o
ambiente completo de geração e um tokenizer em cache. Eles não fazem parte da
instalação mínima; o comando estatístico executa os controles matemáticos
necessários sem esses pacotes.

## Edição do texto

`tcc/main.typ` define os capítulos ativos. A prosa autoral está em
`tcc/chapters/`; a prosa das seções que contêm tabelas numéricas fica nos cinco
arquivos `tcc/templates/*.typ.tpl`. O builder preenche `tcc/generated/`.
Edite o template e reconstrua, em vez de alterar somente um arquivo gerado.

A bibliografia combina `tcc/referencias.bib` e `tcc/referencias_extensao.bib`.
As equações de Bellman estão no capítulo de fundamentação. O documento
conceitual histórico mantém sua notação original; o manuscrito final usa
consistentemente k como número de novas tentativas além da atual.

## Limites da reprodução

Essa sequência não reproduz a coleta, a execução funcional dos programas ou o
treinamento completo dos probes. Reexecutá-los exige dados extensos, pesos,
ambiente histórico, isolamento de execução e os controles dos protocolos.
Não são necessários para conferir os cálculos desta entrega e não são
acionados pelos comandos acima.

Os IC continuam condicionais aos probes treinados. A reconstrução não mede
latência online, não elimina contaminação de pré-treinamento e não converte
intervalos que cruzam zero em equivalência.
