# O que publicar no GitHub

Publique **o conteúdo de `dist/github_final/`**. Essa pasta é uma cópia curada
da edição final; não é necessário copiar a pasta de trabalho inteira.

Para prepará-la, depois da auditoria e da construção final:

```powershell
.venv/Scripts/python.exe scripts/export_final_repository.py
```

O script não altera o índice, os commits ou a branch do repositório de pesquisa.
Não faz commit, push ou publicação remota. A lista exata de arquivos e seus
hashes fica em `dist/github_final/PUBLICATION_MANIFEST.json`.

| Publicar | Motivo |
|---|---|
| `README.md`, `LICENSE`, `requirements-final.txt` | Apresentação, licença existente e ambiente mínimo |
| `tcc/` selecionado | Um PDF final, capítulos ativos, templates, arquivos gerados, biblioteca e bibliografia |
| `scripts/` selecionado | Construção, auditoria, exportação e fontes científicos exigidos pelo congelamento |
| `src/` selecionado | Módulos usados e dependências locais dos fontes científicos preservados |
| `docs/` selecionado e `RESEARCH_PROTOCOL.md` | Reprodução, publicação, protocolos e proveniência |
| `reports/` selecionado | Resultados usados, figuras ativas e auditorias finais |
| `runs/` mínimo | Arquivos históricos e bundles leves de probes que os hashes congelados exigem |
| `tests/` selecionado | Testes preservados no congelamento e verificação leve de recursos |
| `.gitattributes`, `.gitignore`, `pytest.ini` | Preservação dos bytes, exclusões e perfil leve de testes |

O histórico mínimo é mantido porque os manifests o exigem para verificar a
integridade das evidências. Não são incluídas versões antigas do PDF, previews,
backups editoriais, notebooks parciais, logs completos, caches, ambiente virtual,
ativações NPZ ou pesos dos geradores. A publicação compacta usa o CSV de
metadados externos em `reports/finalization/reproduction/`.

## Conferir a cópia

```powershell
.venv/Scripts/python.exe scripts/export_final_repository.py --verify
```

Para repetir também a passagem por um Git isolado e os comandos em uma cópia
extraída, use `python scripts/verify_publication_roundtrip.py --smoke`. A opção
`--python CAMINHO` permite selecionar outro ambiente mínimo. O teste não cria
commit nem altera o índice do repositório de pesquisa.

O manifesto confere os arquivos selecionados e o limite de tamanho por arquivo.
O `.gitattributes` da cópia desativa conversões automáticas de fim de linha,
preservando os SHA256 históricos ao passar pelo Git em sistemas diferentes.
As exclusões da cópia são próprias: os poucos arquivos `runs/` necessários
precisam ser publicados, embora o workspace completo ignore sua pasta `runs/`.

Se a cópia já existir, uma nova exportação exige `--refresh`. Esse modo atualiza
somente arquivos gerenciados, após conferir os caminhos. Não use esse comando
em uma pasta na qual você já iniciou outro trabalho; uma cópia com `.git/`
é recusada. Gere a entrega a partir do workspace de pesquisa.

## Enviar

Copie o conteúdo de `dist/github_final/` para uma pasta separada destinada ao
repositório público e inicialize ou use o Git nessa pasta. Inclua todos os
arquivos da cópia, inclusive os ocultos. Confira a lista antes de fazer commit
e push para o repositório que você escolher. O `PUBLICATION_MANIFEST.json`
permite comparar os bytes publicados com esta entrega.

O README e o PDF dessa cópia são a apresentação principal. Os protocolos
versionados permanecem como registros históricos: seus estados de proposta
não devem ser reescritos para aparentar uma aprovação retrospectiva.
