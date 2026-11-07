# Recuperação de gravação no Windows — 15/10/2026

A coleta principal parou em 15/10/2026 com `PermissionError: [WinError 5]`
ao substituir `manifest.json.tmp` por `manifest.json`. O manifesto registrava
2.432 tentativas, mas o JSONL já continha 2.433 registros completos, com hashes
e identidades válidos. O último registro foi persistido antes da falha de
atualização do manifesto; ele é preservado na retomada.

A auditoria está em
`reports/research/definitive_v0.2_io_recovery_audit.json`. O manifesto anterior
está preservado em
`reports/research/full_execution_v0.2/manifest_before_io_recovery.json`.

## Compatibilidade de filesystem

`scripts/run_research_safely.py` inicia o runner congelado com
`scripts/research_runtime/` no `PYTHONPATH` dos processos Windows. O módulo
`sitecustomize.py` instala uma camada de compatibilidade em `os.replace`:

- Apenas substituições de arquivos `.json.tmp` por `.json` dentro do repositório.
- Apenas erros Windows 5, 32 e 33; demais erros continuam sendo propagados.
- Esperas de 10 ms até 250 ms entre tentativas, limitadas a dez segundos.
- A substituição continua atômica e preserva os bytes JSON produzidos pelo código original.

Os 29 fontes e arquivos do snapshot científico permanecem intactos. A camada
não modifica geração, precisão, amostragem, sementes, rótulos, probes, hipóteses
ou critérios de decisão. Cada lançamento preserva o estado anterior e os hashes
do launcher e dos módulos de compatibilidade em
`reports/research/full_execution_v0.2/io_recovery_*/runtime.json`.
Essa alteração administrativa deve acompanhar a reprodução do ambiente Windows.

## Verificação

Quatro testes cobrem bloqueio transitório, bloqueio permanente, arquivos fora
do escopo e um leitor Windows real que não permite exclusão/substituição.
Um teste de integração adicional executa a função `write_json` congelada sob
esse bloqueio real, ativando a compatibilidade via `PYTHONPATH`; seu resultado
está em `reports/research/runtime_io_compatibility_test.json`.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_research_runtime.py -q
.venv/Scripts/python.exe scripts/run_research_safely.py --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python
```

A amostra planejada permanece em 960 tentativas no piloto, 2.640 no banco
principal e 640 na transferência. Erros de execução não recebem rótulo científico.
