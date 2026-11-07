# Autorização de execução integral — 14/10/2026

O estudante solicitou explicitamente:

> Pelo relatório (manuscrito) temos apenas uma parte excploratório, poderia por favor executar o código para termos o experimento completo e podermos tirar conclusões deles? Traga mais visualizações desses resultados para o manuscrito

Essa instrução autoriza a execução integral local. O registro administrativo
alternativo do Gate 0 preserva o protocolo, a configuração e os fontes em
`reports/research/execution_freeze_v0.2/snapshot/`, com hashes verificáveis.
Não concede aprovação ao orientador, não cria um pré-registro independente
e não altera temperatura por preferência, amostra, checkpoints, sementes,
regras de seleção, hipóteses, comparadores ou critérios de decisão.

O `RESEARCH_PROTOCOL.md` permanece preservado: seu hash identifica o piloto
já iniciado. A aprovação do orientador continua pendente. O caminho anterior
de aprovação científica e tag Git permanece disponível; esta execução usa a
autorização explícita do estudante e a cópia verificável anterior aos resultados
principais. O manuscrito deve divulgar essa distinção e a exposição histórica.

O Gate 2 continua obrigatório: somente um piloto completo e um regime elegível
podem liberar o banco principal. Invariância de prefixos é conferida em todos
os checkpoints disponíveis de cada banco completo, sem seleção por corretude.
A validação online usa as primeiras 12 tarefas por ID lexicográfico, definidas
antes dos resultados, e rotaciona a ordem das políticas. Tempos são descritivos.

O runner executa piloto, auditorias, congelamento de amostragem, banco principal,
validação causal, probes aninhados, validação online, inferência agrupada,
congelamento do sistema fonte, MBPP+, notebook e PDF. Qualquer falha interrompe
o fluxo e fica registrada; não é substituída por um rótulo de falha funcional.

Para acompanhar sem carregar o modelo:

```powershell
.venv/Scripts/python.exe scripts/research_status.py
Get-Content reports/research/full_execution_v0.2/pilot.log -Tail 20
```

Para retomar após uma parada normal, sem abrir outro escritor concorrente:

```powershell
.venv/Scripts/python.exe scripts/run_complete_research.py --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python
```

Se existir uma trava de processo encerrado, inspecione o PID, o último registro
JSONL completo e a auditoria antes de recuperá-la. Não apague travas de processos
ativos. Os bancos ficam em `runs/`; resultados pequenos e gráficos de publicação
ficam em `reports/`; o documento fica em `tcc/`.
