# Benchmarks

Benchmarks implementados no registry:

- `humaneval`: HumanEval canonico via Hugging Face Datasets.
- `humaneval_plus`: adaptador HumanEval+/EvalPlus com fallbacks de schema.
- `mbpp`: adaptador MBPP/MBPP Sanitized que transforma asserts em
  `check(candidate)`.
- `humaneval_stress`: benchmark local pequeno para casos de borda.

Todos retornam `CodingProblem` com:

- `task_id`
- `prompt`
- `test`
- `entry_point`
- metadados opcionais (`benchmark`, `difficulty`, `source`, `tags`)

Smoke local:

```powershell
python scripts/run_baseline.py --benchmark humaneval_stress --limit 3 --n 2
```

MBPP e HumanEval+ podem exigir download de dataset. Testes unitarios usam mocks
e nao dependem da rede.
