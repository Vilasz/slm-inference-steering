# Latent Geometry

A Fase 3.5 testa se o sinal latente e robusto:

```powershell
python scripts/analyze_latent_geometry.py `
  --activations-dir runs/phase3/qwen_phase3_probe `
  --phase2-runs-dir runs/phase2 `
  --output-dir runs/phase35/latent_geometry `
  --n-bootstrap 1000 `
  --n-permutations 500
```

Artefatos:

- `probe_results.csv`: AUC, accuracy, F1 e calibracao por camada.
- `latent_score_results.csv`: scores por amostra e controle.
- `permutation_results.csv`: significancia empirica.
- `regression_results.csv`: ganho incremental do score latente.
- `survival_results.csv`: tempo/tentativas ate sucesso.
- `bootstrap_results.csv`: intervalos de confianca.

Se `summary.json` indicar `missing_activations`, rode a Fase 3 antes de
interpretar a Fase 3.5.
