# Results

Gere artefatos finais com:

```powershell
python scripts/generate_final_report_artifacts.py
```

Saidas principais:

- `reports/tables/main_results.csv`
- `reports/tables/paired_comparisons.csv`
- `reports/tables/steering_controls.csv`
- `reports/tables/adaptive_policy_results.csv`
- `reports/tables/cross_model_results.csv`
- `reports/tables/robustness_results.csv`
- `reports/figures/accuracy_cost_scatter.png`
- `reports/figures/probe_auc_by_layer.png`
- `reports/figures/steering_alpha_sweep.png`
- `reports/figures/adaptive_budget_accuracy.png`

Figuras sem dados reais exibem mensagem de insuficiencia, em vez de inventar
resultados.
