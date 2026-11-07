# Adaptive Best-of-N

A Fase 5 avalia politicas que decidem continuar ou parar a amostragem.

Politicas implementadas:

- `fixed_n_1`
- `fixed_n_5`
- `fixed_n_10`
- `verifier_early_stop`
- `difficulty_adaptive`
- `latent_adaptive`
- `steering_fixed_n`
- `steering_latent_adaptive`

Comando exemplo:

```powershell
python scripts/run_adaptive_bon.py `
  --policy latent_adaptive `
  --directions-dir runs/phase3/qwen_phase3_probe `
  --max-n 5 `
  --limit 30 `
  --output-dir runs/phase5/adaptive_bon `
  --require-cuda
```

Metricas salvas incluem taxa de sucesso, tokens medios/medianos, tokens por
sucesso, tentativas medias, economia de budget, utilidade esperada e deltas
quando um baseline fixo comparavel e fornecido.
