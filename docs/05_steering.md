# Steering

A Fase 4 aplica direcoes latentes durante geracao e mede efeito causal.

Comando minimo:

```powershell
python scripts/run_steering_sweep.py `
  --directions-dir runs/phase3/qwen_phase3_probe `
  --layers 8,12,16 `
  --alphas 0,0.5,1 `
  --direction-types correctness_direction,negative_correctness_direction,random_direction,shuffled_label_direction `
  --limit 20 `
  --n 2 `
  --max-new-tokens 256 `
  --skip-existing `
  --require-cuda
```

Controles disponiveis incluem direcao negativa, aleatoria, aleatoria ortogonal,
labels embaralhados, direcao de comprimento, corretude ortogonalizada ao
comprimento e layer-shuffled direction.

O motor suporta single/multi-layer steering, schedules de alpha, selecao de
tokens, normalizacao L2 e clipping opcional.
