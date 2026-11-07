# Activation Extraction

A Fase 3 extrai ativacoes de tentativas ja geradas. O fluxo recomendado e:

```powershell
python scripts/extract_activations.py `
  --input-jsonl runs/phase2/qwen15b_temp08_n5.jsonl `
  --summary-json runs/phase2/qwen15b_temp08_n5_summary.json `
  --output-dir runs/phase3/qwen_phase3_probe `
  --layers 0:28:4 `
  --token-position completion_last `
  --difficulty-filter all `
  --require-cuda

python scripts/analyze_activations.py runs/phase3/qwen_phase3_probe
```

Principais artefatos:

- `activations.npz`
- `metadata.jsonl`
- `manifest.json`
- `separability.csv`
- `linear_probes.csv`
- `latent_directions.npz`

Sem `latent_directions.npz`, fases latentes e steering ainda podem rodar com
controles aleatorios, mas resultados de steering de corretude nao devem ser
interpretados como causais.
