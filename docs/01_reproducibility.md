# Reproducibility

Ambiente recomendado:

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Validacao leve sem GPU:

```powershell
python -m pytest -m "not slow and not gpu"
python scripts/run_experiment.py --config configs/smoke.yaml
```

Execucao pequena com GPU:

```powershell
python scripts/run_experiment.py --config configs/reproduce_small.yaml --continue-on-error
```

Geracao de relatorio:

```powershell
python scripts/generate_final_report_artifacts.py
```

O smoke oficial carrega apenas o benchmark local `humaneval_stress` e regenera
artefatos a partir de dados ja presentes. Ele nao deve ser citado como resultado
final de benchmark.
