PYTHON ?= python

.PHONY: test lint typecheck smoke reproduce-small report protocol-plan protocol-pilot

test:
	$(PYTHON) -m pytest -m "not slow and not gpu"

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src scripts

smoke:
	$(PYTHON) scripts/run_experiment.py --config configs/smoke.yaml

reproduce-small:
	$(PYTHON) scripts/run_experiment.py --config configs/reproduce_small.yaml --continue-on-error

report:
	$(PYTHON) scripts/generate_final_report_artifacts.py

protocol-plan:
	$(PYTHON) scripts/build_candidate_bank.py --dry-run

protocol-pilot:
	$(PYTHON) scripts/build_candidate_bank.py --phase pilot --time-budget-seconds 600
