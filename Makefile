.PHONY: venv install train test api drift

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

train:
	python -m churn.train --config configs/train.yaml

test:
	pytest -q

api:
	uvicorn api.main:app --reload

drift:
	python -m monitoring.drift_report --config configs/train.yaml
