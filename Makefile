PY ?= .venv/bin/python

.PHONY: setup data train evaluate test lint app check

setup:            ## create venv and install pinned dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

data:             ## regenerate all synthetic data (deterministic)
	$(PY) -m reteach.cli generate-data

train:            ## fit the classifier on the dev split and save it to models/
	$(PY) -m reteach.cli train

evaluate:         ## train + evaluate on the held-out test split; writes artifacts/
	$(PY) -m reteach.cli evaluate

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

app:
	$(PY) -m streamlit run app.py

check: lint evaluate test
