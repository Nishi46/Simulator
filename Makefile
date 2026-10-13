PY ?= .venv/bin/python

.PHONY: setup test data baselines run report

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest

data:
	$(PY) -m data.pipeline config/data.yaml

# Wired up in later parts:
baselines run report:
	@echo "'$@' not implemented yet (see TECHNICAL_PLAN.md)"
