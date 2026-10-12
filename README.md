# Simulator

Mini population simulator with a calibration harness: LLM personas predict GSS survey responses, scored against baselines. See [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md) for the build plan and [project_desc.md](project_desc.md) for the original brief.

## Setup
```
make setup   # creates .venv (Python 3.11) and installs deps
make test
```

## Conventions
- Each experiment is a YAML in `config/`; `simulator.config.config_hash` gives a stable 12-char hash.
- Outputs go to `runs/<name>-<hash>/` (gitignored).
- Raw data is never committed; `data/` holds download scripts only.
