# Technical Plan: Mini Population Simulator + Calibration Harness

## Context
`project_desc.md` asks for a small LLM-persona population simulator, scored honestly against real survey data (GSS) with baselines, calibration, and bootstrap CIs. The repo is empty (one initial commit), so this is a greenfield build. Goal of this plan: turn the description into concrete, ordered build parts with interfaces, so each part can be built and tested on its own. Harness and baselines come before any LLM call.

Defaults assumed (change if wrong): GSS (not ANES), Python 3.11, pandas/numpy/scikit-learn, YAML configs, SQLite/JSONL cache, providers behind one adapter interface. Hard deadline: midterm forecast tagged before **Nov 3, 2026**.

## Architecture (data flow)
```
data/ (download) -> splits (train/calib/test, fixed seed, frozen)
                 -> synth/ (raking, personas)  -> elicit/ (LLM -> prob vectors, cached)
                                                     |
eval/ baselines (uniform, logreg/GBM) --------------+--> metrics, calibration, bootstrap
                                                     --> reports/ (figures, writeup)
config/*.yaml -> run_id = hash(config) tags every output
```

## Repo layout
```
config/    experiment YAMLs (questions, models, persona level, N, temp, seed)
data/      download scripts + schema docs only (raw data gitignored)
synth/     raking.py, personas.py
elicit/    templates/, adapters/ (one per provider), cache.py, run.py
eval/      splits.py, baselines.py, metrics.py, calibration.py, bootstrap.py, failure.py
reports/   figures/, writeup.md
tests/     per-module unit tests
```

## Build parts

### Part 0 - Scaffold & conventions
- `pyproject.toml`, `.gitignore` (data/raw, cache), `README`, `Makefile` targets (`data`, `baselines`, `run`, `report`).
- `config/` loader + `config_hash()` util (stable hash of resolved YAML). Every output path includes the hash.
- Done when: `make test` runs and a dummy config produces a stable hash.

### Part 1 - Data loading & question selection (`data/`, `eval/splits.py`)
- Script to download GSS (cumulative file) and ACS PUMS; no raw data committed.
- Pick 15-25 questions across topic types (contested social, factual, behavioral). Record per-question: variable name, answer options, year range, **contamination note** (item release vs. model cutoff).
- Build one tidy respondent table: demographics (age, sex, race, educ, income, region, party, urban/rural), answer, survey weight.
- **Fixed split up front** (train/calib/test, seeded, saved to disk with checksum). Test set locked: only accessed by a final-run flag.
- Done when: split files + checksum exist, split tests assert no overlap and stable reproduction.

### Part 2 - Metrics & scoring harness (`eval/`)
Built and tested on synthetic predictions first.
- `metrics.py`: Brier, log loss (multiclass, with probability clipping), TVD, Wasserstein (ordinal), weighted variants.
- `calibration.py`: reliability bins, ECE, isotonic and temperature scaling fit on calib split only; report before/after.
- `bootstrap.py`: respondent-level bootstrap CIs for any metric; paired bootstrap for model-vs-model differences.
- Subgroup slicer: race, age band, education, region, party.
- Done when: unit tests with known-answer cases pass (perfect/uniform predictions, ECE of a calibrated sampler near 0).

### Part 3 - Baselines (`eval/baselines.py`)
- Uniform and train base-rate prior.
- Demographic-only logistic regression and gradient boosting, trained on train split, output probability vectors in the same schema as LLM output.
- Run end to end on **one question** through the full scoring harness. This is the Weekend 1 milestone and the "baselines before any LLM calls" checklist item.

### Part 4 - Population synthesis (`synth/raking.py`)
- IPF/raking of ACS PUMS to known marginals (age, sex, race, education, income, state), small scale.
- Output: synthetic population table with weights; validation report comparing achieved vs. target marginals.
- Path for GSS-respondent-direct runs (weight by survey weights) kept as the primary evaluation population; synthetic population used for the aggregate/poststratification demo.

### Part 5 - Persona rendering (`synth/personas.py`, `elicit/templates/`)
- Row -> prompt, with levels: L0 no persona, L1 demographics only, L2 demographics + attitudinal anchors (anchors must not include the target variable or near-duplicates).
- Multiple paraphrased question wordings per item for the prompt-sensitivity ablation.
- Done when: golden-file tests for rendered prompts at each level.

### Part 6 - Elicitation layer (`elicit/`)
- `adapters/`: common interface `get_distribution(prompt, options, n, temperature) -> probs, usage`. Implementations: logprob-based where supported, sampling-and-count otherwise. 3-4 models across providers.
- `cache.py`: key = hash(model, prompt, options, n, temperature, template version); stores raw outputs, probs, tokens, cost, latency.
- `run.py`: config-driven batch runner, resumable, rate-limited, writes per-run cost/latency summary.
- Done when: mock adapter passes tests; a real 20-persona smoke run caches and re-run costs $0.

### Part 7 - Aggregation (`eval/aggregate.py`)
- Poststratify persona-level probabilities with weights to predicted population distributions; compare to real marginals, overall and by subgroup.

### Part 8 - Experiments & ablations (`config/`)
- Core grid per question: {no-persona, L1, L2} x {models} x {temperature} x {N samples} x {prompt wording}.
- Staged: pilot on 3 questions -> scale to full set. Budget cap per run in config.
- Report which factor matters most (variance decomposition or simple effect sizes with CIs).

### Part 9 - Failure analysis (`eval/failure.py`)
- Homogenization: within-group variance, simulated vs. real.
- Stereotyping: error by minority subgroup / group-prototype comparison.
- Overconfidence: reliability diagrams, ECE before/after recalibration.
- Sensitivity: ranking flips across wording and models.
- Topic dependence: metrics by question category.
- Written **before** the polished results.

### Part 10 - Midterm pre-registration (deadline-driven, start early)
- Fix a small question set and method (config + hash) for a Nov 3, 2026 forecast; run it, write predictions to `reports/forecast_2026.md`, commit, and `git tag` (e.g. `preregistered-2026-midterms`) **before Nov 3**. Do not edit after tagging; score afterward.
- Note: needs its own population target (e.g. state/district electorate via PUMS) and clearly stated limitations.

### Part 11 - Report (`reports/writeup.md`)
Order: failures first -> question -> method -> baselines -> results with CIs -> where it was wrong and why -> next steps. Includes contamination section, cost/latency table, config hashes.

## Sequencing
| When | Parts |
|---|---|
| Weekend 1 | 0, 1, 2, 3 (one question end to end, no LLM) |
| Week 2 | 4, 5, 6, 7, 8 (scale to 15-25 questions, calibration) |
| Weekend 2 | 9, 11; plus 10 done before Nov 3 |
Suggest pulling Part 10 forward in parallel since it has the only hard date.

## Verification
- Unit tests per module (`pytest`); metric known-answer tests; split reproducibility and no-leakage tests; prompt golden files; cache hit tests with a mock adapter.
- Integration: `make baselines` runs one question end to end and emits metrics with CIs and a reliability plot.
- Checklist gate before final run: baselines exist, splits frozen, test set touched once, all outputs carry config hash, bootstrap CIs on every reported metric, cost/latency logged, contamination documented, midterm tag exists.
