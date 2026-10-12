# Mini Population Simulator with a Calibration Harness

A project plan: build a small version of what Aaru sells, and score it honestly against ground truth.

## 1. Pick the ground truth carefully

**Primary: GSS (General Social Survey) or ANES, predicting survey responses.** These have respondent-level data with demographics and answers, so you can score at both the individual and aggregate level. That gives you many more data points than a single election result, which is one number per state and nearly useless for calibration.

**Bonus: a pre-registered forecast of the Nov 3, 2026 midterms.** Commit your predictions with a timestamp (a git tag works) before the election, then score them afterward. A forecast made before the outcome is known is more credible than any backtest.

**Watch for contamination.** LLMs have likely seen the 2020 and 2024 election results and well-known Pew findings. For backtests, use GSS items released after the model's cutoff, or obscure items where the model can't just recall the answer. State the contamination risk in the writeup.

## 2. Build the pipeline in four stages

1. **Population synthesis.** Use ACS PUMS (Census microdata) and iterative proportional fitting (raking) to build a synthetic population that matches known marginals by age, sex, race, education, income, state, and so on. If you use GSS respondents directly, you can skip this and weight by the survey weights. Do the synthesis anyway on a small scale, since it's the "populations represent real people" part of the job.
2. **Persona rendering.** Turn each row into a prompt. Keep versions of varying richness (demographics only versus demographics plus a few attitudinal anchors) so you can ablate them later.
3. **Elicitation.** For each persona and question, get a probability distribution over answer options, not just one answer. Either read token logprobs or sample N=10-20 times and count. Cache everything and log cost and latency per call.
4. **Aggregation.** Poststratify the persona-level predictions to the target population and compare against the real marginals and subgroup breakdowns.

## 3. Baselines

This is where credibility comes from. The LLM must beat:

- **Uniform / base-rate prior** (the floor)
- **No-persona LLM** (same question, no demographics) to show whether personas add anything
- **A demographic-only model**, such as logistic regression or gradient boosting trained on a GSS training split. This is the real bar. LLM persona simulation often fails to beat a simple model trained on the same demographics, and finding that out is a legitimate result.

## 4. Scoring

- **Individual level:** Brier score and log loss on held-out respondents
- **Calibration:** reliability diagrams and expected calibration error (ECE), plus recalibration with isotonic regression or temperature scaling fit on a separate calibration split. Report before and after.
- **Aggregate level:** total variation distance or Wasserstein distance between predicted and actual answer distributions, overall and by subgroup (race, age, education, region, party)
- **Uncertainty:** bootstrap confidence intervals on everything, so you can say which differences are real
- **Discipline:** split train, calibration, and test sets up front, and touch the test set once

## 5. Where it will likely fail (and test for it)

These are known failure modes of LLM-simulated populations, and each can be tested:

- **Homogenization.** Simulated variance is too low, so the spread of opinions within a group is too narrow. Compare within-group variance to the real data.
- **Stereotyping.** The model predicts the group stereotype instead of the actual distribution. Look at minority subgroups in particular.
- **Overconfidence.** Predicted probabilities are too extreme, which shows up directly in the reliability diagram.
- **Prompt and model sensitivity.** Rerun with reworded prompts and different models (3-4 across providers, as in injection-pareto). If rankings flip, say so.
- **Topic dependence.** Contested social topics may do worse than factual or behavioral ones.

Ablate persona richness, model, temperature, and N samples, and report which choices matter most.

## 6. Repo structure

```
config/          # YAML per experiment, so every run is reproducible
data/            # download scripts, not raw data
synth/           # raking + persona construction
elicit/          # prompt templates, model adapters, caching
eval/            # metrics, calibration, baselines, bootstrap
reports/         # figures + the writeup
```

Log a config hash with each run. Track cost and latency per simulation, which is the "instrumentation" the posting cares about.

## 7. The writeup

Structure: question -> method -> baselines -> results with CIs -> "where it was wrong and why" -> what I'd do next.

Lead with the failure section. A sentence like "the simulation beat uniform but lost to logistic regression on 6 of 9 questions, and was most overconfident for older rural respondents" is the kind of intellectual honesty the posting asks for.

## 8. Timeline

- **Weekend 1:** data loading, baselines, scoring harness, and one question end to end. Build the harness before the LLM part, so the whole thing is testable.
- **Week 2:** scale to ~15-25 questions, run ablations, and calibration.
- **Weekend 2:** failure analysis, figures, writeup, and the midterm pre-registration (do this before Nov 3).

## 9. Checklist

- [ ] Train / calibration / test split fixed up front
- [ ] Baselines implemented before any LLM calls
- [ ] Elicitation outputs probabilities, not single answers
- [ ] All runs cached and config-hashed
- [ ] Bootstrap CIs on every reported metric
- [ ] Contamination risk documented
- [ ] Ablations: persona richness, model, prompt wording, N samples
- [ ] Failure analysis written before the polished results section
- [ ] Midterm forecast committed and tagged before Nov 3
- [ ] Cost and latency per simulation reported