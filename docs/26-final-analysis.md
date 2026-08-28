# 26 — Final analysis and research outcome

## Outcome

The selected result is the original three-seed 1.5B SFT baseline. It improves
internal validation accuracy from 62.50% to a three-seed mean of 94.72%, while
BFCL decision accuracy averages 92.08%. The central hypothesis is only partially
supported: supervised fine-tuning teaches abstention, but every tested preference
optimization variant regresses tool calling enough to fail its predeclared gate.

This is a useful negative result, not evidence that DPO never works. It shows that
preference reward accuracy can be badly disconnected from free-generation agent
behavior in this small-model, low-data setting.

## Statistical analysis

The committed `make analysis` pipeline reads metrics and per-example evaluations
only. It never opens predictions, external raw data, training records, adapters, or
the sealed internal test split.

Across the three independent 1.5B SFT seeds, internal accuracy is 94.72% (sample
SD 0.96%; t-based 95% CI 92.33–97.11%). Act accuracy is 95.00% and abstention
accuracy is 94.44%, though their three-seed intervals are wide and extend beyond
the natural probability range. Intervals are deliberately not clipped; n=3 makes
them exploratory rather than definitive.

Deterministic paired bootstrap comparisons use 10,000 resamples:

- SFT seed 0 improves internal accuracy over base by 33.33 points (95% CI
  25.00–41.67) on the same 120 validation tasks.
- SFT seed 0 improves BFCL decision accuracy over base by 5.16 points (95% CI
  2.97–7.34) on the same 640 records.
- Protocol repair is tied with SFT seed 0 internally (difference 0; 95% CI
  -5.00–5.00), but loses 9.22 BFCL points (95% CI -11.72–-6.88).

These intervals describe these fixed evaluation sets. They do not establish broad
population-level significance, and hyperparameter decisions were not based on
BFCL.

## Model-selection record

The 1.5B SFT seed-0/1/2 runs all remain selected evidence. Seed 0 is not presented
alone as the conclusion because it was unusually favorable on external
abstention. The protocol-repair adapter is rejected because its BFCL abstention
accuracy fell to 58.33%. Standard 1.5B DPO is rejected because act accuracy fell
to 0% despite 100% preference reward accuracy.

The 0.5B experiments are a separate screening family. The competent SFT screen
reached 80.83% internal accuracy and 63.33% act accuracy. Mean DPO, corrected
stratification, and supervised anchors all failed frozen behavior gates. They are
retained as failure evidence and are never compared as replacements for 1.5B.

## Compute accounting

All training and inference ran locally on Apple Metal. Recorded peak memory was
4.20 GB for the primary 1.5B SFT configuration, 12.327 GB for 1.5B DPO, 7.13 GB
for the competent 0.5B SFT screen, and at most 8.991 GB for the terminal 0.5B DPO
screens. The 1.5B DPO run took 2,455 seconds; the recorded 0.5B preference screens
took 8.7, 63.6, 55.5, 27.1, and 28.9 seconds. Earlier SFT/inference wall times were
not consistently captured, so no invented total compute figure is reported.

## Reproduction and provenance

Run:

```bash
make analysis
```

This CPU-only, network-free command emits canonical JSON, CSV, Markdown, and SVG
files under `reports/final/`. Every input file and output artifact is SHA-256
pinned in the generated summary and manifest. Repeated builds are byte-identical.
The comparison config rejects any path containing a `test` component. Public BFCL
records remain evaluation-only, and AgentAbstain remains catalog-only.

## Limitations and next research step

The internal corpus is synthetic, templated, and modest; BFCL tests only the
CALL-versus-ABSTAIN decision here, not exact function arguments. Three seeds are
enough to expose instability but not enough for tight uncertainty estimates. The
project has one model family and no human evaluation of final model generations
beyond evaluator calibration.

The right next research experiment is not another local DPO sweep. It is a larger,
more diverse, licensed internal-training corpus plus a conservative objective with
generation-based checkpoint gates. Until that data exists, the honest release is
the SFT result and the characterized preference-optimization failure.
