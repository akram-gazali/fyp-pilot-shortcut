# SIFT-VQA protocol (pilot-stage decisions)

Numbers below are pilot numbers from `reports/stage5/`; none is evidence for the hypothesis.

## Metrics (decided 2026-11-20 by the project owner)
- **Primary:** joint score on the paired held-out subset (original and edited question both right), held-out combination compare_cs.
- **Secondary (declared):** plain accuracy on all held-out compare_cs questions.
- **Guard:** seen-template accuracy must not drop by more than 2 points against the matched-random arm.
- **Margin (unchanged):** a gain of 3 points counts as an effect.

## How the primary metric was chosen
It was chosen after a frozen-model headroom check (`reports/stage5/zero_shot.json`: plain held-out 95.0%, joint 60.0%, relevant-edit 67.32%) and before any trained arm was evaluated. No fine-tuned arm has been run or scored on a GPU at the time of writing.

## Known issue, flagged and not resolved
The 3-point margin was not changed. With the current test set the joint score detects about 8 points (80% power, ASSUMED seed sd and paired-question counts; `reports/pilot_report.md` section 8), which is above 3 points. A 3-point joint effect is therefore not detectable at the current test size. Larger test sets are costed in the report (3x and 5x).

## Platform
Pilot runs: Colab T4 (Stage 4, torch 2.14.1+cu130) and Kaggle T4 (Stage 5, torch 2.11.0+cu130). The full study runs on Kaggle only. Format `choice` and 224 px were fixed on the dev set and are frozen.

## Disclosure: the pipeline check used the test split
The tiny fine-tune (`reports/stage5/tiny_finetune.json`, PIPELINE CHECK ONLY) evaluated trained arms on 200 questions sampled from the test split (100 held-out compare_cs, 100 seen), after the primary metric was chosen. Its results are not used to tune anything and are not evidence for the hypothesis. The first run recorded plain accuracy only; a Kaggle rerun (commit 7e1383b) also recorded the paired-edit joint score per arm (63 paired held-out items).
