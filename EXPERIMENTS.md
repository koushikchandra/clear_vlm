# Experiment Log & Results Index

Every evaluation run, its Slurm job, config, log files, and result artifacts. Auto-generated from `logs/` and `results/`.

**Total runs:** 20  ·  Models: Qwen2.5-VL-7B/3B, LLaVA-1.6-7B, MedGemma-4B  ·  Datasets: VQA-RAD, PathVQA

Reproduce any row with:

```bash
sbatch code/eval/run.sbatch <MODEL_ID> <DATASET> <N> <TAG> [CONDITIONS] [VERBALIZED]
```

## Main (intact/blank/mismatch)

| Job | Model | Dataset | N | Tag | Status | Logs | Results |
|-----|-------|---------|---|-----|--------|------|---------|
| `12189910` | Qwen2.5-VL-7B | pathvqa | all | `qwen7b_pathvqa` | ✅ done | [out](logs/medvlm_eval_12189910.out) · [err](logs/medvlm_eval_12189910.err) | [preds](results/qwen7b_pathvqa.jsonl) · [metrics](results/qwen7b_pathvqa_metrics.json) · [strata](results/qwen7b_pathvqa_strata.json) |
| `12189909` | Qwen2.5-VL-7B | vqarad | all | `qwen7b_vqarad` | ✅ done | [out](logs/medvlm_eval_12189909.out) · [err](logs/medvlm_eval_12189909.err) | [preds](results/qwen7b_vqarad.jsonl) · [metrics](results/qwen7b_vqarad_metrics.json) · [strata](results/qwen7b_vqarad_strata.json) |
| `12193557` | MedGemma-4B | pathvqa | all | `medgemma_pathvqa` | ✅ done | [out](logs/medvlm_eval_12193557.out) · [err](logs/medvlm_eval_12193557.err) | [preds](results/medgemma_pathvqa.jsonl) · [metrics](results/medgemma_pathvqa_metrics.json) · [strata](results/medgemma_pathvqa_strata.json) |
| `12193556` | MedGemma-4B | vqarad | all | `medgemma_vqarad` | ✅ done | [out](logs/medvlm_eval_12193556.out) · [err](logs/medvlm_eval_12193556.err) | [preds](results/medgemma_vqarad.jsonl) · [metrics](results/medgemma_vqarad_metrics.json) · [strata](results/medgemma_vqarad_strata.json) |
| `12190512` | LLaVA-1.6-7B | pathvqa | all | `llava16_pathvqa` | ✅ done | [out](logs/medvlm_eval_12190512.out) · [err](logs/medvlm_eval_12190512.err) | [preds](results/llava16_pathvqa.jsonl) · [metrics](results/llava16_pathvqa_metrics.json) · [strata](results/llava16_pathvqa_strata.json) |
| `12190511` | LLaVA-1.6-7B | vqarad | all | `llava16_vqarad` | ✅ done | [out](logs/medvlm_eval_12190511.out) · [err](logs/medvlm_eval_12190511.err) | [preds](results/llava16_vqarad.jsonl) · [metrics](results/llava16_vqarad_metrics.json) · [strata](results/llava16_vqarad_strata.json) |

## Severity (noise/blur sweep)

| Job | Model | Dataset | N | Tag | Status | Logs | Results |
|-----|-------|---------|---|-----|--------|------|---------|
| `12190617` | Qwen2.5-VL-7B | pathvqa | all | `qwen7b_sev_pathvqa` | ✅ done | [out](logs/medvlm_eval_12190617.out) · [err](logs/medvlm_eval_12190617.err) | [preds](results/qwen7b_sev_pathvqa.jsonl) · [metrics](results/qwen7b_sev_pathvqa_metrics.json) |
| `12190616` | Qwen2.5-VL-7B | vqarad | all | `qwen7b_sev_vqarad` | ✅ done | [out](logs/medvlm_eval_12190616.out) · [err](logs/medvlm_eval_12190616.err) | [preds](results/qwen7b_sev_vqarad.jsonl) · [metrics](results/qwen7b_sev_vqarad_metrics.json) |
| `12198625` | MedGemma-4B | pathvqa | all | `medgemma_sev_pathvqa` | ✅ done | [out](logs/medvlm_eval_12198625.out) · [err](logs/medvlm_eval_12198625.err) | [preds](results/medgemma_sev_pathvqa.jsonl) · [metrics](results/medgemma_sev_pathvqa_metrics.json) |
| `12209412` | MedGemma-4B | pathvqa | all | `medgemma_sev_pathvqa_r2` | ⚠️ check log | [out](logs/medvlm_eval_12209412.out) · [err](logs/medvlm_eval_12209412.err) | [preds](results/medgemma_sev_pathvqa_r2.jsonl) |
| `12198624` | MedGemma-4B | vqarad | all | `medgemma_sev_vqarad` | ✅ done | [out](logs/medvlm_eval_12198624.out) · [err](logs/medvlm_eval_12198624.err) | [preds](results/medgemma_sev_vqarad.jsonl) · [metrics](results/medgemma_sev_vqarad_metrics.json) |
| `12198623` | LLaVA-1.6-7B | pathvqa | all | `llava16_sev_pathvqa` | ✅ done | [out](logs/medvlm_eval_12198623.out) · [err](logs/medvlm_eval_12198623.err) | [preds](results/llava16_sev_pathvqa.jsonl) · [metrics](results/llava16_sev_pathvqa_metrics.json) |
| `12198622` | LLaVA-1.6-7B | vqarad | all | `llava16_sev_vqarad` | ✅ done | [out](logs/medvlm_eval_12198622.out) · [err](logs/medvlm_eval_12198622.err) | [preds](results/llava16_sev_vqarad.jsonl) · [metrics](results/llava16_sev_vqarad_metrics.json) |

## Verbalized confidence

| Job | Model | Dataset | N | Tag | Status | Logs | Results |
|-----|-------|---------|---|-----|--------|------|---------|
| `12190884` | Qwen2.5-VL-7B | pathvqa | all | `qwen7b_verb_pathvqa` | ✅ done | [out](logs/medvlm_eval_12190884.out) · [err](logs/medvlm_eval_12190884.err) | [preds](results/qwen7b_verb_pathvqa.jsonl) · [metrics](results/qwen7b_verb_pathvqa_metrics.json) |
| `12190883` | Qwen2.5-VL-7B | vqarad | all | `qwen7b_verb_vqarad` | ✅ done | [out](logs/medvlm_eval_12190883.out) · [err](logs/medvlm_eval_12190883.err) | [preds](results/qwen7b_verb_vqarad.jsonl) · [metrics](results/qwen7b_verb_vqarad_metrics.json) |
| `12198629` | MedGemma-4B | pathvqa | all | `medgemma_verb_pathvqa` | ✅ done | [out](logs/medvlm_eval_12198629.out) · [err](logs/medvlm_eval_12198629.err) | [preds](results/medgemma_verb_pathvqa.jsonl) · [metrics](results/medgemma_verb_pathvqa_metrics.json) |
| `12198628` | MedGemma-4B | vqarad | all | `medgemma_verb_vqarad` | ✅ done | [out](logs/medvlm_eval_12198628.out) · [err](logs/medvlm_eval_12198628.err) | [preds](results/medgemma_verb_vqarad.jsonl) · [metrics](results/medgemma_verb_vqarad_metrics.json) |
| `12198627` | LLaVA-1.6-7B | pathvqa | all | `llava16_verb_pathvqa` | ✅ done | [out](logs/medvlm_eval_12198627.out) · [err](logs/medvlm_eval_12198627.err) | [preds](results/llava16_verb_pathvqa.jsonl) · [metrics](results/llava16_verb_pathvqa_metrics.json) |
| `12198626` | LLaVA-1.6-7B | vqarad | all | `llava16_verb_vqarad` | ✅ done | [out](logs/medvlm_eval_12198626.out) · [err](logs/medvlm_eval_12198626.err) | [preds](results/llava16_verb_vqarad.jsonl) · [metrics](results/llava16_verb_vqarad_metrics.json) |

## Smoke test

| Job | Model | Dataset | N | Tag | Status | Logs | Results |
|-----|-------|---------|---|-----|--------|------|---------|
| `12189776` | Qwen2.5-VL-3B | vqarad | 20 | `smoke_qwen3b` | ✅ done | [out](logs/medvlm_eval_12189776.out) · [err](logs/medvlm_eval_12189776.err) | [preds](results/smoke_qwen3b.jsonl) · [metrics](results/smoke_qwen3b_metrics.json) |

## Key metrics — main runs (intact vs blank vs mismatch)

Accuracy and ECE per image condition. High blank/mismatch accuracy ⇒ the model is ignoring the image (hallucination signal); high ECE ⇒ poorly calibrated confidence.

| Model | Dataset | Acc intact | Acc blank | Acc mismatch | ECE intact |
|-------|---------|-----------|-----------|--------------|-----------|
| Qwen2.5-VL-7B | pathvqa | 0.338 | 0.246 | 0.250 | 0.481 |
| Qwen2.5-VL-7B | vqarad | 0.554 | 0.386 | 0.439 | 0.229 |
| MedGemma-4B | pathvqa | 0.293 | 0.139 | 0.173 | 0.527 |
| MedGemma-4B | vqarad | 0.548 | 0.231 | 0.337 | 0.351 |
| LLaVA-1.6-7B | pathvqa | 0.312 | 0.224 | 0.270 | 0.484 |
| LLaVA-1.6-7B | vqarad | 0.439 | 0.328 | 0.353 | 0.375 |

> Full per-condition metrics (mean confidence, high-confidence error rate, reliability bins) are in each `results/<tag>_metrics.json`.
