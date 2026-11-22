# CLEAR: Calibrating Medical Vision-Language Reasoning Under Image Corruption

Inference-only evaluation of open medical vision-language models (VLMs) on
medical VQA. For each `(image, question)` the model is queried under several
**image conditions** and we measure whether it actually *uses the image* and
whether its confidence is *calibrated*:

- **intact** — the real image
- **blank** — a uniform gray image (no visual evidence)
- **mismatch** — a different, randomly chosen image from the same dataset
- **severity** — graded corruptions `noise@<sigma>` and `blur@<radius>`

From the responses we compute accuracy, **ECE** (expected calibration error),
**image-drop consistency** (how often the answer is unchanged when the image is
blanked/swapped — the hallucination signal), high-confidence error rate, and
reliability curves.

## Repository layout

```
code/eval/
  run_eval.py        # run a model over a dataset × conditions -> results/<tag>.jsonl
  score.py           # accuracy / ECE / image-drop consistency -> <tag>_metrics.json
  bootstrap_ci.py    # bootstrap confidence intervals (all / closed / open strata)
  stratify.py        # split metrics by closed- vs open-ended questions
  severity_curve.py  # plot accuracy/consistency vs corruption severity
  plot_reliability.py# reliability (calibration) curves from *_metrics.json
  run.sbatch         # Slurm wrapper (1× A100)
results/             # generated .jsonl predictions + *_metrics.json
logs/                # Slurm job logs
paper/               # workshop paper (LaTeX + figures)
WORKSHOP_PLAN.md     # project scope / plan
```

## Setup

### 1. Environment

Python ≥ 3.10 with a CUDA-capable GPU (experiments used a single A100; a 24 GB
GPU is enough for the 3B/4B models, ~16–24 GB for the 7B models in bf16).

```bash
conda create -n clear-vlm python=3.11 -y
conda activate clear-vlm

# PyTorch with CUDA (match your CUDA version — see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install "transformers>=4.45" accelerate datasets huggingface_hub \
            pillow numpy matplotlib
```

> Qwen2.5-VL and MedGemma need a recent `transformers` (≥ 4.45). If
> `AutoModelForImageTextToText` fails to load a model, upgrade transformers.

### 2. Hugging Face authentication

Model weights and datasets are pulled from the Hugging Face Hub. Some models
(e.g. `google/medgemma-4b-it`) are **gated** — accept their license on the model
page first, then log in with a token that has `read` access:

```bash
huggingface-cli login            # paste a token from https://huggingface.co/settings/tokens
# or, non-interactively:
export HF_TOKEN=hf_xxx
```

> Do **not** hardcode your token in committed files. Use `HF_TOKEN` /
> `huggingface-cli login` instead.

### 3. Cache location (optional)

Datasets + weights are large; point the HF cache at a roomy disk:

```bash
export HF_HOME=/path/to/.hf_cache
```

## Datasets

Both datasets are downloaded automatically from the Hub on first run — no manual
preparation needed. They are small (images are KB-sized, not gigapixel slides);
the real storage cost is the model weights.

| Key       | Hub id                          | Domain     | Images | Q&A pairs |
|-----------|---------------------------------|------------|--------|-----------|
| `vqarad`  | `flaviagiammarino/vqa-rad`      | radiology  | ~315   | ~3.5k     |
| `pathvqa` | `flaviagiammarino/path-vqa`     | pathology  | ~5k    | ~32.8k    |

To pre-download onto a login node (useful if compute nodes are offline):

```bash
export HF_HOME=/path/to/.hf_cache
huggingface-cli download flaviagiammarino/vqa-rad  --repo-type dataset
huggingface-cli download flaviagiammarino/path-vqa --repo-type dataset
```

## Models evaluated

The pipeline is model-agnostic (`AutoProcessor` + `AutoModelForImageTextToText`
with each model's chat template). Reported runs use:

- `Qwen/Qwen2.5-VL-7B-Instruct` (and `-3B-Instruct`)
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `google/medgemma-4b-it` *(gated)*

## Running

### Single run (direct)

```bash
python code/eval/run_eval.py \
  --model google/medgemma-4b-it \
  --dataset vqarad \
  --conditions intact,blank,mismatch \
  --n -1 \                       # -1 = all questions; use e.g. 50 for a smoke test
  --out results/medgemma_vqarad.jsonl

python code/eval/score.py \
  --pred results/medgemma_vqarad.jsonl \
  --out  results/medgemma_vqarad_metrics.json
```

Useful `run_eval.py` flags: `--verbalized` (also elicit a spoken "Confidence:
N%"), `--max_new_tokens`, `--seed`, `--dtype bfloat16`, and severity conditions
such as `--conditions intact,noise@0.1,blur@3`.

### Slurm

```bash
# sbatch code/eval/run.sbatch <MODEL_ID> <DATASET> <N> <OUT_TAG> [CONDITIONS] [VERBALIZED]
sbatch code/eval/run.sbatch Qwen/Qwen2.5-VL-3B-Instruct vqarad 50 smoke
sbatch code/eval/run.sbatch google/medgemma-4b-it pathvqa -1 medgemma_pathvqa
```

The sbatch script sets `HF_HOME`, `HF_HUB_OFFLINE=1` (compute nodes assumed
offline — pre-download first), activates conda, runs `run_eval.py`, then scores.
Edit the `#SBATCH` partition/account lines and the `cd` path for your cluster.

### Analysis & figures

```bash
# Bootstrap CIs, overall and by question type
python code/eval/bootstrap_ci.py --pred results/qwen7b_vqarad.jsonl --stratum closed

# Reliability (calibration) curves
python code/eval/plot_reliability.py \
  --metrics results/qwen7b_vqarad_metrics.json results/medgemma_vqarad_metrics.json \
  --titles "Qwen2.5-VL" "MedGemma" \
  --out paper/figures/reliability.pdf

# Severity curves (needs runs with noise@/blur@ conditions, e.g. *_sev_*.jsonl)
python code/eval/severity_curve.py \
  --pred results/qwen7b_sev_vqarad.jsonl results/qwen7b_sev_pathvqa.jsonl \
  --titles VQA-RAD PathVQA \
  --out paper/figures/severity_qwen7b.pdf --csv results/qwen7b_sev.csv
```

## Outputs

- `results/<tag>.jsonl` — one row per question: per-condition answer text and the
  mean-token-probability confidence.
- `results/<tag>_metrics.json` — accuracy, ECE, high-confidence error rate,
  image-drop consistency, and reliability bins.
- Result tags follow `<model>_<dataset>` with suffixes `_sev` (severity),
  `_verb` (verbalized confidence), and `_strata` (by question type).

See `WORKSHOP_PLAN.md` and `paper/` for the accompanying short paper.
