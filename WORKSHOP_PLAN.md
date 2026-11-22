# Med-Reasoner @ NeurIPS 2026 — Short Paper Working Plan

## Target venue
**Med-Reasoner Workshop** — "Medical Reasoning with Vision Language Foundation Models"
- CFP: https://med-reasoner.github.io/neurips2026/call_for_paper.html
- **Submission deadline:** Sept 5, 2026 (11:59 PM AoE) — ~4 days out
- Notification: Sept 29, 2026 · Workshop: Dec 12–13, 2026 (Atlanta)
- Format: **Short paper = 4 pages max** (refs excess OK), official NeurIPS 2026 template

## Key constraint / scope check
- Workshop requires a **vision-language / medical-imaging** angle.
- Existing lab work (`Graph_Transformer/multimodal/`) is **molecular only** (mutation + CNV + gene expression → pathway graph transformer → Cox/classification). It does **not** fit this venue as-is.
- Decision: **do not reuse the omics model**; do a small, self-contained **VLM evaluation** paper instead.

## Chosen strategy: evaluation/benchmarking (NO training)
Rationale — fastest, lowest risk in a 4-day window:
- No model training (biggest time sink removed) — inference only.
- Small public datasets, fully runnable end-to-end.
- Objective, table-ready metrics; no subjective grading.
- Fits CFP tracks: **Evaluation & Benchmarking** + **Hallucination mitigation** + **CoT reasoning**.

## Datasets (small — a plus for a short paper)
| Dataset  | Images            | Q&A pairs | Size        | Notes                          |
|----------|-------------------|-----------|-------------|--------------------------------|
| VQA-RAD  | ~315 radiology    | ~3,515    | few hundred MB | **start here** — smallest/cleanest |
| SLAKE    | ~642              | ~14k      | ~200 MB     | has bounding boxes (grounding) |
| PathVQA  | ~4,998 pathology  | ~32.8k    | ~700 MB–1 GB| pathology — plays to background |

- Images are small (KB each), not gigapixel WSIs.
- Real "size" cost is the **model weights** (e.g. LLaVA-Med 7B ≈ 14 GB, 13B ≈ 26 GB) — one-time download.

## Model
- Off-the-shelf **open medical VLM** (e.g. LLaVA-Med or similar). No fine-tuning.

## Novelty — LITERATURE CHECK DONE (2026-09-01). Verdict: original 3 angles are TAKEN.
Vanilla "blank image → measure answer change" is known/not novel. Lit-check results:

1. **Wrong-scan detection — TAKEN.** HALT-MedVQA (github.com/knowlab/halt-medvqa) modifies
   VQA-RAD/PathVQA/PMC-VQA with *mismatched image queries*; MedVH (arxiv 2407.02730) has a
   "Wrongful Image" task. This is exactly the proposed probe — already a benchmark.
2. **CoT faithfulness — CROWDED & RECENT (2025–26).** Evaluating Reasoning Faithfulness in
   Medical VLMs (2510.11196), S-Chain (2510.22728), SPD-Faith Bench (2602.07833),
   CoRGI (2508.00378), step-level grounding-faithfulness (2603.06828).
3. **Stratified faithfulness benchmark — COVERED** by HALT-MedVQA and MedVH across same datasets.

### New lead candidate (gap found, still cheap):
**Calibration under image corruption.** Prior work measures accuracy/refusal on blank/mismatched
images; it does NOT clearly measure whether model **confidence is calibrated** in those conditions
(ECE / reliability curves as image degrades). Question: "Are medical VLMs appropriately *uncertain*
when they can't actually see the image?" Reuses HALT-MedVQA's blank/mismatch pipeline — builds ON
an existing benchmark instead of recreating one.

### >>> CALIBRATION ANGLE ALSO TAKEN (checked 2026-09-01). NOT novel.
- Confidence Calibration for Multimodal LLMs, MICCAI 2025 (arxiv 2606.19950) — ECE on 3 med-VQA sets
- CARE: Confidence-Aware Reasoning for Reliable Medical VQA (arxiv 2608.10964)
- Verbalized Uncertainty Calibration in Medical VQA (arxiv 2606.27023)
- Assessing VLM Reliability Under Corruption & Bias (arxiv 2607.01973)
- Verbalized Uncertainty Estimation Robustness, TrustNLP 2025 (aclanthology 2025.trustnlp-main.16)
  ^ reports the EXACT proposed result: corruption severity ↑ => ECE ↑ while confidence stays high.

## ===== STRATEGIC VERDICT (2026-09-01) =====
Pure medical-VLM EVALUATION space is SATURATED. Four angles checked, all occupied
(wrong-scan, CoT-faithfulness, stratified benchmark, calibration-under-corruption).
Any abstract "clever eval twist" is likely already run by someone with a GPU cluster.
=> Do NOT chase another pure-eval angle in 4 days (near-duplicate risk).

### Real moat = the lab's MOLECULAR / PATHWAY expertise (no one else at this workshop has it).
Under-explored intersection: connect imaging VLMs with MOLECULAR ground truth.
Candidate: does a medical VLM's histopathology reasoning agree with the molecular
subtype (e.g. PAM50/stage) the same TCGA case actually has? Bridges omics + pathway + VLM.
- Defensible novelty (not a crowded lane).
- More work than a pure eval paper, BUT "easy + novel" proved empty here.
- Leverages existing lab assets (TCGA cohorts, PAM50/stage labels, pathway pipeline).

### DECISION PENDING (user): incremental-but-safe eval  vs.  harder-but-defensible omics×VLM.
### >>> USER CHOSE OPTION A (incremental-but-safe eval), 2026-09-01.

## ===== OPTION A — LOCKED SCOPE & INFRA =====
Title (working): "How much do open medical VLMs actually rely on the image?
A faithfulness + calibration audit on pathology VQA."
Delta vs prior: pathology focus + newest open medical VLMs + faithfulness AND
calibration audited together (prior work does them separately, mostly radiology).

### Datasets (cached offline in .hf_cache/):
- VQA-RAD  : test=451  (radiology contrast)
- PathVQA  : test=6719 (pathology — primary, plays to background)

### Conditions: intact / blank (gray) / mismatch (random other image).
### Metrics: accuracy, image-drop consistency vs intact, ECE + reliability bins,
### high-confidence-error rate (conf>0.8). Confidence = mean generated-token prob.

### Code (code/eval/):
- run_eval.py  : model-agnostic inference (AutoProcessor + AutoModelForImageTextToText)
- score.py     : all metrics + console table + metrics.json
- run.sbatch   : SLURM launcher — `sbatch run.sbatch <MODEL> <DATASET> <N> <TAG>`

### Compute: SLURM partition `nova`, gpu:a100. Compute nodes assumed OFFLINE ->
### weights/datasets pre-downloaded on login node to .hf_cache, HF_HUB_OFFLINE=1.

### Model roster:
- Qwen/Qwen2.5-VL-3B-Instruct  : cached, used to VALIDATE pipeline (general VLM).
- google/medgemma-4b-it        : TARGET medical VLM — GATED, needs user HF token + license accept.
- (add 1 more medical VLM, e.g. HuatuoGPT-Vision-7B, for the roster.)

### Status (2026-09-01 eve): FULL RUNS DONE for Qwen2.5-VL-7B.
- VQA-RAD  n=451  (job 12189909, 9 min):  ECE intact 0.229 -> blank 0.427; hi-conf-err up to .47
- PathVQA  n=6719 (job 12189910, 1h50m):  ECE intact 0.481 -> mismatch 0.576; hi-conf-err .64
- Key finding: confidence stays high (or rises) when image is blanked/mismatched;
  models detect a BLANK image better than a plausible WRONG one (worse clinical failure).
- paper.tex: abstract + Table 1 (both datasets) filled with REAL numbers; compiles (3 pp).
- results/{qwen7b_vqarad,qwen7b_pathvqa}.jsonl + *_metrics.json saved.

### DONE since: reliability figure (code/eval/plot_reliability.py -> figures/reliability_qwen7b.pdf);
### full prose drafted for Intro/Method/Experiments/Conclusion; bib expanded (calib + model refs);
### paper compiles (5 pp incl refs). 2nd model = LLaVA-1.6-Mistral-7B (NOT HuatuoGPT: that is a
### custom LlavaQwen2 arch w/ no HF auto-loader -> too fragile for the timeline).
### LLaVA runs submitted: 12190511 (vqarad), 12190512 (pathvqa) -- waiter running.

## ===== INTEGRATION DONE (2026-09-02) — paper now 7pp, compiles clean. =====
- Table 1: BOTH models (Qwen2.5-VL-7B + LLaVA-1.6-Mistral-7B) x 2 datasets x 3 conds. Same
  qualitative pattern in both families (mismatch keeps conf high; blank partially detected).
- Reliability fig -> figures/reliability_both.pdf (4 panels: {Qwen,LLaVA} x {VQA-RAD,PathVQA}).
- Severity subsec + figures/severity_qwen7b.pdf (blur degrades acc, conf flat/rises => ECE up;
  noise negligible). CSV: results/qwen7b_sev.csv.
- Verbalized subsec + Table 2 (tab:verbalized): stated conf HIGHER & worse-calibrated than
  token prob; on VQA-RAD blank the model verbally claims MORE conf (.893) than intact (.883).
  Eliciting confidence also inflates token-prob conf (intact ECE .23->.33).
- Discussion (clinical implications + mitigations) + Limitations added.
- Related Work CoT-faithfulness todo resolved; contributions updated (no more \todo{two}).
- Only remaining \todo: authors/affiliation/email (kept anonymous for double-blind).

### 3rd MODEL = MedGemma-4b-it (medical VLM): GATED. Download DENIED with user token
### (hf_...AXaTG) -> account must click "Agree and access" at hf.co/google/medgemma-4b-it.
### HuatuoGPT-Vision-7B IS already cached (fallback, but custom LlavaQwen2 arch = fragile).

## ===== SCOPE CHANGE: user wants 8-page LONG paper (CFP long = 8pp max). =====
Plan to fill 8pp with REAL substance (not padding):
- [DONE] closed-vs-open stratification (code/eval/stratify.py) -> Table 2 in paper.
- [CODE READY] corruption-severity sweep: run_eval.py now supports blur@R / noise@sigma;
  severity_curve.py plots acc/conf/ECE vs severity. Jobs 12190616 (vqarad) + 12190617 (pathvqa)
  QUEUED (conds: intact,blur@1,blur@3,blur@5,noise@0.05,noise@0.1,noise@0.2).
- [QUEUED] 2nd model LLaVA-1.6: jobs 12190511/12190512 -> Table 1 rows + regen reliability fig.
- [TODO] verbalized-confidence variant (ask model to state confidence; compare to token-prob) -- needs code.
- [TODO] expand Related Work + add Discussion (clinical implications, mitigations) + Limitations.

### CLUSTER NOTE: hitting AssocGrpGRES GPU cap (user's other jobs running) -> medvlm jobs
### run serially as slots free. Waiter b7zmp18o3 watching LLaVA completion.

### Analysis code inventory (code/eval/): run_eval.py, score.py, stratify.py,
### plot_reliability.py, severity_curve.py, run.sbatch (5th arg = conditions).

### STILL TODO after runs land: confirm single/double blind; verify bib placeholders;
### length pass to <=8pp; regen figures with both models.

### Key references pulled (for related-work section):
- HALT-MedVQA — Hallucination Benchmark in Medical VQA (arxiv 2401.05827), github knowlab/halt-medvqa
- MedVH (arxiv 2407.02730 / PMC12363988) — systematic hallucination eval, Wrongful Image task
- Evaluating Reasoning Faithfulness in Medical VLMs (arxiv 2510.11196)
- S-Chain: Structured Visual CoT for Medicine (arxiv 2510.22728)
- Robustness of VLMs to medical artifacts (nature s41746-025-02108-w)

## Metrics
- Accuracy (baseline, real image)
- **Image-drop / wrong-image consistency** — fraction of answers unchanged when image is blanked/swapped
  (high = model ignores image = hallucination)
- Robustness under mild perturbation (noise/blur)

## ~4-day timeline
- **Day 1:** confirm cluster GPU + disk quota; download VQA-RAD; get one open medical VLM running inference.
- **Day 2:** run 3 conditions (real / blanked / wrong image); log answers.
- **Day 3:** compute accuracy + consistency/faithfulness metrics; build tables + qualitative examples.
- **Day 4:** write 4 pages (NeurIPS template) + polish. Add PathVQA if time allows.

## Open decisions (waiting on user)
- [ ] Confirm novelty angle (1 / 2 / 3) — pending lit-check.
- [ ] Do we have local imaging data/model access, or download fresh?
- [ ] Commit to Med-Reasoner vs. another venue for the omics work.

## Deliverable
4-page short paper: setup + one probe method + 2 tables + 1 figure.
