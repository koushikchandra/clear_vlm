#!/usr/bin/env python
"""
Faithfulness + calibration audit of open medical VLMs.

For each (image, question) we query the model under three image conditions:
  intact   -> the real image
  blank    -> a uniform gray image (no visual evidence)
  mismatch -> a different, randomly chosen image from the same dataset

For every response we record:
  - the generated answer text
  - a confidence score = mean token probability of the generated answer
    (the model's own softmax confidence; used for ECE / reliability)

Downstream (score.py) computes accuracy, image-drop consistency, ECE,
reliability curves, and high-confidence-error rate.

Model-agnostic: uses AutoProcessor + AutoModelForImageTextToText with the
model's chat template, so it works for MedGemma, Qwen2.5-VL, LLaVA-1.6, etc.
"""
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASETS = {
    "vqarad":  "flaviagiammarino/vqa-rad",
    "pathvqa": "flaviagiammarino/path-vqa",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. google/medgemma-4b-it")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=-1,
                   help="limit #questions (-1 = all)")
    p.add_argument("--conditions", default="intact,blank,mismatch")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output .jsonl path")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--verbalized", action="store_true",
                   help="also elicit a verbalized 0-100%% confidence")
    return p.parse_args()


def blank_like(img: Image.Image) -> Image.Image:
    return Image.new("RGB", img.size, (128, 128, 128))


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise; sigma in 0..1 as fraction of full 255 range."""
    a = np.asarray(img, dtype=np.float32)
    a = a + np.random.normal(0, sigma * 255.0, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def blur(img: Image.Image, radius: float) -> Image.Image:
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_variant(cond: str, img: Image.Image, mism_img: Image.Image) -> Image.Image:
    """Resolve a condition name to an image. Supports severity suffixes:
    noise@0.1, blur@3 ; plus intact/blank/mismatch."""
    if cond == "intact":
        return img
    if cond == "blank":
        return blank_like(img)
    if cond == "mismatch":
        return mism_img
    if cond.startswith("noise@"):
        return gaussian_noise(img, float(cond.split("@")[1]))
    if cond.startswith("blur@"):
        return blur(img, float(cond.split("@")[1]))
    raise ValueError(f"unknown condition: {cond}")


import re
_VCONF_RE = re.compile(r"confidence[:\s]*([0-9]{1,3})\s*%?", re.IGNORECASE)


def parse_verbalized(text):
    """Extract answer + verbalized confidence in [0,1] from a response of the
    form '<answer> ... Confidence: NN%'. Returns (answer_text, vconf or None)."""
    m = _VCONF_RE.search(text)
    vconf = None
    if m:
        vconf = max(0.0, min(1.0, float(m.group(1)) / 100.0))
        text = text[:m.start()].strip().rstrip(".,;")
    # drop a leading "Answer:" label if present
    text = re.sub(r"^\s*answer[:\s]*", "", text, flags=re.IGNORECASE).strip()
    return text, vconf


@torch.no_grad()
def answer(model, processor, image, question, max_new_tokens, verbalized=False):
    """Return (answer_text, mean_token_prob, vconf) for one image+question.
    vconf is None unless verbalized=True and a percentage was parsed."""
    if verbalized:
        prompt = (f"{question}\nAnswer the question in a few words, then on a new "
                  f"line write 'Confidence: N%' where N (0-100) is how certain "
                  f"you are that your answer is correct.")
        max_new_tokens = max(max_new_tokens, 48)
    else:
        prompt = f"{question}\nAnswer concisely."
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    gen_ids = out.sequences[0, in_len:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    # confidence = mean prob of the chosen token at each generated step
    probs = []
    for step, score in enumerate(out.scores):
        if step >= gen_ids.shape[0]:
            break
        tok = gen_ids[step]
        p = torch.softmax(score[0].float(), dim=-1)[tok].item()
        probs.append(p)
    conf = float(np.mean(probs)) if probs else 0.0

    vconf = None
    if verbalized:
        text, vconf = parse_verbalized(text)
    return text, conf, vconf


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText

    ds = load_dataset(DATASETS[args.dataset], split=args.split)
    n = ds.num_rows if args.n < 0 else min(args.n, ds.num_rows)
    conditions = args.conditions.split(",")

    # precompute a mismatch permutation (each item paired with a different image)
    idx = list(range(ds.num_rows))
    shift = np.random.permutation(ds.num_rows)
    mism_for = {i: int(shift[i]) if int(shift[i]) != i
                else int(shift[(i + 1) % ds.num_rows]) for i in idx}

    print(f"[load] model={args.model} dtype={args.dtype}", flush=True)
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
    ).eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(n):
            row = ds[i]
            q, gold = row["question"], str(row["answer"]).strip()
            img = row["image"].convert("RGB")
            mism_img = ds[mism_for[i]]["image"].convert("RGB")
            rec = {"idx": i, "question": q, "gold": gold, "dataset": args.dataset}
            for cond in conditions:
                variant = make_variant(cond, img, mism_img)
                a, c, vc = answer(model, processor, variant, q,
                                  args.max_new_tokens, verbalized=args.verbalized)
                rec[f"{cond}_answer"] = a
                rec[f"{cond}_conf"] = c
                if args.verbalized:
                    rec[f"{cond}_vconf"] = vc
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{n}] q='{q[:40]}' "
                      f"intact='{rec.get('intact_answer','')[:20]}'", flush=True)

    print(f"[done] wrote {n} records x {len(conditions)} conditions -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
