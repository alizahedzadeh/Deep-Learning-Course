#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Gemma-2 2B (Persian) on a deterministic sample of 200 test items
with one method at a time (Soft-Prompt, LoRA, Classic-FT).

- Loads dataset exactly as requested: hf_hub_download -> JSONL stream -> normalize -> pairs
- Reproduces split: Train/Val/Test with seed=42 (val ≈ 10% of train)
- Samples 200 items from the test split (seed=42) and evaluates ONLY those
- One-by-one generation (low VRAM) with tqdm progress bar
- Metrics (Persian-friendly):
    • BERTScore via bert_score.BERTScorer (ParsBERT; no baseline)
    • ROUGE-1/2/L (F1) via rouge_score.RougeScorer + Persian whitespace tokenizer
- Logs to console and file; saves predictions and metrics

Usage (tmux examples):
  tmux new -s eval_soft200 -d    "python eval_gemma_sample200.py --method soft-prompt --log-file logs/soft200.log"
  tmux new -s eval_lora200 -d    "python eval_gemma_sample200.py --method lora        --log-file logs/lora200.log"
  tmux new -s eval_classic200 -d "python eval_gemma_sample200.py --method classic-ft  --log-file logs/classic200.log"

Python deps:
  pip install transformers peft datasets huggingface_hub bert-score rouge-score tqdm pandas
"""

import os, re, gc, sys, json, time, argparse, logging, random
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer, AutoModelForCausalLM
from jinja2 import TemplateError

try:
    from peft import AutoPeftModelForCausalLM
except Exception:
    AutoPeftModelForCausalLM = None


# ---------------------------
# Defaults specific to YOUR runs
# ---------------------------
DEFAULTS = {
    "model_name": "google/gemma-2-2b-it",
    "seed": 42,
    "test_size": 0.1,
    "n_samples": 5,  # << only 200 from test split
    # Your tuned artifacts:
    "soft_adapter": "./gemma_soft_prompt/gemma2b_it_prompt_tuning_vt16_20250831_061713/prompt_adapter",
    "lora_adapter": "./gemma_lora/gemma2b_it_lora_r8_alpha16_20250831_193700/lora_adapter",
    "classic_model": "./gemma_classic_ft_2/gemma2b_it_classic_ft_first2_last2_20250901_072844/classic_ft_model",
    "out_dir": "./eval_outputs_sample200",
    "cache_dir": "./my_dataset_cache",
    "repo_id": "miladmim/slim-orca-dedup-chat-50k-persian",
    "repo_filename": "data.jsonl",
}


# ---------------------------
# Logging
# ---------------------------
def setup_logger(log_file: str = None, verbose: bool = True):
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------
# Dataset loading (YOUR WAY)
# ---------------------------
def load_pairs_from_hub(logger, repo_id: str, filename: str, cache_dir: str) -> Dataset:
    """
    Download JSONL via hf_hub_download, stream rows, normalize roles,
    filter for at least one user→assistant pair, then extract (system, prompt, response).
    Returns a `pairs` Dataset with columns: system, prompt, response.
    """
    logger.info(f"Downloading dataset file from hub: {repo_id}/{filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    logger.info(f"Local path: {path}")

    # 1) Stream + normalize rows into a consistent schema
    def rows():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, list):
                    yield {"conversations": obj}
                elif isinstance(obj, dict):
                    yield obj
                else:
                    yield {"text": str(obj)}

    # 2) Build Dataset object
    ds = Dataset.from_generator(rows, cache_dir=cache_dir)
    logger.info(f"Raw dataset: {ds}")

    # 3) Normalize roles (keep only user/assistant/system; drop others)
    def clean_roles(row):
        new_conv = []
        for turn in row.get("conversations", []):
            role = (turn.get("role", "") or "").strip().lower()
            if role in ("user", "assistant", "system"):
                new_conv.append({"role": role, "content": turn.get("content", "")})
        row["conversations"] = new_conv
        return row

    ds_cleaned = ds.map(clean_roles, desc="Normalize roles")
    ds_cleaned = ds_cleaned.filter(lambda x: len(x["conversations"]) > 0)
    logger.info(f"After role cleanup: {ds_cleaned}")

    # 4) Keep rows that contain at least one user→assistant pair
    def has_valid_pair(row):
        roles = [(t.get("role") or "").strip().lower() for t in row["conversations"]]
        for i in range(len(roles) - 1):
            if roles[i] == "user" and roles[i + 1] == "assistant":
                return True
        return False

    ds_valid = ds_cleaned.filter(has_valid_pair, desc="Filter rows with at least one user→assistant pair")

    # 5) Extract the first user→assistant pair (+ optional system)
    def extract_first_pair(row):
        system = None
        prompt = None
        response = None

        # first system (if any)
        for turn in row["conversations"]:
            if (turn.get("role") or "").strip().lower() == "system":
                system = turn.get("content", "")
                break

        saw_user = False
        for turn in row["conversations"]:
            role = (turn.get("role") or "").strip().lower()
            text = turn.get("content", "")
            if role == "user" and prompt is None:
                prompt = text
                saw_user = True
            elif role == "assistant" and saw_user and response is None:
                response = text
                break

        return {"system": system, "prompt": prompt, "response": response}

    pairs = ds_valid.map(
        extract_first_pair,
        remove_columns=[c for c in ds_valid.column_names if c != "conversations"],
        desc="Extract first user→assistant pair"
    )

    # 6) Final filter: remove malformed rows
    def ok(ex):
        return isinstance(ex["prompt"], str) and isinstance(ex["response"], str) and ex["prompt"].strip() and ex["response"].strip()

    pairs = pairs.filter(ok, desc="Drop malformed pairs")
    logger.info(f"Final pairs dataset: {pairs} (columns={pairs.column_names})")
    return pairs


# ---------------------------
# Split, tokenizer, chat template
# ---------------------------
def build_splits(logger, pairs, test_size: float, seed: int) -> DatasetDict:
    split = pairs.train_test_split(test_size=test_size, seed=seed)
    train_val = split["train"].train_test_split(test_size=0.1111, seed=seed)  # ≈ 10% of 90% → val
    dataset = DatasetDict({"train": train_val["train"], "validation": train_val["test"], "test": split["test"]})
    logger.info(f"Split sizes: Train={len(dataset['train'])} | Val={len(dataset['validation'])} | Test={len(dataset['test'])}")
    return dataset


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def render_chat(tokenizer, messages):
    try:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    except TemplateError:
        sys_txt = ""
        if messages and messages[0].get("role") == "system":
            sys_txt = (messages[0].get("content") or "").strip()
            messages = messages[1:]
        user_txt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        merged = (sys_txt + "\n\n" if sys_txt else "") + user_txt
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            add_generation_prompt=True, return_tensors="pt"
        )
    attn = (ids != tokenizer.pad_token_id).long()
    return ids, attn


# ---------------------------
# Generation & memory helpers
# ---------------------------
def generate_one(model, tokenizer, user_text, system_text=None, gen_cfg=None):
    msgs = []
    if system_text and system_text.strip():
        msgs.append({"role": "system", "content": system_text.strip()})
    msgs.append({"role": "user", "content": (user_text or "").strip()})
    ids, attn = render_chat(tokenizer, msgs)
    ids, attn = ids.to(model.device), attn.to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=attn, **gen_cfg)
    new_tokens = out[0, ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del ids, attn, out, new_tokens
    return text


def empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------
# Normalization + metrics (NEW)
# ---------------------------
_ARABIC_YEH = "\u064A"; _PERSIAN_YEH = "\u06CC"
_ARABIC_KEHEH = "\u0643"; _PERSIAN_KEHEH = "\u06A9"
_TATWEEL = "\u0640"; _ZWNJ = "\u200c"
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

def normalize_fa(s: str) -> str:
    if s is None: return ""
    s = s.replace(_ARABIC_YEH, _PERSIAN_YEH).replace(_ARABIC_KEHEH, _PERSIAN_KEHEH)
    s = s.replace(_TATWEEL, "").replace(_ZWNJ, " ")
    s = _DIACRITICS.sub("", s)
    s = re.sub(r"[“”«»]", '"', s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

class PersianWhitespaceTokenizer:
    def tokenize(self, text: str):
        return text.split()

def compute_metrics(preds, refs, logger):
    """
    Persian-friendly metrics:
      - BERTScore (ParsBERT via bert_score.BERTScorer, no baseline)
      - ROUGE-1/2/L (F1) via rouge_score.RougeScorer + Persian whitespace tokenizer
    """
    preds_n = [normalize_fa(x) for x in preds]
    refs_n  = [normalize_fa(x) for x in refs]

    # ---- BERTScore (ParsBERT)
    if logger: logger.info("Computing BERTScore with ParsBERT (no baseline)…")
    from bert_score import BERTScorer
    scorer = BERTScorer(
        model_type="HooshvareLab/bert-fa-base-uncased",
        num_layers=12,
        rescale_with_baseline=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32
    )
    P, R, F1 = scorer.score(preds_n, refs_n)
    bert_p  = float(P.mean().item())
    bert_r  = float(R.mean().item())
    bert_f1 = float(F1.mean().item())

    # ---- ROUGE (whitespace tokenizer)
    if logger: logger.info("Computing ROUGE-1/2/L with rouge-score (Persian whitespace tokenizer)…")
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(
        ['rouge1','rouge2','rougeL'],
        use_stemmer=False,
        tokenizer=PersianWhitespaceTokenizer()
    )
    r1_list, r2_list, rL_list = [], [], []
    for ref, pred in zip(refs_n, preds_n):
        s = rs.score(ref, pred)
        r1_list.append(s['rouge1'].fmeasure)
        r2_list.append(s['rouge2'].fmeasure)
        rL_list.append(s['rougeL'].fmeasure)

    metrics = {
        "bert_p":  float(sum(r1_list)*0 + bert_p),  # keep numeric type stable
        "bert_r":  float(bert_r),
        "bert_f1": float(bert_f1),
        "rouge1":  float(sum(r1_list)/max(1,len(r1_list))),
        "rouge2":  float(sum(r2_list)/max(1,len(r2_list))),
        "rougeL":  float(sum(rL_list)/max(1,len(rL_list))),
        "n_samples": len(preds_n),
        "bert_model": "HooshvareLab/bert-fa-base-uncased",
        "bert_layers": 12
    }
    return metrics


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma-2 tuning methods on a deterministic 200-sample of test (low VRAM).")
    parser.add_argument("--method", choices=["soft-prompt", "lora", "classic-ft"], required=True)
    parser.add_argument("--soft-adapter", default=DEFAULTS["soft_adapter"])
    parser.add_argument("--lora-adapter", default=DEFAULTS["lora_adapter"])
    parser.add_argument("--classic-model", default=DEFAULTS["classic_model"])
    parser.add_argument("--model-name", default=DEFAULTS["model_name"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--test-size", type=float, default=DEFAULTS["test_size"])
    parser.add_argument("--n-samples", type=int, default=DEFAULTS["n_samples"])
    parser.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    parser.add_argument("--cache-dir", default=DEFAULTS["cache_dir"])
    parser.add_argument("--repo-id", default=DEFAULTS["repo_id"])
    parser.add_argument("--repo-filename", default=DEFAULTS["repo_filename"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep-penalty", type=float, default=1.05)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--ckpt-every", type=int, default=200, help="Write rolling checkpoint every N samples.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.log_file or os.path.join(args.out_dir, f"eval200_{args.method.replace('-','_')}.log"))

    # 0) Dataset -> pairs (your exact pipeline)
    pairs = load_pairs_from_hub(logger, args.repo_id, args.repo_filename, args.cache_dir)
    dataset = build_splits(logger, pairs, args.test_size, args.seed)

    # ---- Deterministic SAMPLE from test ----
    test_ds = dataset["test"]
    total_test = len(test_ds)
    rng = random.Random(args.seed)
    k = min(args.n_samples, total_test)
    picked_indices = sorted(rng.sample(range(total_test), k=k))
    logger.info(f"Sampling {k} items from test (size={total_test}) with seed={args.seed}.")
    sampled = [test_ds[i] for i in picked_indices]

    # 1) Tokenizer & gen config
    tokenizer = build_tokenizer(args.model_name)
    gen_cfg = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.rep_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 2) Load ONE model only (VRAM-safe)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto"

    if args.method == "soft-prompt":
        if AutoPeftModelForCausalLM is None:
            logger.error("peft is not installed. pip install peft")
            sys.exit(1)
        load_path = args.soft_adapter
        logger.info(f"Loading Soft-Prompt adapter: {load_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            load_path, device_map=device_map, torch_dtype=dtype, attn_implementation="eager"
        ).eval()

    elif args.method == "lora":
        if AutoPeftModelForCausalLM is None:
            logger.error("peft is not installed. pip install peft")
            sys.exit(1)
        load_path = args.lora_adapter
        logger.info(f"Loading LoRA adapter: {load_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            load_path, device_map=device_map, torch_dtype=dtype, attn_implementation="eager"
        ).eval()

    else:  # classic-ft
        load_path = args.classic_model
        logger.info(f"Loading Classic-FT model: {load_path}")
        model = AutoModelForCausalLM.from_pretrained(
            load_path, device_map=device_map, torch_dtype=dtype, attn_implementation="eager"
        ).eval()

    # 3) Output paths
    stamp = time.strftime("%Y%m%d_%H%M%S")
    method_tag = f"{args.method.replace('-', '_')}_sample{args.n_samples}"
    pred_path = os.path.join(args.out_dir, f"predictions_{method_tag}_{stamp}.jsonl")
    ckpt_path = os.path.join(args.out_dir, f"predictions_{method_tag}_{stamp}.ckpt.jsonl")
    metrics_json = os.path.join(args.out_dir, f"metrics_{method_tag}_{stamp}.json")
    metrics_csv  = os.path.join(args.out_dir, f"metrics_{method_tag}_{stamp}.csv")

    logger.info(f"Predictions JSONL: {pred_path}")
    logger.info(f"Checkpoint JSONL : {ckpt_path}")
    logger.info(f"Metrics JSON     : {metrics_json}")
    logger.info(f"Metrics CSV      : {metrics_csv}")

    # 4) Generate one-by-one (progress bar + rolling ckpt)
    preds, refs = [], []
    t0 = time.time()
    with open(ckpt_path, "w", encoding="utf-8") as f_ckpt:
        for i, ex in enumerate(tqdm(sampled, desc=f"{args.method}-200", dynamic_ncols=True)):
            sys_txt = ex.get("system") or ""
            user = ex["prompt"]
            gold = ex["response"]
            try:
                pred = generate_one(model, tokenizer, user, sys_txt, gen_cfg)
            except RuntimeError as e:
                logger.error(f"Generation OOM/error at idx {i}: {e}")
                empty_cuda(); gc.collect()
                try:
                    gen_cfg_short = {**gen_cfg, "max_new_tokens": max(64, int(args.max_new_tokens/2))}
                    pred = generate_one(model, tokenizer, user, sys_txt, gen_cfg_short)
                    logger.warning(f"Recovered with shorter max_new_tokens at idx {i}.")
                except Exception as e2:
                    logger.exception(f"Failed again at idx {i}; writing empty prediction. {e2}")
                    pred = ""

            preds.append(pred)
            refs.append(gold)

            if (i + 1) % args.ckpt_every == 0:
                f_ckpt.write(json.dumps({"i": i, "prediction": pred, "reference": gold}, ensure_ascii=False) + "\n")
                f_ckpt.flush()
                empty_cuda(); gc.collect()

    logger.info(f"Finished {len(sampled)} examples in {time.time()-t0:.1f}s")
    del model
    empty_cuda(); gc.collect()

    # 5) Save full predictions
    with open(pred_path, "w", encoding="utf-8") as f_out:
        for p, r in zip(preds, refs):
            f_out.write(json.dumps({"prediction": p, "reference": r}, ensure_ascii=False) + "\n")
    logger.info("Saved predictions.")

    # 6) Metrics
    metrics = compute_metrics(preds, refs, logger)
    logger.info("Metrics:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(metrics_json, "w", encoding="utf-8") as f_js:
        json.dump(metrics, f_js, indent=2, ensure_ascii=False)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    logger.info("All done.")


if __name__ == "__main__":
    main()
