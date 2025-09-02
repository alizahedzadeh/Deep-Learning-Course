import os
import math
import torch
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# Logging
# =========================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_log_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"=== Training Session Started: {timestamp} ===")
    logger.info(f"Log file: {log_file}")
    return logger, timestamp

BASE_OUTPUT_DIR = "./gemma_classic_ft_2"
logger, session_timestamp = setup_logging(BASE_OUTPUT_DIR)

try:
    logger.info("Starting imports...")

    from huggingface_hub import hf_hub_download, login
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from jinja2 import TemplateError
    from collections import Counter
    import numpy as np
    import transformers

    logger.info("All imports successful")

    # =========================
    # Config
    # =========================
    CONFIG = {
        "model_name": "google/gemma-2-2b-it",
        "dataset_repo": "miladmim/slim-orca-dedup-chat-50k-persian",
        "max_length": 1024, # We changed from 1024

        # Classic FT (no PEFT): we'll freeze all but first 2 + last 2 layers + lm_head
        "unfreeze_first_n": 2,
        "unfreeze_last_n": 2,

        "num_epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,      # lower LR recommended for partial unfreeze
        "warmup_ratio": 0.03,
        "eval_steps": 500,
        "save_steps": 500,
        "session_timestamp": session_timestamp
    }

    # Save config
    config_file = os.path.join(BASE_OUTPUT_DIR, f"config_{session_timestamp}.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    logger.info(f"Configuration saved to: {config_file}")

    # =========================
    # HF login (optional if you need gated model/dataset)
    # =========================
    logger.info("Logging into HuggingFace Hub...")
    # Replace token if needed, or comment out if already logged in via CLI.
    login("")
    logger.info("HuggingFace login successful")

    # =========================
    # Dataset Load (stream JSONL)
    # =========================
    logger.info("=== PHASE 1: Dataset Loading ===")
    logger.info(f"Downloading dataset: {CONFIG['dataset_repo']}")
    path = hf_hub_download(repo_id=CONFIG["dataset_repo"], filename="data.jsonl", repo_type="dataset")
    logger.info(f"Dataset downloaded to: {path}")

    def rows():
        row_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, list):
                        yield {"conversations": obj}
                    elif isinstance(obj, dict):
                        yield obj
                    else:
                        yield {"text": str(obj)}
                    row_count += 1
                    if row_count % 10000 == 0:
                        logger.info(f"Processed {row_count} rows...")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue

    logger.info("Building dataset from generator...")
    ds = Dataset.from_generator(rows, cache_dir="./my_dataset_cache")
    logger.info(f"Dataset loaded: {len(ds)} samples")

    # =========================
    # Analysis (roles)
    # =========================
    logger.info("=== PHASE 2: Dataset Analysis ===")
    all_keys = set()
    sub_keys = set()
    sample_size = min(1000, len(ds))
    for i in range(sample_size):
        row = ds[i]
        for k in row.keys():
            all_keys.add(k)
            if k == "conversations" and isinstance(row[k], list):
                for turn in row[k]:
                    if isinstance(turn, dict):
                        sub_keys.update(turn.keys())
    logger.info(f"Top-level keys: {all_keys}")
    logger.info(f"Sub-keys in conversations: {sub_keys}")

    roles = set()
    role_counter = Counter()
    for i in range(len(ds)):
        if "conversations" in ds[i] and isinstance(ds[i]["conversations"], list):
            for turn in ds[i]["conversations"]:
                if isinstance(turn, dict) and "role" in turn:
                    role = turn["role"]
                    roles.add(role)
                    role_counter[role] += 1
    logger.info(f"Unique roles found: {roles}")
    logger.info("Role distribution:")
    for role, count in role_counter.most_common():
        logger.info(f"  {role}: {count}")

    # =========================
    # Cleaning
    # =========================
    logger.info("=== PHASE 3: Data Cleaning ===")
    def clean_roles(row):
        if "conversations" not in row or not isinstance(row["conversations"], list):
            return {"conversations": []}
        new_conv = []
        for turn in row["conversations"]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role", "").strip().lower()
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role in ["user"]:
                new_conv.append({"role": "user", "content": content})
            elif role in ["assistant", "gpt", "دستیار"]:
                new_conv.append({"role": "assistant", "content": content})
            elif role in ["system"]:
                new_conv.append({"role": "system", "content": content})
        return {"conversations": new_conv}

    logger.info("Cleaning roles...")
    ds_clean = ds.map(clean_roles)
    initial_count = len(ds_clean)
    ds_clean = ds_clean.filter(lambda x: len(x["conversations"]) > 0)
    logger.info(f"Filtered dataset: {len(ds_clean)} samples (removed {initial_count - len(ds_clean)} empty conversations)")

    def has_valid_pair(row):
        roles_in_conv = [(t.get("role") or "").strip().lower() for t in row["conversations"]]
        for i in range(len(roles_in_conv) - 1):
            if roles_in_conv[i] == "user" and roles_in_conv[i + 1] == "assistant":
                return True
        return False

    valid_pairs_count = len(ds_clean)
    ds_clean = ds_clean.filter(has_valid_pair)
    logger.info(f"Valid conversation pairs: {len(ds_clean)} samples (removed {valid_pairs_count - len(ds_clean)} without user->assistant pairs)")

    # =========================
    # Extract first user->assistant pair (+ optional system)
    # =========================
    logger.info("=== PHASE 4: Pair Extraction ===")
    def extract_first_pair(row):
        system = None
        prompt = None
        response = None
        for turn in row["conversations"]:
            role = (turn.get("role") or "").strip().lower()
            if role == "system":
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

    pairs = ds_clean.map(extract_first_pair, remove_columns=[c for c in ds_clean.column_names if c != "conversations"])

    def is_valid_pair(ex):
        return (isinstance(ex["prompt"], str) and ex["prompt"].strip()
                and isinstance(ex["response"], str) and ex["response"].strip())

    before_filter = len(pairs)
    pairs = pairs.filter(is_valid_pair)
    logger.info(f"Final valid pairs: {len(pairs)} (removed {before_filter - len(pairs)} invalid pairs)")

    dataset_stats = {
        "original_size": len(ds),
        "after_cleaning": len(ds_clean),
        "final_pairs": len(pairs),
        "roles_found": list(roles),
        "role_distribution": dict(role_counter)
    }
    stats_file = os.path.join(BASE_OUTPUT_DIR, f"dataset_stats_{session_timestamp}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Dataset statistics saved to: {stats_file}")

    # =========================
    # Model & Tokenizer
    # =========================
    logger.info("=== PHASE 5: Model Setup ===")
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires GPU.")
        raise RuntimeError("CUDA not available")
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

    logger.info(f"Loading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")

    logger.info(f"Loading base model: {CONFIG['model_name']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )

    # Small architecture summary (EXTRACTION)
    logger.info("=== MODEL ARCHITECTURE SUMMARY ===")
    # Gemma2ForCausalLM(model=..., lm_head=...)
    # Transformer blocks live at base_model.model.layers (list-like)
    # Each block typically has self_attn, mlp, layernorms, etc.
    n_layers = None
    try:
        n_layers = len(base_model.model.layers)
        logger.info(f"Backbone module: base_model.model (type: {type(base_model.model).__name__})")
        logger.info(f"Number of transformer layers: {n_layers}")
        logger.info(f"Example submodules in a block: {list(dict(base_model.model.layers[0].named_children()).keys())}")
    except Exception as _:
        logger.warning("Could not introspect layers via base_model.model.layers")

    # Freeze then unfreeze (FIRST 2 + LAST 2 + LM HEAD)
    for p in base_model.parameters():
        p.requires_grad = False

    if n_layers is None:
        # Fallback if path differs (print to guide)
        logger.error("Could not locate base_model.model.layers — architecture layout may differ.")
        raise RuntimeError("Layer path not found. Inspect `print(base_model)` to adjust unfreeze path.")

    first_n = CONFIG["unfreeze_first_n"]
    last_n = CONFIG["unfreeze_last_n"]
    total = n_layers

    def set_layer_trainable(layer, flag=True):
        for p in layer.parameters():
            p.requires_grad = flag

    # Unfreeze first N
    for i in range(min(first_n, total)):
        set_layer_trainable(base_model.model.layers[i], True)

    # Unfreeze last N
    for i in range(max(0, total - last_n), total):
        set_layer_trainable(base_model.model.layers[i], True)

    # Always unfreeze lm_head for CausalLM adaptation
    if hasattr(base_model, "lm_head"):
        for p in base_model.lm_head.parameters():
            p.requires_grad = True
    else:
        logger.warning("lm_head not found; logits projection may remain frozen.")

    # Enable gradient checkpointing and ensure inputs require grad (to avoid warnings)
    base_model.gradient_checkpointing_enable()
    try:
        base_model.enable_input_require_grads()
    except Exception:
        # fallback hook for models without the helper
        def make_inputs_require_grad(module, input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
        if hasattr(base_model, "get_input_embeddings"):
            emb = base_model.get_input_embeddings()
            emb.register_forward_hook(make_inputs_require_grad)

    # Sanity check: trainable params > 0
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100*trainable_params/total_params:.4f}%)")

    if trainable_params == 0:
        raise RuntimeError("No parameters are trainable after unfreezing; check layer paths.")

    # =========================
    # Tokenization
    # =========================
    logger.info("=== PHASE 6: Tokenization ===")

    @dataclass
    class DataCollatorForSFT:
        def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
            def pad_to_max(key):
                seqs = [f[key] for f in features]
                maxlen = max(len(s) for s in seqs)
                pad_id = tokenizer.pad_token_id
                arr = np.full((len(seqs), maxlen), pad_id if key != "labels" else -100, dtype=np.int64)
                for i, s in enumerate(seqs):
                    L = min(len(s), maxlen)
                    arr[i, :L] = s[:L]
                return torch.tensor(arr)
            input_ids = pad_to_max("input_ids")
            batch = {
                "input_ids": input_ids,
                "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
                "labels": pad_to_max("labels"),
            }
            return batch

    def tokenize_with_chat_template(batch):
        input_ids, attention_masks, labels = [], [], []
        systems = batch.get("system", [None] * len(batch["prompt"]))
        successful_tokenizations = 0
        for sys_txt, user_txt, resp_txt in zip(systems, batch["prompt"], batch["response"]):
            if not isinstance(user_txt, str) or not isinstance(resp_txt, str):
                continue
            user_clean = user_txt.strip()
            sys_clean = (sys_txt or "").strip()
            messages = []
            if sys_clean:
                messages.append({"role": "system", "content": sys_clean})
            messages.append({"role": "user", "content": user_clean})
            try:
                prompt_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )
            except TemplateError:
                merged_user = (sys_clean + "\n\n" if sys_clean else "") + user_clean
                messages = [{"role": "user", "content": merged_user}]
                prompt_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )
            ans_text = resp_txt.strip()
            if tokenizer.eos_token:
                ans_text += tokenizer.eos_token
            answer_ids = tokenizer.encode(ans_text, add_special_tokens=False)

            budget = CONFIG["max_length"] - len(answer_ids)
            if budget < 0:
                answer_ids = answer_ids[-CONFIG["max_length"]:]
                prompt_trunc = []
            else:
                prompt_trunc = prompt_ids[:budget]

            ids = prompt_trunc + answer_ids
            att = [1] * len(ids)
            lab = [-100] * len(prompt_trunc) + answer_ids[:]

            if len(ids) > CONFIG["max_length"]:
                ids = ids[:CONFIG["max_length"]]
                att = att[:CONFIG["max_length"]]
                lab = lab[:CONFIG["max_length"]]

            input_ids.append(ids)
            attention_masks.append(att)
            labels.append(lab)
            successful_tokenizations += 1

        if successful_tokenizations % 1000 == 0:
            logger.info(f"Tokenized {successful_tokenizations} samples...")
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    logger.info("Starting tokenization...")
    start_time = time.time()
    tokenized = pairs.map(
        tokenize_with_chat_template,
        batched=True,
        remove_columns=pairs.column_names,
        desc="Tokenizing with chat template",
    )
    tokenization_time = time.time() - start_time
    logger.info(f"Tokenization completed in {tokenization_time:.2f} seconds")

    # =========================
    # Train / Val / Test split
    # =========================
    logger.info("=== PHASE 7: Dataset Split (train/val/test) ===")
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_val = split['train'].train_test_split(test_size=0.1111, seed=42)
    datasets = {
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': split['test'],
    }
    logger.info(f"Dataset split - Train: {len(datasets['train'])}, Validation: {len(datasets['validation'])}, Test: {len(datasets['test'])}")

    # =========================
    # Training setup
    # =========================
    logger.info("=== PHASE 8: Training Setup ===")
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"gemma2b_it_classic_ft_first{CONFIG['unfreeze_first_n']}_last{CONFIG['unfreeze_last_n']}_{session_timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     num_train_epochs=CONFIG["num_epochs"],
    #     per_device_train_batch_size=CONFIG["batch_size"],
    #     per_device_eval_batch_size=CONFIG["batch_size"],
    #     gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    #     learning_rate=CONFIG["learning_rate"],
    #     lr_scheduler_type="cosine",
    #     warmup_ratio=CONFIG["warmup_ratio"],
    #     weight_decay=0.001,
    #     logging_steps=50,
    #     max_grad_norm=1.0,
    #     eval_strategy="steps",
    #     eval_steps=CONFIG["eval_steps"],
    #     save_strategy="steps",
    #     save_steps=CONFIG["save_steps"],
    #     save_total_limit=2,
    #     optim="adamw_torch",
    #     fp16=False,
    #     gradient_checkpointing=True,
    #     report_to="none",
    #     logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    # )
    args = TrainingArguments(
        output_dir= OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",           # <-- disable saving/checkpointing
        logging_steps=100,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=False,
        group_by_length=True,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        report_to="none",
        eval_strategy="steps",    # Evaluate at the same frequency as logging
        eval_steps=500,                  # Run evaluation every 50 steps
    )
                # Run evaluation every 50 steps

    collator = DataCollatorForSFT()

    # Callback to capture loss history
    class LossHistoryCallback(transformers.TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            if logs is not None:
                if "loss" in logs and step > 0:
                    self.train_losses.append(logs["loss"])
                    self.steps.append(step)
                if "eval_loss" in logs and step > 0:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(step)

    loss_callback = LossHistoryCallback()

    trainer = Trainer(
        model=base_model,                 # classic FT (partially unfrozen)
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        callbacks=[loss_callback],
    )

    effective_batch_size = CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"]
    total_steps = len(datasets["train"]) * CONFIG["num_epochs"] // max(1, effective_batch_size)
    logger.info("Training configuration:")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Total training steps: {total_steps}")
    logger.info(f"  Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")

    # =========================
    # Training
    # =========================
    logger.info("=== PHASE 9: Training ===")
    logger.info("Starting training...")
    train_start = time.time()
    train_metrics = trainer.train()
    train_duration = time.time() - train_start
    logger.info(f"Training completed in {train_duration:.2f} seconds ({train_duration/60:.1f} minutes)")
    logger.info(f"Final training loss: {train_metrics.training_loss:.4f}")

    # =========================
    # Plot loss curves
    # =========================
    loss_curve_file = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.figure(figsize=(10, 6))
    if loss_callback.steps:
        plt.plot(loss_callback.steps, loss_callback.train_losses, label="Train Loss")
    if loss_callback.eval_steps:
        plt.plot(loss_callback.eval_steps, loss_callback.eval_losses, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve (Classic FT: first 2 + last 2 + lm_head)")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_curve_file, dpi=140, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curve saved to: {loss_curve_file}")

    # =========================
    # Final evaluation on TEST
    # =========================
    logger.info("=== PHASE 10: Final Evaluation (TEST) ===")
    test_metrics = trainer.evaluate(datasets["test"])
    if "eval_loss" in test_metrics:
        test_ppl = math.exp(test_metrics["eval_loss"]) if test_metrics["eval_loss"] < 20 else float("inf")
        logger.info(f"Test evaluation loss: {test_metrics['eval_loss']:.4f}")
        logger.info(f"Test perplexity: {test_ppl:.2f}")
    else:
        test_ppl = None

    # =========================
    # Save artifacts
    # =========================
    logger.info("=== PHASE 11: Saving Models and Artifacts ===")
    # Save full model (partially fine-tuned)
    model_dir = os.path.join(OUTPUT_DIR, "classic_ft_model")
    base_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to: {model_dir}")
    logger.info(f"Tokenizer saved to: {OUTPUT_DIR}")

    training_summary = {
        "session_timestamp": session_timestamp,
        "config": CONFIG,
        "dataset_stats": dataset_stats,
        "model_info": {
            "n_layers": n_layers,
            "unfrozen": {
                "first_n": CONFIG["unfreeze_first_n"],
                "last_n": CONFIG["unfreeze_last_n"],
                "lm_head": True
            },
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_params / total_params
        },
        "training_metrics": {
            "final_train_loss": float(train_metrics.training_loss),
            "training_duration_seconds": float(train_duration),
            "val_last_eval_loss": float(loss_callback.eval_losses[-1]) if loss_callback.eval_losses else None,
            "test_eval_loss": float(test_metrics.get("eval_loss")) if test_metrics.get("eval_loss") is not None else None,
            "test_perplexity": float(test_ppl) if test_ppl is not None else None
        },
        "paths": {
            "output_dir": OUTPUT_DIR,
            "config_file": config_file,
            "stats_file": stats_file,
            "loss_curve": loss_curve_file,
            "model_dir": model_dir
        }
    }
    summary_file = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Training summary saved to: {summary_file}")

except Exception as e:
    logger.error(f"FAILED: {e}")
    raise