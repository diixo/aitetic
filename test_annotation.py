# t5_json_annotator.py
# One-task T5: sentence -> full JSON annotation (with events inside)
#
# Install:
#   pip install -U transformers datasets accelerate sentencepiece evaluate
#
# Train:
#   python t5_json_annotator.py train --train data_train.jsonl --valid data_valid.jsonl --out ./t5-annotator
#
# Predict:
#   python t5_json_annotator.py predict --ckpt ./t5-annotator --text "He was accused of stealing money."
#
# Predict (file):
#   python t5_json_annotator.py predict --ckpt ./t5-annotator --in sentences.txt --out preds.jsonl

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# -----------------------------
# 1) Schema + prompt
# -----------------------------

ALLOWED_TENSE = {"past", "present", "future", "unknown"}
ALLOWED_ASPECT = {"simple", "progressive", "perfect", "perfect_progressive", "unknown"}
ALLOWED_TIME_REL = {"past", "present", "future", "unknown"}
ALLOWED_MODALITY = {
    "none", "will", "going_to", "can", "could", "may", "might", "must", "should",
    "would", "want_to", "need_to", "unknown"
}

SCHEMA_HINT = """
Return ONLY valid JSON with this structure:

{
  "example": "<verbatim input sentence>",
  "annotation": [
    {
      "event_text": "<verbatim span from example>",
      "action_text": "<verbatim span from event_text or example>",
      "action": "<lemma/base form, may differ from surface>",
      "tense": "past|present|future|unknown",
      "tense_aspect": "simple|progressive|perfect|perfect_progressive|unknown",
      "time_relation": "past|present|future|unknown",
      "future_marker": "<verbatim if present like 'will'/'is going to', else null>",
      "subject": "<verbatim span or null>",
      "object": "<verbatim span or null>",
      "prep": "<verbatim preposition or null>",
      "prep_object": "<verbatim span or null>",
      "time_expr": "<verbatim span or null>",
      "negation": true|false,
      "modality": "none|will|going_to|can|could|may|might|must|should|would|want_to|need_to|unknown"
    }
  ]
}

Hard rules:
- Every field that is a span MUST appear verbatim in "example" (exact substring), otherwise use null.
- Use only allowed label values.
- Output ONLY JSON. No markdown, no commentary.
""".strip()


def build_prompt(sentence: str) -> str:
    # "one task": annotate full JSON
    return f"task: annotate_json\nexample: {sentence}\n{SCHEMA_HINT}"


# -----------------------------
# 2) Utilities: JSON parsing + validation + repair prompt
# -----------------------------

def try_load_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust-ish: finds first {...} block if the model adds stray tokens.
    """
    text = text.strip()
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract the biggest JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def _span_ok(span: Optional[str], example: str) -> bool:
    if span is None:
        return True
    if not isinstance(span, str):
        return False
    if span == "":
        return False
    return span in example


def validate_annotation(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["root is not a dict"]

    ex = obj.get("example")
    ann = obj.get("annotation")

    if not isinstance(ex, str) or not ex:
        errors.append("example missing or not a string")
        ex = ""  # keep going

    if not isinstance(ann, list) or len(ann) == 0:
        errors.append("annotation missing or empty list")
        return False, errors

    for i, ev in enumerate(ann):
        if not isinstance(ev, dict):
            errors.append(f"annotation[{i}] is not a dict")
            continue

        # Label checks
        t = ev.get("tense", "unknown")
        a = ev.get("tense_aspect", "unknown")
        tr = ev.get("time_relation", "unknown")
        mod = ev.get("modality", "unknown")

        if t not in ALLOWED_TENSE:
            errors.append(f"annotation[{i}].tense invalid: {t}")
        if a not in ALLOWED_ASPECT:
            errors.append(f"annotation[{i}].tense_aspect invalid: {a}")
        if tr not in ALLOWED_TIME_REL:
            errors.append(f"annotation[{i}].time_relation invalid: {tr}")
        if mod not in ALLOWED_MODALITY:
            errors.append(f"annotation[{i}].modality invalid: {mod}")

        # Span checks: must be verbatim substrings or null
        for k in ["event_text", "action_text", "future_marker", "subject", "object",
                  "prep", "prep_object", "time_expr"]:
            v = ev.get(k, None)
            if v is not None and not isinstance(v, str):
                errors.append(f"annotation[{i}].{k} not string/null")
                continue
            if not _span_ok(v, ex):
                errors.append(f"annotation[{i}].{k} not verbatim substring")

        # Negation boolean
        neg = ev.get("negation", False)
        if not isinstance(neg, bool):
            errors.append(f"annotation[{i}].negation not boolean")

        # action lemma should be string (can be non-span)
        act = ev.get("action", "")
        if not isinstance(act, str) or not act:
            errors.append(f"annotation[{i}].action missing or not string")

    return len(errors) == 0, errors


def build_repair_prompt(sentence: str, bad_output: str, errors: List[str]) -> str:
    # still "one task" but with repair instruction
    err_txt = "; ".join(errors[:12])
    return (
        "task: repair_annotate_json\n"
        f"example: {sentence}\n"
        "rule: fix the JSON so it matches schema and hard rules.\n"
        f"validation_errors: {err_txt}\n"
        f"bad_output: {bad_output}\n"
        f"{SCHEMA_HINT}"
    )


# -----------------------------
# 3) Data IO (jsonl)
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def to_hf_dataset(items: List[Dict[str, Any]]) -> Dataset:
    """
    Expects your training jsonl lines like:
      {"example":"...", "annotation":[...]}   # as you already use
    We train T5 to output the full JSON object (same structure).
    """
    rows = []
    for obj in items:
        ex = obj.get("example")
        ann = obj.get("annotation")
        if not isinstance(ex, str) or not ex:
            continue
        if not isinstance(ann, list) or len(ann) == 0:
            continue

        prompt = build_prompt(ex)
        target = json.dumps({"example": ex, "annotation": ann}, ensure_ascii=False)

        rows.append({"input_text": prompt, "target_text": target})
    return Dataset.from_list(rows)


# -----------------------------
# 4) Tokenization
# -----------------------------

@dataclass
class TokenizeCfg:
    max_input_len: int = 512
    max_target_len: int = 512


def tokenize_batch(tokenizer, batch, cfg: TokenizeCfg):
    model_in = tokenizer(
        batch["input_text"],
        max_length=cfg.max_input_len,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=cfg.max_target_len,
            truncation=True,
            padding=False,
        )
    model_in["labels"] = labels["input_ids"]
    return model_in


# -----------------------------
# 5) Train
# -----------------------------

def cmd_train(args):
    train_items = read_jsonl(args.train)
    valid_items = read_jsonl(args.valid) if args.valid else []

    train_ds = to_hf_dataset(train_items)
    valid_ds = to_hf_dataset(valid_items) if valid_items else None

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tcfg = TokenizeCfg(max_input_len=args.max_in, max_target_len=args.max_out)
    train_tok = train_ds.map(lambda b: tokenize_batch(tokenizer, b, tcfg), batched=True)
    valid_tok = valid_ds.map(lambda b: tokenize_batch(tokenizer, b, tcfg), batched=True) if valid_ds else None

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        predict_with_generate=False,
        logging_steps=50,
        save_steps=500,
        eval_steps=500 if valid_tok is not None else None,
        evaluation_strategy="steps" if valid_tok is not None else "no",
        save_total_limit=2,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved to: {args.out}")


# -----------------------------
# 6) Predict + (optional) repair pass
# -----------------------------

@torch.inference_mode()
def generate_json(
    model,
    tokenizer,
    sentence: str,
    max_new_tokens: int = 512,
    do_repair: bool = True,
) -> Dict[str, Any]:
    prompt = build_prompt(sentence)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=0.9,
        early_stopping=True,
    )
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    obj = try_load_json(raw)

    if obj is not None:
        ok, errs = validate_annotation(obj)
        if ok:
            return obj
        if do_repair:
            rep_prompt = build_repair_prompt(sentence, raw, errs)
            rep_in = tokenizer(rep_prompt, return_tensors="pt", truncation=True, max_length=512)
            rep_in = {k: v.to(model.device) for k, v in rep_in.items()}
            rep_ids = model.generate(
                **rep_in,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                length_penalty=0.9,
                early_stopping=True,
            )
            rep_raw = tokenizer.decode(rep_ids[0], skip_special_tokens=True)
            rep_obj = try_load_json(rep_raw)
            if rep_obj is not None:
                ok2, _ = validate_annotation(rep_obj)
                if ok2:
                    return rep_obj

    # Fallback: minimal valid structure (keeps you running)
    return {
        "example": sentence,
        "annotation": [{
            "event_text": sentence if sentence else "",
            "action_text": None,
            "action": "unknown",
            "tense": "unknown",
            "tense_aspect": "unknown",
            "time_relation": "unknown",
            "future_marker": None,
            "subject": None,
            "object": None,
            "prep": None,
            "prep_object": None,
            "time_expr": None,
            "negation": False,
            "modality": "unknown"
        }]
    }


def cmd_predict(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if args.text:
        obj = generate_json(model, tokenizer, args.text, max_new_tokens=args.max_new, do_repair=not args.no_repair)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return

    if args.inp:
        in_path = Path(args.inp)
        lines = [x.rstrip("\n") for x in in_path.read_text(encoding="utf-8").splitlines()]
        out_rows = []
        for s in lines:
            s = s.strip()
            if not s:
                continue
            out_rows.append(generate_json(model, tokenizer, s, max_new_tokens=args.max_new, do_repair=not args.no_repair))

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Wrote: {args.out}")
        else:
            for row in out_rows[:5]:
                print(json.dumps(row, ensure_ascii=False))
        return

    raise SystemExit("Provide --text or --in")


# -----------------------------
# 7) CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True, help="train jsonl with {example, annotation}")
    t.add_argument("--valid", default=None, help="valid jsonl")
    t.add_argument("--out", required=True, help="output dir")
    t.add_argument("--model", default="t5-small", help="base model checkpoint")
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--bs", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--max_in", type=int, default=512)
    t.add_argument("--max_out", type=int, default=512)
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--bf16", action="store_true")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--ckpt", required=True, help="trained checkpoint dir")
    pr.add_argument("--text", default=None, help="single sentence")
    pr.add_argument("--in", dest="inp", default=None, help="text file (1 sentence per line)")
    pr.add_argument("--out", default=None, help="output jsonl for predictions")
    pr.add_argument("--max_new", type=int, default=512)
    pr.add_argument("--no_repair", action="store_true", help="disable repair pass")
    pr.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
