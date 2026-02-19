# t5_json_annotator_min.py
# pip install -U transformers sentencepiece torch

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


ALLOWED_TENSE = {"past", "present", "future", "unknown"}
ALLOWED_ASPECT = {"simple", "progressive", "perfect", "perfect_progressive", "unknown"}
ALLOWED_TIME_REL = {"past", "present", "future", "unknown"}

SPAN_FIELDS = {
    "action_text",
    "future_marker",
    "subject",
    "object",
    "prep",
    "prep_object",
    "time_expr",
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_quotes(x: str) -> str:
    x = x.strip()
    if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
        return x[1:-1].strip()
    return x


def _first_line(x: str) -> str:
    return x.splitlines()[0].strip()


def _split_lines_to_list(x: str) -> List[str]:
    lines = [l.strip() for l in x.splitlines() if l.strip()]
    out = []
    for l in lines:
        l = re.sub(r"^[-*]\s*", "", l)
        l = re.sub(r"^\d+[\)\.]\s*", "", l)
        if l:
            out.append(l)
    return out


def _unique_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _as_null_if_not_substring(span: Optional[str], example: str) -> Optional[str]:
    if span is None:
        return None
    span = _clean(span)
    if not span:
        return None
    return span if span in example else None


def _normalize_label(value: str, allowed: set) -> str:
    v = _clean(value).lower().replace("-", "_")
    return v if v in allowed else "unknown"


@dataclass
class T5QA:
    model_name: str = "google/flan-t5-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_input_len: int = 512
    max_new_tokens: int = 64
    num_beams: int = 4

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()

    @torch.inference_mode()
    def gen(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        inp = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_len
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out_ids = self.model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


# ---- 1) split sentence into events (optional but useful) ----

def split_events(qa: T5QA, sentence: str) -> List[str]:
    sentence = sentence.strip()
    if not sentence:
        return []

    prompt = (
        "Split the sentence into minimal event clauses.\n"
        "Rules:\n"
        "- Each item MUST be an exact substring from the sentence.\n"
        "- Do not paraphrase.\n"
        "- Output one item per line.\n\n"
        f"Sentence: {sentence}"
    )
    raw = qa.gen(prompt, max_new_tokens=128)
    candidates = _unique_preserve_order([_strip_quotes(_clean(x)) for x in _split_lines_to_list(raw)])
    events = [c for c in candidates if c and c in sentence]

    return events or [sentence]


# ---- 2) ask per-field questions ----

import string
import re
from typing import Optional

_PUNCT_STRIP = string.punctuation + "“”„«»"

def _strip_outer_punct(s: str) -> str:
    return s.strip().strip(_PUNCT_STRIP).strip()

def _find_span_in_text(pred: Optional[str], text: str) -> Optional[str]:
    """
    Try to map model output `pred` to an EXACT substring from `text`.
    - exact match
    - stripped punctuation match
    - case-insensitive regex match (returns exact slice from text)
    """
    if pred is None:
        return None
    p = _clean(_strip_quotes(pred))
    if not p:
        return None

    # 1) exact
    if p in text:
        return p

    # 2) strip outer punctuation
    p2 = _strip_outer_punct(p)
    if p2 and p2 in text:
        return p2

    # 3) case-insensitive find in text, return exact slice
    # try with p2 first, then p
    for q in [p2, p]:
        if not q:
            continue
        # collapse spaces for matching
        q_norm = re.sub(r"\s+", r"\\s+", re.escape(q))
        m = re.search(q_norm, text, flags=re.IGNORECASE)
        if m:
            return text[m.start():m.end()]

    return None


def ask_span_field(qa: T5QA, example: str, event_text: str, question: str) -> Optional[str]:
    prompt = (
        "Answer the question by COPYING an EXACT span from the EVENT text.\n"
        "If there is no answer in the event text, output: null\n\n"
        f"Event text: {event_text}\n"
        f"Question: {question}\n"
        "Return ONLY the copied span or null."
    )
    raw = _first_line(qa.gen(prompt))
    raw = _strip_quotes(raw)
    if raw.lower().strip() == "null":
        return None

    # prefer matching inside event, then ensure it maps to sentence
    span = _find_span_in_text(raw, event_text)
    if span is None:
        return None
    # map to example (usually exact, but just in case punctuation differs)
    span2 = _find_span_in_text(span, example)
    return span2



def ask_label_field(qa: T5QA, example: str, event_text: str, question: str, allowed: set) -> str:
    prompt = (
        f"Choose ONE label from: {sorted(list(allowed))}\n"
        "If unsure, output: unknown\n\n"
        f"Sentence: {example}\n"
        f"Event: {event_text}\n"
        f"Question: {question}\n"
        "Return ONLY the label."
    )
    raw = _first_line(qa.gen(prompt))
    return _normalize_label(raw, allowed)


def ask_text_field(qa: T5QA, example: str, event_text: str, question: str) -> Optional[str]:
    """
    For 'action' (base form). Not required to be substring; but keep short.
    If unknown -> null
    """
    prompt = (
        "Answer briefly.\n"
        "If unknown, output: null\n\n"
        f"Sentence: {example}\n"
        f"Event: {event_text}\n"
        f"Question: {question}\n"
        "Return ONLY the answer or null."
    )
    raw = _strip_quotes(_clean(_first_line(qa.gen(prompt))))
    if raw.lower() == "null" or raw == "":
        return None
    return raw


# ---- 3) annotate one event in YOUR schema ----

def annotate_event(qa: T5QA, example: str, event_text: str) -> Dict[str, Any]:
    ann: Dict[str, Any] = {}

    # spans (must be substrings -> else null)
    ann["action_text"] = ask_span_field(
        qa, example, event_text, "What is the main action phrase (verb phrase) in this event?"
    )
    ann["future_marker"] = ask_span_field(
        qa, example, event_text, "If future is expressed, what exact marker is used (e.g., 'will', 'is going to')?"
    )
    ann["subject"] = ask_span_field(
        qa, example, event_text, "Who/what performs the action (subject) in this event?"
    )
    ann["object"] = ask_span_field(
        qa, example, event_text, "What is acted on (direct object) in this event?"
    )
    ann["prep"] = ask_span_field(
        qa, example, event_text, "If there is an important preposition linked to the action, what is it?"
    )
    ann["prep_object"] = ask_span_field(
        qa, example, event_text, "If there is a prepositional object (object of the preposition), what is it?"
    )
    ann["time_expr"] = ask_span_field(
        qa, example, event_text, "What time expression is mentioned (e.g., 'next year', 'yesterday')?"
    )

    # labels
    ann["tense"] = ask_label_field(
        qa, example, event_text, "What is the tense of the main action in this event?", ALLOWED_TENSE
    )
    ann["tense_aspect"] = ask_label_field(
        qa, example, event_text, "What is the tense-aspect of the main action?", ALLOWED_ASPECT
    )
    ann["time_relation"] = ask_label_field(
        qa, example, event_text, "Relative to now, is the event in the past, present, or future?", ALLOWED_TIME_REL
    )

    # text (lemma/base form)
    ann["action"] = ask_text_field(
        qa, example, event_text,
        "Give the base/lemma form of the main action (e.g., 'was accused' -> 'accuse'; 'is going to close down' -> 'close down')."
    )

    # harden: guarantee substring-only for span fields
    for k in SPAN_FIELDS:
        ann[k] = _as_null_if_not_substring(ann.get(k), example)

    # harden labels
    if ann["tense"] not in ALLOWED_TENSE:
        ann["tense"] = "unknown"
    if ann["tense_aspect"] not in ALLOWED_ASPECT:
        ann["tense_aspect"] = "unknown"
    if ann["time_relation"] not in ALLOWED_TIME_REL:
        ann["time_relation"] = "unknown"

    return ann


def annotate_sentence(qa: T5QA, sentence: str) -> Dict[str, Any]:
    sentence = sentence.strip()
    events = split_events(qa, sentence)
    ann_list = [annotate_event(qa, sentence, ev) for ev in events]
    return {"example": sentence, "annotation": ann_list}


def annotate_many(sentences: List[str], out_path: str = "out.json") -> None:
    qa = T5QA()
    data = [annotate_sentence(qa, s) for s in sentences if s.strip()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    samples = [
        "The company is going to close down its factory next year.",
        "They fell in love and got married.",
    ]
    annotate_many(samples, out_path="out.json")
    print("Wrote out.json")
