import re
import json
import torch
import spacy
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 0) MODEL (T5)
# ============================================================
MODEL_NAME = "google/flan-t5-base"   # try "google/flan-t5-large" if you can
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def _t5_generate(prompt: str, max_new_tokens: int = 180) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ============================================================
# 1) spaCy (tokenize + lemma, no dependency requirement)
# ============================================================
def build_nlp():
    # try full pipeline if installed
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        pass
    # fallback minimal English
    nlp = English()
    nlp.add_pipe("attribute_ruler")
    nlp.add_pipe("lemmatizer", config={"mode": "rule"})
    try:
        nlp.initialize()
    except Exception:
        pass
    return nlp

nlp = build_nlp()


# ============================================================
# 2) helpers: JSON salvage + substring validation
# ============================================================
def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())

def _is_verbatim_substring(span: str, sentence: str) -> bool:
    if not span or not sentence:
        return False
    return _normalize_ws(span).lower() in _normalize_ws(sentence).lower()

def _safe_substring_or_none(span: str, sentence: str):
    if not isinstance(span, str):
        return None
    span = span.strip()
    return span if _is_verbatim_substring(span, sentence) else None

def safe_json_load(raw: str) -> dict:
    """
    Parse strict JSON. If model adds garbage, salvage first {...} block.
    """
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

def lemma_phrase(text: str) -> str:
    """
    Lemmatize surface phrase. If lemmatizer unavailable, fallback to lowercase tokens.
    """
    if not text:
        return ""
    doc = nlp(text)
    out = []
    for t in doc:
        if t.is_space:
            continue
        out.append((t.lemma_ or t.text).lower())
    return " ".join(out)

def tense_aspect_time_from_action(action_text: str) -> tuple[str, str, str]:
    """
    Minimal tense/aspect/time_relation just from the action_text.
    """
    if not action_text:
        return "unknown", "unknown", "unknown"

    d = nlp(action_text)
    tense = "present"
    aspect = "simple"

    # very rough: if any past tag
    if any(t.tag_ in ("VBD", "VBN") for t in d):
        tense = "past"
    if any(t.tag_ == "VBG" for t in d):
        aspect = "progressive"

    time_relation = "past" if tense == "past" else ("present" if tense == "present" else "unknown")
    return tense, aspect, time_relation

def infer_polarity(sentence: str, action_text: str | None) -> str:
    if not action_text or not sentence:
        return "positive"
    s = sentence.lower()
    idx = s.find(action_text.lower())
    if idx < 0:
        return "positive"
    win = s[max(0, idx - 40): min(len(s), idx + len(action_text) + 40)]
    return "negative" if re.search(r"\bnot\b|n't\b|\bnever\b", win) else "positive"

def infer_imperative(sentence: str, subject: str | None) -> bool:
    """
    Simple: negative imperative often starts with Don't / Do not
    """
    if subject:
        return False
    s = sentence.strip()
    return bool(re.match(r"^(Don['’]t|Do not)\b", s, flags=re.IGNORECASE))


# ============================================================
# 3) clause split (deterministic)
# ============================================================
def split_clauses(sentence: str) -> list[str]:
    """
    Cheap deterministic split:
      - split on ';'
      - optionally split on ' and ' only if both parts contain verb-ish token
    """
    parts = [p.strip() for p in sentence.split(";") if p.strip()]
    out = []
    for p in parts:
        if " and " in p:
            left, right = p.split(" and ", 1)
            dl = nlp(left)
            dr = nlp(right)
            has_v_l = any(t.pos_ in ("VERB", "AUX") for t in dl)
            has_v_r = any(t.pos_ in ("VERB", "AUX") for t in dr)
            if has_v_l and has_v_r:
                out.append(left.strip())
                out.append(("and " + right).strip())  # keep surface
                continue
        out.append(p)
    return out if out else [sentence.strip()]


# ============================================================
# 4) Candidate generation (no dependencies)
# ============================================================
def build_candidates_for_clause(clause: str) -> dict:
    """
    Build conservative candidate lists from tokens:
      actions: verb-centered surface spans
      subjects: NP-ish spans to the left of a verb
      objects: NP-ish spans to the right
      pps: (prep, prep_object) pairs to the right
    """
    doc = nlp(clause)

    verb_idxs = [t.i for t in doc if t.pos_ in ("VERB", "AUX")]
    # fallback: if nothing parsed, return empty
    if not verb_idxs:
        return {"actions": [], "subjects": [], "objects": [], "pps": []}

    actions = set()
    subjects = set()
    objects = set()
    pps = set()

    def is_stop_punct(t) -> bool:
        return t.is_punct and t.text in {",", ";", ":", "—", "–"}

    for v_i in verb_idxs:
        v = doc[v_i]

        # ---- action candidates: verb / verb+ADP/PART/ADV / verb+2 tokens / verb+VBN/ADJ complement
        actions.add(v.text)

        if v_i + 1 < len(doc):
            t1 = doc[v_i + 1]
            if not t1.is_punct and t1.pos_ in ("ADP", "PART", "ADV"):
                actions.add(f"{v.text} {t1.text}")

        if v_i + 2 < len(doc):
            t1, t2 = doc[v_i + 1], doc[v_i + 2]
            if not t1.is_punct and not t2.is_punct:
                actions.add(f"{v.text} {t1.text} {t2.text}")

        if v_i + 1 < len(doc):
            t1 = doc[v_i + 1]
            if (t1.tag_ == "VBN") or (t1.pos_ == "ADJ"):
                actions.add(f"{v.text} {t1.text}")

        # ---- subject candidates: suffix spans in window left of verb
        L = 7
        start = max(0, v_i - L)
        window = doc[start:v_i]
        for i in range(len(window)):
            span = window[i:]
            if not span:
                continue
            if any(t.pos_ in ("NOUN", "PROPN", "PRON") for t in span) and not any(is_stop_punct(t) for t in span):
                subjects.add(_normalize_ws(span.text))

        # ---- object candidates: spans right of verb until punct
        R = 10
        end = min(len(doc), v_i + 1 + R)
        right_tokens = []
        for t in doc[v_i + 1:end]:
            if is_stop_punct(t) or t.text in {".", "!", "?"}:
                break
            right_tokens.append(t)

        for j in range(1, len(right_tokens) + 1):
            span = doc[v_i + 1: v_i + 1 + j]
            if not span:
                continue
            if span[0].pos_ == "ADP":
                continue
            if any(t.pos_ in ("NOUN", "PROPN", "PRON") for t in span):
                objects.add(_normalize_ws(span.text))

        # ---- PP candidates: ADP + following NP-ish chunk
        for k, t in enumerate(right_tokens):
            if t.pos_ != "ADP":
                continue
            prep = t.text.lower()
            pobj_toks = []
            for u in right_tokens[k + 1: k + 1 + 7]:
                if u.pos_ == "ADP" or is_stop_punct(u) or u.is_punct:
                    break
                pobj_toks.append(u)
            if pobj_toks and any(x.pos_ in ("NOUN", "PROPN", "PRON") for x in pobj_toks):
                prep_object = _normalize_ws(" ".join(x.text for x in pobj_toks))
                pps.add((prep, prep_object))

    # cleanup ordering
    actions = [a for a in sorted(actions, key=lambda s: (-len(s.split()), s)) if a]
    subjects = [s for s in sorted(subjects, key=lambda s: (-len(s.split()), s)) if s]
    objects = [o for o in sorted(objects, key=lambda s: (-len(s.split()), s)) if o]

    pp_list = list(pps)
    pp_list.sort(key=lambda p: (-len(p[1].split()), p[0], p[1]))
    pps = [{"prep": p, "prep_object": po} for (p, po) in pp_list]

    return {"actions": actions, "subjects": subjects, "objects": objects, "pps": pps}


# ============================================================
# 5) LLM: extract_lists(sentence)
#    - LLM tries to return verbatim lists
#    - If it fails validation -> fallback to deterministic candidates
# ============================================================
def extract_lists(sentence: str) -> dict:
    """
    Returns:
      {
        "actions": [action_text...],
        "subjects": [...],
        "objects": [...],
        "pps": [{"prep":..,"prep_object":..}, ...]
      }
    All strings are validated as verbatim substrings of sentence.
    """
    prompt = (
        "Task: extract_lists\n"
        f"Sentence: {sentence}\n"
        "Return ONLY valid JSON.\n"
        "Schema:\n"
        "{\n"
        '  "actions": [string],\n'
        '  "subjects": [string],\n'
        '  "objects": [string],\n'
        '  "pps": [{"prep": string, "prep_object": string}]\n'
        "}\n"
        "Rules:\n"
        "- Every string MUST be copied verbatim from the Sentence.\n"
        "- actions should include phrasal parts if present (e.g. 'zoomed by', 'fell in love', 'got married').\n"
        "- If uncertain, use empty lists.\n"
    )

    raw = _t5_generate(prompt, max_new_tokens=220)
    obj = safe_json_load(raw)

    # Validate + clean
    actions = []
    for a in obj.get("actions", []) if isinstance(obj.get("actions"), list) else []:
        a2 = _safe_substring_or_none(a, sentence)
        if a2:
            actions.append(a2)

    subjects = []
    for s in obj.get("subjects", []) if isinstance(obj.get("subjects"), list) else []:
        s2 = _safe_substring_or_none(s, sentence)
        if s2:
            subjects.append(s2)

    objects = []
    for o in obj.get("objects", []) if isinstance(obj.get("objects"), list) else []:
        o2 = _safe_substring_or_none(o, sentence)
        if o2:
            objects.append(o2)

    pps = []
    for p in obj.get("pps", []) if isinstance(obj.get("pps"), list) else []:
        if not isinstance(p, dict):
            continue
        prep = p.get("prep")
        prep_object = p.get("prep_object")
        if isinstance(prep, str) and prep.strip() and isinstance(prep_object, str) and prep_object.strip():
            po2 = _safe_substring_or_none(prep_object, sentence)
            if po2:
                pps.append({"prep": prep.strip().lower(), "prep_object": po2})

    # If LLM failed badly -> fallback to candidates from the sentence itself
    # (this is the key to avoid "nothing annotated")
    if not actions:
        cand = build_candidates_for_clause(sentence)
        return cand

    # Otherwise merge with deterministic candidates to increase recall
    # (still safe: candidates are from the sentence)
    cand = build_candidates_for_clause(sentence)
    actions = list(dict.fromkeys(actions + cand["actions"]))
    subjects = list(dict.fromkeys(subjects + cand["subjects"]))
    objects = list(dict.fromkeys(objects + cand["objects"]))

    # merge pps with de-dup
    seen_pp = set()
    merged_pp = []
    for pp in (pps + cand["pps"]):
        key = (pp["prep"], pp["prep_object"])
        if key in seen_pp:
            continue
        seen_pp.add(key)
        merged_pp.append(pp)

    return {"actions": actions, "subjects": subjects, "objects": objects, "pps": merged_pp}


# ============================================================
# 6) LLM: link_roles(sentence, action_text, lists)
#    - model only returns indices (or -1)
# ============================================================
def link_roles(sentence: str, action_text: str, lists: dict) -> dict:
    """
    lists must have keys: subjects, objects, pps
    Returns:
      {"subject_idx": int, "object_idx": int, "pp_idx": int}
    Indices refer to lists; -1 means null.
    """
    subjects = lists.get("subjects", [])
    objects = lists.get("objects", [])
    pps = lists.get("pps", [])

    # bound size (prompt stability)
    subjects = subjects[:25]
    objects = objects[:25]
    pps = pps[:25]

    subj_opts = "\n".join([f"{i}: {s}" for i, s in enumerate(subjects)]) if subjects else "(none)"
    obj_opts  = "\n".join([f"{i}: {o}" for i, o in enumerate(objects)]) if objects else "(none)"
    pp_opts   = "\n".join([f"{i}: {p['prep']} :: {p['prep_object']}" for i, p in enumerate(pps)]) if pps else "(none)"

    prompt = (
        "Task: link_roles\n"
        f"Sentence: {sentence}\n"
        f"Action_text: {action_text}\n"
        "Pick best indices for this action.\n"
        "Return ONLY JSON exactly like this:\n"
        '{"subject_idx": -1, "object_idx": -1, "pp_idx": -1}\n'
        "Rules:\n"
        "- Indices must refer to the provided lists.\n"
        "- Use -1 if none.\n"
        "- Do NOT add any other keys.\n"
        "Subjects:\n"
        f"{subj_opts}\n"
        "Objects:\n"
        f"{obj_opts}\n"
        "PPs:\n"
        f"{pp_opts}\n"
    )

    raw = _t5_generate(prompt, max_new_tokens=80)

    def clamp_idx(v, n):
        if not isinstance(v, int):
            return -1
        if v == -1:
            return -1
        return v if 0 <= v < n else -1
    
    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    obj = _as_dict(safe_json_load(raw))

    sidx = clamp_idx(obj.get("subject_idx", -1), len(subjects))
    oidx = clamp_idx(obj.get("object_idx", -1), len(objects))
    pidx = clamp_idx(obj.get("pp_idx", -1), len(pps))

    sidx = clamp_idx(obj.get("subject_idx", -1), len(subjects))
    oidx = clamp_idx(obj.get("object_idx", -1), len(objects))
    pidx = clamp_idx(obj.get("pp_idx", -1), len(pps))

    return {"subject_idx": sidx, "object_idx": oidx, "pp_idx": pidx}


# ============================================================
# 7) annotate(sentence) -> JSON in your format
# ============================================================
def annotate(sentence: str) -> dict:
    """
    Output:
    {
      "example": sentence,
      "annotation": [
         {action_text, action, tense, tense_aspect, time_relation, subject, object, prep, prep_object, ...}
      ]
    }

    Uses deterministic clause split, but linking uses original sentence to keep indices stable.
    """
    clauses = split_clauses(sentence)
    all_events = []

    for clause in clauses:
        lists = extract_lists(clause)
        actions = lists.get("actions", [])

        # If no actions even after fallback -> nothing
        if not actions:
            continue

        # For each action, link roles (on the SAME clause)
        for action_text in actions[:6]:  # cap to avoid exploding on long sentences
            # must still be verbatim
            action_text = _safe_substring_or_none(action_text, clause)
            if not action_text:
                continue

            link = link_roles(clause, action_text, lists)

            subjects = lists.get("subjects", [])
            objects = lists.get("objects", [])
            pps = lists.get("pps", [])

            subject = subjects[link["subject_idx"]] if link["subject_idx"] != -1 else None
            obj = objects[link["object_idx"]] if link["object_idx"] != -1 else None

            prep = None
            prep_object = None
            if link["pp_idx"] != -1:
                prep = pps[link["pp_idx"]]["prep"]
                prep_object = pps[link["pp_idx"]]["prep_object"]

            # normalize: ensure still substring of ORIGINAL sentence (not only clause)
            subject = _safe_substring_or_none(subject, sentence) if subject else None
            obj = _safe_substring_or_none(obj, sentence) if obj else None
            prep_object = _safe_substring_or_none(prep_object, sentence) if prep_object else None
            if prep and not prep_object:
                prep = None

            action = lemma_phrase(action_text)
            tense, aspect, time_relation = tense_aspect_time_from_action(action_text)

            polarity = infer_polarity(sentence, action_text)
            mood = "imperative" if infer_imperative(clause, subject) else None
            speech_act = "directive" if mood == "imperative" else None
            if mood == "imperative":
                # implied subject
                if subject is None:
                    subject = "you"
                tense, aspect, time_relation = "present", "simple", "present"

            ev = {
                "action_text": action_text,
                "action": action,
                "tense": tense,
                "tense_aspect": aspect,
                "time_relation": time_relation,
                "subject": subject,
                "object": obj,
                "prep": prep,
                "prep_object": prep_object,
            }
            if mood:
                ev["mood"] = mood
            if speech_act:
                ev["speech_act"] = speech_act
            if polarity != "positive":
                ev["polarity"] = polarity

            all_events.append(ev)

    # de-dup events (same action_text + args)
    uniq = []
    seen = set()
    for e in all_events:
        key = (
            e.get("action_text"),
            e.get("subject"),
            e.get("object"),
            e.get("prep"),
            e.get("prep_object"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    return {"example": sentence, "annotation": uniq}


# ============================================================
# 8) Demo
# ============================================================
if __name__ == "__main__":
    tests = [
        "Cars zoomed by in a steady flow.",
        "The cars zoomed by in an endless rapid flow.",
        "They fell in love and got married.",
        "Don't expect good work from him; he is lazy and careless.",
        "He fell off the bicycle and hurt his leg.",
        "He was exposed to the cold for too long.",
    ]
    for s in tests:
        print("\n---", s)
        print(json.dumps(annotate(s), ensure_ascii=False, indent=2))
