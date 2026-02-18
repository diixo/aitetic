import json
import spacy
from spacy.matcher import Matcher

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"  # можно large, если есть память

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def t5_pick_action(sentence: str, candidates: list[str], event_hint: str = "") -> tuple[str|None, str|None]:
    """
    Returns (action_lemma, action_text) chosen from candidates.
    Output must be one of candidates for action_lemma.
    """
    cand_str = " | ".join(candidates)

    prompt = (
        "Task: pick_action\n"
        f"Sentence: {sentence}\n"
        f"{event_hint}"
        f"Candidates (lemma): {cand_str}\n"
        "Rules:\n"
        "1) Choose exactly one candidate that best matches the meaning.\n"
        "2) action must be exactly one of the candidates.\n"
        "3) action_text must be the exact words from the sentence.\n"
        "Return format: action=<...>; action_text=<...>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=4,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    m1 = re.search(r"action\s*=\s*(.+?);", text)
    m2 = re.search(r"action_text\s*=\s*(.+)$", text)
    action = m1.group(1).strip() if m1 else None
    action_text = m2.group(1).strip() if m2 else None

    # строгая валидация: action должен быть из candidates
    if action not in candidates:
        return None, None
    return action, action_text


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# --- MWE patterns ---
matcher.add("FALL_IN_LOVE", [[{"LEMMA": "fall"}, {"LOWER": "in"}, {"LOWER": "love"}]])
matcher.add("FALL_IN_LOVE_FLEX", [[
    {"LEMMA": "fall"},
    {"LOWER": "in"},
    {"OP": "*", "POS": {"IN": ["ADV", "ADJ"]}},
    {"LOWER": "love"}
]])
matcher.add("FALL_IN_WITH", [[{"LEMMA": "fall"}, {"LOWER": "in"}, {"LOWER": "with"}]])

PARTICLE_WORDS = {
    "apart", "up", "down", "off", "out", "away", "back", "over", "around",
    "through", "along", "aside", "together"
}

# Optional: allow some prep to be part of action if you don't have MWE for it
VERB_PREP_WHITELIST = {
    "fall": {"off", "over", "down", "out"},
}

TENSE_MAP = {"pres": "present", "past": "past", "fut": "future"}


def _extract_subject_span(doc, verb):
    for c in verb.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return doc[c.left_edge.i : c.right_edge.i + 1]
    return None


def _extract_object_span(doc, verb):
    for c in verb.children:
        if c.dep_ in ("dobj", "obj"):
            return doc[c.left_edge.i : c.right_edge.i + 1]
    return None


def _extract_prep_object(doc, verb, preferred_preps=None):
    preferred_preps = set(preferred_preps or [])
    candidates = []
    for c in verb.children:
        if c.dep_ == "prep":
            pobj = next((x for x in c.children if x.dep_ == "pobj"), None)
            if pobj is None:
                continue
            span = doc[pobj.left_edge.i : pobj.right_edge.i + 1]
            score = 1 + (10 if c.lower_ in preferred_preps else 0)
            candidates.append((score, c.lower_, span.text))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, prep, pobj_text = candidates[0]
    return prep, pobj_text


def _tense_aspect(verb):
    aux = [c for c in verb.children if c.dep_ in ("aux", "auxpass")]
    aux_lemmas = {a.lemma_.lower() for a in aux}

    # aspect
    if verb.tag_ == "VBG" and ("be" in aux_lemmas):
        aspect = "progressive"
    elif "have" in aux_lemmas:
        aspect = "perfect"
    else:
        aspect = "simple"

    # tense
    tense_raw = None
    for a in aux:
        t = a.morph.get("Tense")
        if t:
            tense_raw = t[0].lower()
            break
    if tense_raw is None:
        t = verb.morph.get("Tense")
        if t:
            tense_raw = t[0].lower()
        else:
            tense_raw = "past" if verb.tag_ in ("VBD", "VBN") else "pres"

    tense = TENSE_MAP.get(tense_raw, tense_raw)

    if aspect == "progressive" and tense == "present":
        time_relation = "ongoing"
    elif tense == "past":
        time_relation = "past"
    else:
        time_relation = "unknown"

    return tense, aspect, time_relation


def _verb_phrase_with_particles(doc, verb, matcher):
    """
    Build (action_text, action_lemma) for a specific verb token inside a doc.

    Priority:
      1) MWE (Matcher) that includes this verb (or is very close to it)
      2) Verb + particles among children (prt / RP / advmod in PARTICLE_WORDS)
      2.5) Verb + whitelisted preposition (prep) with pobj (VERB_PREP_WHITELIST)
      3) Separable particle in a right window (e.g. "pick the book up")

    Returns:
      action_text: surface span like "fell off"
      action: lemma form like "fall off"
    """

    # ---------- helper ----------
    def _dedup(tokens):
        seen = set()
        out = []
        for t in tokens:
            if t.i not in seen:
                seen.add(t.i)
                out.append(t)
        return out

    # ---------- 1) MWE first, but ONLY if it matches this verb ----------
    matches = matcher(doc)
    if matches:
        # keep only matches whose span contains this verb,
        # otherwise in multi-event sentences the 2nd verb may incorrectly inherit the 1st MWE
        filtered = []
        for m_id, start, end in matches:
            if start <= verb.i < end:
                filtered.append((m_id, start, end))

        if filtered:
            # pick the longest
            m_id, start, end = max(filtered, key=lambda x: x[2] - x[1])
            span = doc[start:end]
            action_text = span.text
            action = " ".join(t.lemma_ for t in span)
            return action_text, action

    # ---------- 2) particles (prt/RP/advmod particle words) ----------
    particle_children = []
    for c in verb.children:
        if c.dep_ == "prt":
            particle_children.append(c)
        elif c.tag_ == "RP":
            particle_children.append(c)
        elif c.dep_ == "advmod" and c.lower_ in PARTICLE_WORDS:
            particle_children.append(c)

    # ---------- 2.5) allow selected preps as part of action (whitelist) ----------
    allowed_preps = VERB_PREP_WHITELIST.get(verb.lemma_.lower(), set())
    if allowed_preps:
        for c in verb.children:
            if c.dep_ == "prep" and c.lower_ in allowed_preps:
                # include only if it actually has an object ("off the bicycle")
                if any(x.dep_ == "pobj" for x in c.children):
                    particle_children.append(c)

    # ---------- 3) separable particle to the right (window) ----------
    # helps with "pick the book up" where "up" might not be a child of the verb
    WINDOW = 5  # чуть шире, чтобы "pick the big heavy book up" чаще ловилось
    for i in range(verb.i + 1, min(len(doc), verb.i + 1 + WINDOW)):
        t = doc[i]
        if t.lower_ in PARTICLE_WORDS and (t.tag_ == "RP" or t.pos_ == "ADV"):
            particle_children.append(t)

    # de-dup and order
    particle_children = _dedup(particle_children)
    parts = sorted([verb] + particle_children, key=lambda t: t.i)

    action_text = " ".join(t.text for t in parts)

    # lemma action (keep particle order)
    lemma_parts = [verb.lemma_] + [t.lemma_ for t in sorted(particle_children, key=lambda t: t.i)]
    action = " ".join(lemma_parts)

    return action_text, action


def _collect_event_verbs(doc):
    """
    Return list of verb tokens representing events:
    ROOT verb + its conj verbs (and conj of conj, etc).
    """
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return []

    verbs = [root]

    # BFS over conj verbs
    queue = [root]
    seen = {root.i}
    while queue:
        v = queue.pop(0)
        for c in v.children:
            # conj verbs like "hurt" in "fell ... and hurt ..."
            if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX"):
                if c.i not in seen:
                    seen.add(c.i)
                    verbs.append(c)
                    queue.append(c)

    # keep sentence order
    verbs.sort(key=lambda t: t.i)
    return verbs


def _event_subject(doc, verb, fallback_subject_text=None):
    span = _extract_subject_span(doc, verb)
    if span is not None:
        return span.text
    return fallback_subject_text


def build_action_candidates(doc, verb):
    """
    Return list of lemma-candidates like:
    ["fall", "fall off", "fall apart", "fall in with", ...]
    Only candidates we can justify from syntax:
      - verb alone
      - verb + prt/RP/advmod-particle
      - verb + prep (only if has pobj)
      - verb + prep + pobj head noun (optional, good for 'fall in love')
    """
    cands = set()
    vlem = verb.lemma_.lower()
    cands.add(vlem)

    # particles attached to verb
    particle_tokens = []
    for c in verb.children:
        if c.dep_ == "prt" or c.tag_ == "RP":
            particle_tokens.append(c.lower_)
        elif c.dep_ == "advmod" and c.lower_ in PARTICLE_WORDS:
            particle_tokens.append(c.lower_)
    for p in particle_tokens:
        cands.add(f"{vlem} {p}")

    # prepositions with pobj
    for prep in (c for c in verb.children if c.dep_ == "prep"):
        pobj = next((x for x in prep.children if x.dep_ == "pobj"), None)
        if pobj is None:
            continue
        plem = prep.lower_
        cands.add(f"{vlem} {plem}")

        # optional: include head noun lemma (helps for "fall in love")
        cands.add(f"{vlem} {plem} {pobj.lemma_.lower()}")

    # keep stable order: longest first often helps T5
    cands = sorted(cands, key=lambda s: (-len(s.split()), s))
    return cands


def annotate(sentence: str) -> dict:
    doc = nlp(sentence)

    event_verbs = _collect_event_verbs(doc)
    if not event_verbs:
        return {"example": sentence, "events": []}

    # subject of root as fallback for conj verbs
    root = event_verbs[0]
    root_subj_span = _extract_subject_span(doc, root)
    root_subject = root_subj_span.text if root_subj_span is not None else None

    events = []
    for v in event_verbs:
        action_text, action = _verb_phrase_with_particles(doc, v, matcher)

        subject = _event_subject(doc, v, fallback_subject_text=root_subject)

        obj_span = _extract_object_span(doc, v)
        obj_text = obj_span.text if obj_span is not None else None

        prep, prep_object = _extract_prep_object(
            doc, v, preferred_preps={"with", "to", "into", "for", "from", "of", "off"}
        )

        tense, aspect, time_relation = _tense_aspect(v)

        events.append({
            "action_text": action_text,
            "action": action,
            "tense": tense,
            "tense_aspect": aspect,
            "time_relation": time_relation,
            "subject": subject,
            "object": obj_text,
            "prep": prep,
            "prep_object": prep_object
        })

    return {"example": sentence, "events": events}


if __name__ == "__main__":
    txt = "He fell off the bicycle and hurt his leg."
    print(json.dumps(annotate(txt), ensure_ascii=False, indent=2))
