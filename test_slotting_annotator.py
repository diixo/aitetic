import re
import json
import torch
import spacy
from spacy.matcher import Matcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------- T5 ----------------
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def _t5_generate(prompt: str, max_new_tokens: int = 120) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def t5_pick_action(sentence: str, candidates: list[str], event_hint: str = "") -> tuple[str | None, str | None]:
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

    text = _t5_generate(prompt, max_new_tokens=40)

    m1 = re.search(r"action\s*=\s*(.+?);", text)
    m2 = re.search(r"action_text\s*=\s*(.+)$", text)
    action = m1.group(1).strip() if m1 else None
    action_text = m2.group(1).strip() if m2 else None

    if action not in candidates:
        return None, None
    return action, action_text


def t5_split_clauses(sentence: str) -> list[str]:
    """
    Split sentence into minimal clauses/events.
    Returns a list of clauses as substrings.

    Output format expected from T5: clause1 | clause2 | clause3
    We validate: each clause must have non-trivial length.
    Fallback: split on ';' if T5 output looks bad.
    """
    prompt = (
        "Task: split_into_clauses\n"
        f"Sentence: {sentence}\n"
        "Split the sentence into minimal clauses/events.\n"
        "Return only the clauses separated by ' | ' (pipe).\n"
        "Rules:\n"
        "- Keep the original words (no paraphrasing).\n"
        "- Do NOT add or remove meaning.\n"
        "- Do NOT add numbering or extra text.\n"
        "Example output: Don't expect good work from him | he is lazy and careless\n"
    )

    raw = _t5_generate(prompt, max_new_tokens=80)

    # parse by |
    parts = [p.strip() for p in raw.split("|")]
    parts = [p for p in parts if p]

    # basic validation
    def ok_clause(c: str) -> bool:
        # length and at least one letter
        return len(c) >= 3 and any(ch.isalpha() for ch in c)

    parts = [p for p in parts if ok_clause(p)]

    # fallback if T5 gives junk / single clause that's obviously wrong
    if not parts:
        return [s.strip() for s in sentence.split(";") if s.strip()] or [sentence.strip()]

    # If T5 returns one clause identical to full sentence -> still ok
    # But if it returns one clause that is very short -> fallback
    if len(parts) == 1 and len(parts[0]) < max(6, len(sentence) // 6):
        return [s.strip() for s in sentence.split(";") if s.strip()] or [sentence.strip()]

    return parts


# ---------------- spaCy ----------------
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

matcher.add("FALL_IN_LOVE", [[{"LEMMA": "fall"}, {"LOWER": "in"}, {"LOWER": "love"}]])
matcher.add(
    "FALL_IN_LOVE_FLEX",
    [[{"LEMMA": "fall"}, {"LOWER": "in"}, {"OP": "*", "POS": {"IN": ["ADV", "ADJ"]}}, {"LOWER": "love"}]],
)
matcher.add("FALL_IN_WITH", [[{"LEMMA": "fall"}, {"LOWER": "in"}, {"LOWER": "with"}]])

PARTICLE_WORDS = {
    "apart",
    "up",
    "down",
    "off",
    "out",
    "away",
    "back",
    "over",
    "around",
    "through",
    "along",
    "aside",
    "together",
}

VERB_PREP_WHITELIST = {
    "fall": {"off", "over", "down", "out"},
}

TENSE_MAP = {"pres": "present", "past": "past", "fut": "future"}


# ---------------- helpers: spans ----------------
def _extract_subject_span(doc, verb):
    for c in verb.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return doc[c.left_edge.i : c.right_edge.i + 1]
    return None


def _extract_object_span(doc, verb):
    for c in verb.children:
        if c.dep_ in ("dobj", "obj"):
            # базовый span объекта
            start = c.left_edge.i
            end = c.right_edge.i

            # если внутри span есть prep -> pobj, обрежем на первой prep
            prep_in_subtree = [t for t in c.subtree if t.dep_ == "prep"]
            if prep_in_subtree:
                first_prep = min(prep_in_subtree, key=lambda t: t.i)
                end = min(end, first_prep.i - 1)

            if end < start:
                end = c.i  # fallback хотя бы на голову

            return doc[start : end + 1]
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


def _extract_copula_complement(doc, verb):
    if verb.lemma_.lower() != "be":
        return None
    for dep in ("attr", "acomp"):
        for c in verb.children:
            if c.dep_ == dep:
                return doc[c.left_edge.i : c.right_edge.i + 1]
    return None


# ---------------- tense/aspect ----------------
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
        time_relation = "present" if tense == "present" else "unknown"

    return tense, aspect, time_relation


# ---------------- polarity / imperative ----------------
def _polarity(verb):
    for c in verb.children:
        if c.dep_ == "neg":
            return "negative"
    for c in verb.children:
        if c.dep_ in ("aux", "auxpass"):
            for cc in c.children:
                if cc.dep_ == "neg":
                    return "negative"
    return "positive"


def _is_imperative(doc, verb):
    if _extract_subject_span(doc, verb) is not None:
        return False
    if verb.tag_ != "VB":
        return False
    if verb.dep_ not in ("ROOT", "conj", "parataxis"):
        return False
    return True


# ---------------- action: rule-based (MWE + particles + whitelisted prep) ----------------
def _verb_phrase_with_particles(doc, verb, matcher):
    def _dedup(tokens):
        seen = set()
        out = []
        for t in tokens:
            if t.i not in seen:
                seen.add(t.i)
                out.append(t)
        return out

    matches = matcher(doc)
    if matches:
        filtered = [(m_id, s, e) for (m_id, s, e) in matches if s <= verb.i < e]
        if filtered:
            m_id, start, end = max(filtered, key=lambda x: x[2] - x[1])
            span = doc[start:end]
            return span.text, " ".join(t.lemma_ for t in span)

    particle_children = []
    for c in verb.children:
        if c.dep_ == "prt":
            particle_children.append(c)
        elif c.tag_ == "RP":
            particle_children.append(c)
        elif c.dep_ == "advmod" and c.lower_ in PARTICLE_WORDS:
            particle_children.append(c)

    allowed_preps = VERB_PREP_WHITELIST.get(verb.lemma_.lower(), set())
    if allowed_preps:
        for c in verb.children:
            if c.dep_ == "prep" and c.lower_ in allowed_preps:
                if any(x.dep_ == "pobj" for x in c.children):
                    particle_children.append(c)

    WINDOW = 5
    for i in range(verb.i + 1, min(len(doc), verb.i + 1 + WINDOW)):
        t = doc[i]
        if t.lower_ in PARTICLE_WORDS and (t.tag_ == "RP" or t.pos_ == "ADV"):
            particle_children.append(t)

    particle_children = _dedup(particle_children)
    parts = sorted([verb] + particle_children, key=lambda t: t.i)

    action_text = " ".join(t.text for t in parts)
    action = " ".join([verb.lemma_] + [t.lemma_ for t in sorted(particle_children, key=lambda t: t.i)])
    return action_text, action


# ---------------- candidates for T5 ----------------
def build_action_candidates(doc, verb):
    cands = set()
    vlem = verb.lemma_.lower()
    cands.add(vlem)

    for c in verb.children:
        if c.dep_ == "prt" or c.tag_ == "RP":
            cands.add(f"{vlem} {c.lower_}")
        elif c.dep_ == "advmod" and c.lower_ in PARTICLE_WORDS:
            cands.add(f"{vlem} {c.lower_}")

    for prep in (c for c in verb.children if c.dep_ == "prep"):
        pobj = next((x for x in prep.children if x.dep_ == "pobj"), None)
        if pobj is None:
            continue
        plem = prep.lower_
        cands.add(f"{vlem} {plem}")
        cands.add(f"{vlem} {plem} {pobj.lemma_.lower()}")

    matches = matcher(doc)
    for m_id, start, end in matches:
        if start <= verb.i < end:
            span = doc[start:end]
            cands.add(" ".join(t.lemma_ for t in span))

    return sorted(cands, key=lambda s: (-len(s.split()), s))


# ---------------- collect event verbs: ROOT + conj + parataxis ----------------
def _collect_event_verbs(doc):
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return []

    verbs = []
    queue = [root]
    seen = set()

    def add_if_verb(t):
        if t is None:
            return
        if t.pos_ in ("VERB", "AUX") and t.i not in seen:
            seen.add(t.i)
            verbs.append(t)
            queue.append(t)

    add_if_verb(root)

    while queue:
        v = queue.pop(0)
        for c in v.children:
            if c.dep_ == "conj":
                add_if_verb(c)
            if c.dep_ == "parataxis":
                add_if_verb(c)

    verbs.sort(key=lambda t: t.i)
    return verbs


# ---------------- annotate one clause ----------------
def annotate_clause(clause: str) -> list[dict]:
    doc = nlp(clause)
    event_verbs = _collect_event_verbs(doc)
    if not event_verbs:
        return []

    root = event_verbs[0]
    root_subj_span = _extract_subject_span(doc, root)
    root_subject = root_subj_span.text if root_subj_span is not None else None

    events = []
    for v in event_verbs:
        subj_span = _extract_subject_span(doc, v)
        subject = subj_span.text if subj_span is not None else root_subject

        mood = None
        speech_act = None
        polarity = _polarity(v)

        if _is_imperative(doc, v):
            mood = "imperative"
            speech_act = "directive"
            if subject is None:
                subject = "you"

        obj_span = _extract_object_span(doc, v)
        obj_text = obj_span.text if obj_span is not None else None

        if obj_text is None and v.lemma_.lower() == "be":
            comp = _extract_copula_complement(doc, v)
            if comp is not None:
                obj_text = comp.text

        prep, prep_object = _extract_prep_object(
            doc, v, preferred_preps={"with", "to", "into", "for", "from", "of", "off"}
        )

        if prep is None and obj_span is not None:
            for t in obj_span.root.children:
                if t.dep_ == "prep":
                    pobj = next((x for x in t.children if x.dep_ == "pobj"), None)
                    if pobj is not None:
                        prep = t.lower_
                        prep_object = doc[pobj.left_edge.i : pobj.right_edge.i + 1].text
                        break

        tense, aspect, time_relation = _tense_aspect(v)

        if mood == "imperative":
            tense = "present"
            aspect = "simple"
            time_relation = "present"

        candidates = build_action_candidates(doc, v)
        event_hint = f"Focus verb lemma: {v.lemma_.lower()}\n"

        # IMPORTANT: pick action using the CLAUSE (so action_text is a substring)
        picked_action, picked_text = t5_pick_action(clause, candidates, event_hint=event_hint)

        if picked_action and picked_text and picked_text in clause:
            action = picked_action
            action_text = picked_text
        else:
            action_text, action = _verb_phrase_with_particles(doc, v, matcher)

        ev = {
            "action_text": action_text,
            "action": action,
            "tense": tense,
            "tense_aspect": aspect,
            "time_relation": time_relation,
            "subject": subject,
            "object": obj_text,
            "prep": prep,
            "prep_object": prep_object,
            # optional debug:
            "clause": clause,
        }

        if mood is not None:
            ev["mood"] = mood
        if speech_act is not None:
            ev["speech_act"] = speech_act
        if polarity != "positive":
            ev["polarity"] = polarity

        events.append(ev)

    return events


# ---------------- annotate full sentence: T5 split -> annotate each clause -> merge ----------------
def annotate(sentence: str) -> dict:
    clauses = t5_split_clauses(sentence)
    all_events = []
    for cl in clauses:
        all_events.extend(annotate_clause(cl))
    # remove debug field if you don't want it in dataset:
    for ev in all_events:
        ev.pop("clause", None)
    return {"example": sentence, "events": all_events}


# ---------------- demo ----------------
if __name__ == "__main__":
    s = "Don't expect good work from him; he is lazy and careless."
    print(json.dumps(annotate(s), ensure_ascii=False, indent=2))
