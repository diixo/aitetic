

Person_names = [
    'Aaron', 'Adam', 'Alan', 'Alexander', 'Alfred', 'Alice', 'Amanda', 'Amber', 'Amelia', 'Amy', 'Anatole', 'Anna', 'Audrey',
    'Anastasia', 'Andrew', 'Angela', 'Angelina', 'Anna', 'Anthony', 'Arthur', 'Ashley',
    'Barbara', 'Benjamin', 'Betty', 'Bill', 'Bob', 'Brad', 'Brandon', 'Brenda', 'Brian', 'Bruce', 'Bryan',
    'Cameron', 'Carl', 'Carlos', 'Carol', 'Caroline', 'Carolyn', 'Casey', 'Catherine', 'Cecil', 'Cedric', 'Charles', 'Charlie',
    'Charlotte', 'Cheryl', 'Christian', 'Christine', 'Christopher', 'Christy', 'Cindy', 'Clara', 'Crystal', 'Cynthia',
    'Daniel', 'David', 'Deborah', 'Dennis', 'Diana', 'Donald', 'Donna', 'Dorothy', "Dylan",
    'Edward', 'Eleanor', 'Elizabeth', 'Emily', 'Emma', 'Eric', 'Ethan', 'Eve', 'Evelyn',
    'Frank', 'Frederick',
    'Gabriel', 'Gabriella', 'Gary', 'George', 'Gerald', 'Grace', 'Gregory', "Greg",
    'Hannah', 'Halle', 'Harold', 'Helen', 'Henry', 'Howard', 'Hugh',
    'Ian', 'Isaac', 'Isabella', 'Ivan', 
    'Jack', 'Jacob', 'Jake', 'James', 'Jamie', 'Jane', 'Jason', 'Jasper', 'Jay', 'Jeff', 'Jeffrey',
    'Jennifer', 'Jenny', 'Jeremy', 'Jerome', 'Jerry', 'Jesse', 'Jessica', 'Jim', 'Jimmy', 'Joe', 'Joel',
    'John', 'Johnny', 'Jonathan', 'Jordan', 'Jose', 'Joseph', 'Josh', 'Joshua', 'Julia', 'Justin',
    'Karen', 'Katherine', 'Kathleen', 'Katie', 'Keith', 'Kenneth', 'Kevin', 'Kimberly', 'Kyle',
    'Larry', 'Laura', 'Linda', 'Lisa', 'Liz',
    'Margaret', 'Mark', 'Martin', 'Mary', 'Matthew', 'Megan', 'Melanie', 'Melissa', 'Michael', 'Michelle', "Mike", 'Molly', 'Monica', 'Morgan',
    'Nancy', 'Natalie', 'Nathan', 'Nicholas', 'Nicole',
    'Olivia', 'Oscar', 
    'Pamela', 'Patrick', 'Paul', 'Peter', 'Philip',
    'Rachel', 'Raymond', 'Rebecca', 'Robert', 'Roger', 'Ronald', 'Ryan', "Ronny", "Ronnie",
    'Samantha', 'Samuel', 'Sandra', 'Sarah', 'Scarlett', 'Scott', 'Sharon', 'Shirley', 'Sophia', 'Stephanie', 'Stephen', 'Steven', 'Susan',
    'Teddy', 'Terry', 'Theodore', 'Thomas', 'Tim', 'Timothy', 'Tina', 'Tom', 'Tyler',
    'Vanessa', 'Veronica', 'Victor', 'Victoria', 'Vincent',
    'Walter', "Warren", 'William',
    'Zachary']


import json
from annotated_types import doc
import spacy
from spacy.matcher import Matcher


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

matcher.add("FALL_IN_LOVE", [[
    {"LEMMA": "fall"},
    {"LOWER": "in"},
    {"LOWER": "love"}
]])

# если хочешь разрешить вставки типа "madly", "deeply":
matcher.add("FALL_IN_LOVE_FLEX", [[
    {"LEMMA": "fall"},
    {"LOWER": "in"},
    {"OP": "*", "POS": {"IN": ["ADV", "ADJ"]}},   # optional modifiers
    {"LOWER": "love"}
]])

PARTICLE_WORDS = {
    "apart", "up", "down", "off", "out", "away", "back", "over", "around",
    "through", "along", "aside", "together"
}


def _verb_phrase_with_particles(doc, verb, matcher):
    # 1) MWE matcher first (e.g. "fell in love")
    matches = matcher(doc)
    if matches:
        m_id, start, end = max(matches, key=lambda x: x[2] - x[1])
        span = doc[start:end]
        action_text = span.text
        action = " ".join(t.lemma_ for t in span)
        return action_text, action

    # 2) collect likely particles among children
    particle_children = []
    for c in verb.children:
        if c.dep_ == "prt":
            particle_children.append(c)
        elif c.tag_ == "RP":
            particle_children.append(c)
        elif c.dep_ == "advmod" and c.lower_ in PARTICLE_WORDS:
            particle_children.append(c)

    # 3) catch separable particle to the right: "pick the book up"
    WINDOW = 4
    for i in range(verb.i + 1, min(len(doc), verb.i + 1 + WINDOW)):
        t = doc[i]
        if t.lower_ in PARTICLE_WORDS and (t.tag_ == "RP" or t.pos_ == "ADV"):
            if all(p.i != t.i for p in particle_children):
                particle_children.append(t)

    parts = sorted([verb] + particle_children, key=lambda t: t.i)
    action_text = " ".join(t.text for t in parts)

    lemma_parts = [verb.lemma_] + [p.lemma_ for p in sorted(particle_children, key=lambda t: t.i)]
    action = " ".join(lemma_parts)

    return action_text, action


def _extract_subject_span(doc, root):
    """
    Return grammatical subject as a spaCy Span (not a generator).
    """
    for c in root.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return doc[c.left_edge.i : c.right_edge.i + 1]
    return None


def _tense_aspect(root):
    """
    Very lightweight tense/aspect detection using auxiliaries + tag.
    Good enough for a starter dataset builder.
    """
    aux = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    aux_lemmas = {a.lemma_.lower() for a in aux}

    # aspect
    # progressive if VBG ("falling", "doing") and has auxiliary "be"
    if root.tag_ == "VBG" and ("be" in aux_lemmas or any(a.lemma_ == "be" for a in aux)):
        aspect = "progressive"
    # perfect if auxiliary "have"
    elif "have" in aux_lemmas:
        aspect = "perfect"
    else:
        aspect = "simple"

    # tense
    # use aux tense if exists, else root morph/tag
    tense = None
    for a in aux:
        # Morphological features if available
        t = a.morph.get("Tense")
        if t:
            tense = t[0].lower()
            break
    if tense is None:
        t = root.morph.get("Tense")
        if t:
            tense = t[0].lower()
        else:
            # fallback from tag
            if root.tag_ in ("VBD", "VBN"):
                tense = "past"
            else:
                tense = "present"


    # normalize tense labels (e.g. "pres" -> "present")
    TENSE_MAP = {
        "pres": "present",
        "past": "past",
        "fut": "future",
    }

    tense = TENSE_MAP.get(tense, tense)  # normalize tense labels

    # time_relation (starter heuristic)
    if aspect == "progressive" and tense == "present":
        time_relation = "ongoing"
    elif tense == "past":
        time_relation = "past"
    else:
        time_relation = "unknown"

    return tense, aspect, time_relation


def annotate(sentence: str) -> dict:

    doc = nlp(sentence)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return {"example": sentence}

    action_text, action = _verb_phrase_with_particles(doc, root, matcher)

    subj_span = _extract_subject_span(doc, root)
    subject = subj_span.text if subj_span is not None else None

    tense, aspect, time_relation = _tense_aspect(root)

    return {
        "example": sentence,
        "action_text": action_text,
        "action": action,
        "tense": tense,
        "tense_aspect": aspect,
        "time_relation": time_relation,
        "subject": subject
    }


if __name__ == "__main__":
    
    txt = "She fell in love with jazz music."

    #txt = "His business is falling apart."

    txt = "She fell in with a strange crowd of people at university."

    result = annotate(txt)

    print(f"result =\n{json.dumps(result, indent=2)}")

