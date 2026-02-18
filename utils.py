
from pathlib import Path


def load_mwe_phrases(path: str, matcher, *, name_prefix: str = "MWE_"):
    """
    Load multi-word expressions (MWEs) from a text file and add them to spaCy Matcher.

    File format (one MWE per line):
      fall in love
      fall in with
      put up with
      look forward to
      # comments allowed

    Matching strategy:
      - First token matched by LEMMA (more robust to inflection)
      - Remaining tokens matched by LOWER (keeps fixed particles/preps stable)

    Example line -> pattern:
      "fall in love" =>
        [{"LEMMA":"fall"}, {"LOWER":"in"}, {"LOWER":"love"}]

    Returns:
      list of names added to matcher
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MWE file not found: {p}")

    added = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        toks = line.split()
        if len(toks) < 2:
            # skip single words: they aren't MWEs
            continue

        first, rest = toks[0], toks[1:]
        pattern = [{"LEMMA": first}]
        pattern += [{"LOWER": t} for t in rest]

        # stable name; avoid spaces
        name = name_prefix + "_".join(toks).upper()

        # Add safely: if already exists, overwrite by removing and re-adding isn't supported directly,
        # so we just add a unique suffix in that rare case.
        try:
            matcher.add(name, [pattern])
        except ValueError:
            # already exists -> make a unique name
            suffix = f"__L{line_no}"
            matcher.add(name + suffix, [pattern])
            name = name + suffix

        added.append(name)

    return added
