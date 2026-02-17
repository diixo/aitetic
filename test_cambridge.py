from bs4 import BeautifulSoup
from pathlib import Path
import json


def filter_str(s: str) -> str:
    import re
    if s is not None:
        s = re.sub(r"\s+([.,!?:;\)])", r"\1", s)   # пробелы перед пунктуацией и ')'
        s = re.sub(r"(\()\s+", r"\1", s)           # пробелы после '('
    return s


def clean_text_keep_words(tag):
    """Remove <a> tags, and return only clear cleaned text."""
    if tag is None:
        return ""
    for a in tag.select("a"):
        a.unwrap()
    return filter_str(tag.get_text(" ", strip=True))


html = Path("html/cambridge.org/PUT _ English meaning - Cambridge Dictionary.html").read_text(encoding="utf-8")
soup = BeautifulSoup(html, "lxml")

head_item = None

hdr = soup.select_one("div.pos-header.dpos-h")
if hdr:
    # 1) head_item
    hw_el = hdr.select_one("span.hw.dhw")
    head_item = hw_el.get_text(" ", strip=True) if hw_el else None

    # 2) parts of speech (pos as definition)
    pos_items = [
        {"def": p.get_text(" ", strip=True), "title": p.get("title")}
        for p in hdr.select("span.pos.dpos")
    ]
    print(pos_items)


if not head_item:
    tb = soup.select_one("h1.fs span.tb")
    head_item = filter_str(tb.get_text(strip=True)) if tb else None


print(head_item)

#############################################################################

# one pass by DOM-ordered elements

groups = []

stream = soup.select("div.sense-body.dsense_b")

for body in stream:

    blocks = body.select("div.def.ddef_d.db, div.examp.dexamp, li.eg.dexamp.hax")

    current = None

    for el in blocks:
        classes = el.get("class", [])

        # DEF
        if "def" in classes and "ddef_d" in classes and "db" in classes:
            def_text = clean_text_keep_words(el)
            current = {"title": head_item, "def": def_text, "examples": []}
            groups.append(current)
            continue

        # EXAMPLE
        if "examp" in classes and "dexamp" in classes: #("div.examp.dexamp"):
            lu = el.select_one("span.lu.dlu")
            eg = el.select_one("span.eg.deg")

            if lu:
                key = clean_text_keep_words(lu)
            else:
                key = head_item

            val = clean_text_keep_words(eg)

            # если пример встретился до первого def — можно создать “пустую” группу
            if current is None:
                current = {"title": head_item, "def": None, "examples": []}
                groups.append(current)

            current["examples"].append({"term": key, "text": val})


        if el.name == "li" and "eg" in classes and "dexamp" in classes and "hax" in classes:
            li = clean_text_keep_words(el)
            if li:
                current.setdefault("list", []).append({"term": head_item, "text": li})

##########################
# read separated example-page

stream = soup.select("div.dexamp.fs16.fs18-s.ti")

for body in stream:
    txt = body.select_one("span.deg").get_text(" ", strip=True)
    txt = filter_str(txt)

    current = {"title": head_item, "list": [{"term": head_item, "text": txt}]}

    groups.append(current)

##########################

total_items = sum(len(g.get("examples", [])) for g in groups)
total_items += sum(len(g.get("list", [])) for g in groups)

# show results
print(groups[0] if groups else None)
print("defs:", len(groups), "total_items:", total_items)


# save jsonl
out_path = Path("data/cambridge.org-dataset.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for i, g in enumerate(groups):
        rec = {"idx": i, **g}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
