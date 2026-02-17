from bs4 import BeautifulSoup
from pathlib import Path
import json


def clean_text_keep_words(tag):
    """Remove <a> tags, and return only clear cleaned text."""
    if tag is None:
        return ""
    for a in tag.select("a"):
        a.unwrap()
    return tag.get_text(" ", strip=True)


html = Path("html/cambridge.org/STAY _ English meaning - Cambridge Dictionary.html").read_text(encoding="utf-8")
soup = BeautifulSoup(html, "lxml")


hdr = soup.select_one("div.pos-header.dpos-h")
if not hdr:
    raise RuntimeError("pos-header not found")

# 1) head_item
hw_el = hdr.select_one("span.hw.dhw")
head_item = hw_el.get_text(" ", strip=True) if hw_el else None

# 2) parts of speech (pos as definition)
pos_items = [
    {"def": p.get_text(" ", strip=True), "title": p.get("title")}
    for p in hdr.select("span.pos.dpos")
]

print(head_item)
print(pos_items)

#############################################################################

# один проход в DOM-порядке
stream = soup.select("div.def.ddef_d.db, div.examp.dexamp")

groups = []
current = None

for el in stream:
    classes = el.get("class", [])

    # DEF
    if "def" in classes and "ddef_d" in classes and "db" in classes:
        def_text = clean_text_keep_words(el)
        current = {"def": def_text, "examples": []}
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
            current = {"def": None, "examples": []}
            groups.append(current)

        current["examples"].append({"term": key, "text": val})


# show results
print("defs:", len(groups))
print(groups[0] if groups else None)

# save jsonl
out_path = Path("data/cambridge.org-dataset.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for i, g in enumerate(groups):
        rec = {"idx": i, **g}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
