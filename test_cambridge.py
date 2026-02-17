from bs4 import BeautifulSoup
from pathlib import Path

html = Path("html/STAY _ English meaning - Cambridge Dictionary.html").read_text(encoding="utf-8")
soup = BeautifulSoup(html, "lxml")

defs = []
for el in soup.select("div.def.ddef_d.db"):
    # убрать <a>, но оставить их текст
    for a in el.select("a"):
        a.unwrap()

    text = el.get_text(" ", strip=True)
    #print(text)
    defs.append(text)

for i, d in enumerate(defs):
    print(f"{i}: {d}")


out = []
for ex in soup.select("div.examp.dexamp"):
    key_el = ex.select_one("span.lu.dlu")
    val_el = ex.select_one("span.eg.deg")
    if not key_el or not val_el:
        continue

    key = key_el.get_text(" ", strip=True)

    # убрать ссылки в примере, оставить только текст
    for a in val_el.select("a"):
        a.unwrap()

    val = val_el.get_text(" ", strip=True)

    out.append({"key": key, "value": val})


for i, d in enumerate(out):
    print(f"{i}: {d}")
