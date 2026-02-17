
from itertools import islice
from pathlib import Path
import json


def str_tokenize_words(s: str):
    import re
    s = re.findall("(\.?\w[\w'\.&-]*\w|\w\+*#?)", s)
    if s: return s
    return []


count = 0

words = set()
texts = dict()


def load_translation_file(file_path: str, texts: dict, count: int = 0):

    #with open("datasets/paracrawl-eu/en-ru.txt", "r", encoding="utf-8") as f:
    with open(file_path, "r", encoding="utf-8") as f:

        i = 0
        for line in f:
            n_tabs = line.count("\t")

            if n_tabs == 0:
                continue

            i += 1

            if i > count and count > 0:
                break

            # always get two first columns, even if there are more
            eng, translation = line.split("\t", 2)[:2]

            #print(i, eng[:80], translation[:80])
            if i % 1000 == 0:
                print(f"...on: {i} lines")

            ws = str_tokenize_words(eng.lower())
            words.update(ws)

            if eng not in texts:
                texts[eng] = len(ws)


if __name__ == "__main__":

    texts = dict()

    load_translation_file("data/manythings.org/rus.txt", texts)

    load_translation_file("data/manythings.org/spa.txt", texts)

    load_translation_file("data/manythings.org/fra.txt", texts)

    load_translation_file("data/manythings.org/deu.txt", texts)

    load_translation_file("data/manythings.org/swe.txt", texts)

    load_translation_file("data/manythings.org/tur.txt", texts)

    load_translation_file("data/manythings.org/por.txt", texts)

    out_path = Path("data/eng-base.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        i = 1
        for k, v in texts.items():
            rec = {"idx": i, "example": k, "words": v}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            i += 1

print(40*"#")
print("unique words:", len(texts.items()))
