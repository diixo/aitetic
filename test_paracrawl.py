
from itertools import islice
from pathlib import Path
import json


def str_tokenize_words(s: str):
    import re
    s = re.findall("(\.?\w[\w'\.&-]*\w|\w\+*#?)", s)
    if s: return s
    return []



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

            ws = str_tokenize_words(eng)

            if eng not in texts:
                texts[eng] = len(ws)


if __name__ == "__main__":

    file_name = "data/eng-base.jsonl"

    texts = dict()

    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            example = obj.get("example")
            ws = obj.get("words")

            if int(ws) < 1:
                ws = len(str_tokenize_words(example))

            texts[example] = ws


    load_translation_file("data/manythings.org/rus.txt", texts)

    load_translation_file("data/manythings.org/spa.txt", texts)

    load_translation_file("data/manythings.org/fra.txt", texts)

    load_translation_file("data/manythings.org/deu.txt", texts)

    load_translation_file("data/manythings.org/swe.txt", texts)

    load_translation_file("data/manythings.org/tur.txt", texts)

    load_translation_file("data/manythings.org/por.txt", texts)

    load_translation_file("data/manythings.org/jpn.txt", texts)

    load_translation_file("data/manythings.org/ita.txt", texts)

    load_translation_file("data/manythings.org/ukr.txt", texts)

    load_translation_file("data/manythings.org/heb.txt", texts)

    load_translation_file("data/manythings.org/pol.txt", texts)

    load_translation_file("data/manythings.org/ces.txt", texts)

    load_translation_file("data/manythings.org/fin.txt", texts)

    load_translation_file("data/manythings.org/hun.txt", texts)

    load_translation_file("data/manythings.org/nld.txt", texts)

    load_translation_file("data/manythings.org/mkd.txt", texts)

    load_translation_file("data/manythings.org/dan.txt", texts)

    load_translation_file("data/manythings.org/ron.txt", texts)

    load_translation_file("data/manythings.org/ber.txt", texts)

    load_translation_file("data/manythings.org/srp.txt", texts)

    load_translation_file("data/manythings.org/cmn.txt", texts)

    #load_translation_file("data/manythings.org/lit.txt", texts)

    #load_translation_file("data/manythings.org/nds.txt", texts)

    out_path = Path(file_name)
    with out_path.open("w", encoding="utf-8") as f:
        i = 1
        for k, v in texts.items():
            rec = {"idx": i, "example": k, "words": v}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            i += 1

    print(40*"#")
    print("unique words:", len(texts.items())) # 742080
