
from itertools import islice


def str_tokenize_words(s: str):
    import re
    s = re.findall("(\.?\w[\w'\.&-]*\w|\w\+*#?)", s)
    if s: return s
    return []


count = 50

words = set()


if __name__ == "__main__":

    #with open("datasets/paracrawl-eu/en-ru.txt", "r", encoding="utf-8") as f:
    with open("data/manythings.org/rus.txt", "r", encoding="utf-8") as f:


        i = 0
        for line in f:
            n_tabs = line.count("\t")

            if n_tabs == 0:
                continue

            i += 1

            if i > count:
                break

            # always get two first columns, even if there are more
            eng, translation = line.split("\t", 2)[:2]

            print(i, eng[:80], translation[:80])

            ws = str_tokenize_words(eng.lower())
            words.update(ws)

print(40*"#")
print("unique words:", len(sorted(words)))

file_name = "html/STAY _ English meaning - Cambridge Dictionary.html"