
from itertools import islice


count = 50


if __name__ == "__main__":

    #with open("datasets/paracrawl-eu/en-ru.txt", "r", encoding="utf-8") as f:
    with open("data/manythings.org/rus.txt", "r", encoding="utf-8") as f:

        for i, line in enumerate(islice(f, count), 1):

            n_tabs = line.count("\t")

            if n_tabs == 0:
                continue

            # always get two first columns, even if there are more
            first, second = line.split("\t", 2)[:2]

            print(i, first[:80], second[:80])

