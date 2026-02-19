import json


def annotate(sentence):
    return {
        "example": sentence,
        "annotation": []
    }



if __name__ == "__main__":

    output_file = "test_slotting_annotator.json"

    # 1920
    data = []

    with open(output_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # force annotate empty annotations
    for item in dataset:
        annotation = item.get("annotation", [])

        if not annotation:
            result = annotate(item["example"])
            result = result.get("annotation", [])
            if result:
                item["annotation"] = result


    if len(data) > 0:
        #data = [s["example"] for s in dataset]

        print("Unique examples in dataset:", len(set(data)))


        dataset = [annotate(s) for s in data]


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


    print("Examples dataset.sz:", len(dataset))
