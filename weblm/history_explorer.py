"""Search the examples.json file and delete any bad entries."""

import json
import os

import cohere
import numpy as np

co = cohere.Client(os.environ.get("COHERE_KEY"))


def search_history(query, history):
    embeds = [h["embedding"] for h in history]
    examples = [h["example"] for h in history]
    embeds = np.array(embeds)
    embedded_state = np.array(co.embed(texts=[query], truncate="RIGHT").embeddings[0])
    scores = np.einsum("i,ji->j", embedded_state,
                       embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
    ind = np.argsort(scores)
    return np.array(examples)[ind], ind


if __name__ == "__main__":
    with open("examples.json", "r") as fd:
        history = json.load(fd)

    indices_for_deletion = []
    s = ""

    try:
        while True:
            s = input("Search: ")
            examples, ind = search_history(s, history)

            for ex, i in reversed(list(zip(examples, ind))):
                print(f"Example:\n{ex}"
                      "\n\t(n) next example"
                      "\n\t(d) delete example")
                s = input("Command: ")
                if s == "n":
                    continue
                elif s == "d":
                    indices_for_deletion.append(i)
                else:
                    raise Exception()
    except Exception as e:
        print(e)
        for i in sorted(indices_for_deletion, reverse=True):
            history.pop(i)

        with open("examples_tmp.json", "w") as fd:
            json.dump(history, fd)
        os.replace("examples_tmp.json", "examples.json")
