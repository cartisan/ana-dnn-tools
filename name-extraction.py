from flair.data import Sentence
from flair.models import SequenceTagger

from utils import Timer

MODEL = "flair/ner-english-ontonotes-large"

TEST_SENTENCES = [
    "My name is Leon Berov.",
    "I'm Leonid.",
    "Leon.",
    "I'm... ehm... let's say it's Bobby.",
    "I'm General House from Washington, howdy."
]


def extract_name():
    timer = Timer()

    print("Start loading model")
    with timer:
        tagger = SequenceTagger.load(MODEL)
    print(f"Finished loading model in {timer.time:.3f}s")
    print()

    for s in TEST_SENTENCES:
        print(f"----------- {s} -----------")
        sentence = Sentence(s)
        with timer:
            tagger.predict(sentence)
        print(f"Finished prediction in {timer.time:.3f}s")

        names = []
        for entity in sentence.get_spans('ner'):
            if entity.tag == 'PERSON':
                tokens = [token.text for token in entity.tokens]
                name = " ".join(tokens)
                names.append(name)

        if len(names) == 0:
            raise ValueError("Didn't find a name in the text")
        if len(names) > 1:
            raise ValueError(f"Found more than one name: {names}")

        print(f"Identified name: {names[0]}")


if __name__ == "__main__":
    extract_name()
