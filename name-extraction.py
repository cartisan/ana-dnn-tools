import logging

from flair.data import Sentence
from flair.models import SequenceTagger

from utils import Timer

TEST_SENTENCES = [
    "My name is Leon Berov.",
    "I'm Leonid.",
    "Leon.",
    "I'm... ehm... let's say it's Bobby.",
    "I'm General House from Washington, howdy.",
    "Sorry, I didn't get that?",
]
TIMER = Timer()
MODEL = "flair/ner-english-ontonotes-large"

logging.basicConfig(level=logging.INFO)


def create_ner_model() -> SequenceTagger:
    """ Instantiates an NER tagger that can analyze sentences.
    Attention: This operation is costly and can take between 30s and 90s time.
    """
    logging.info(f"Loading NER `{MODEL}`, this will take some time")
    with TIMER:
        tagger = SequenceTagger.load(MODEL)
    logging.info(f"Finished loading model in {TIMER.time:.3f}s")
    return tagger


def extract_name(sentence: str, tagger: SequenceTagger) -> str:
    """ Takes an instantiated flair SequenceTagger and a sentence, performs NER tagging,
    and extracts the name in the sentence based on it being tagged `PERSON`. Errors if
    no person is present, and returns only the first in case multiple persons are detected.
    """
    logging.info(f"Start extracting a name from {sentence=}")
    sentence = Sentence(sentence)
    with TIMER:
        tagger.predict(sentence)
    logging.info(f"Finished inference in {TIMER.time:.3f}s")
    logging.info(f"Identified named entities: {sentence.get_spans('ner')}")

    names = []
    for entity in sentence.get_spans("ner"):
        if entity.tag == "PERSON":
            # each entity can contain multiples tokens, like: [Leonid, Berov] -> PERSON
            # we want to concatenate the tokens to a space separated string
            tokens = [token.text for token in entity.tokens]
            name = " ".join(tokens)
            names.append(name)

    # make sure we only found one name, and error out otherwise
    if len(names) == 0:
        logging.error("Didn't find a name in the text")
        raise ValueError("Didn't find a name in the text")
    if len(names) > 1:
        logging.warning(f"Found more than one name: {names}, returning only first")

    result = names[0]
    logging.info(f"Returning name: {result}")
    return result


def _test_name_extraction():
    tagger = create_ner_model()
    for s in TEST_SENTENCES:
        logging.info(f"----------------------")
        try:
            name = extract_name(s, tagger)
        except ValueError:
            name = "no name found"
        print(name)


if __name__ == "__main__":
    _test_name_extraction()
