from pathlib import Path
from time import perf_counter
from typing import List, Tuple

POS_SENTENCES_FP = Path("data/positive_statements.txt")
NEG_SENTENCES_FP = Path("data/negative_statements.txt")
EXAMPLE_STORY_FP = Path("data/example_story.txt")


class Timer:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time


def load_example_story() -> List[str]:
    with open(EXAMPLE_STORY_FP, "r") as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences]
    return sentences


def load_labeled_data() -> List[Tuple[str, str]]:
    def load(filepath: Path, label: str) -> List[Tuple[str, str]]:
        with open(filepath, "r") as f:
            sentences = f.readlines()
        sentences = [(s.strip(), label) for s in sentences]
        return sentences

    data_and_labels = load(POS_SENTENCES_FP, label="POSITIVE") + load(
        NEG_SENTENCES_FP, label="NEGATIVE"
    )
    return data_and_labels

