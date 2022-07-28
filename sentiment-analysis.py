import logging

from transformers import pipeline

from utils import Timer, load_labeled_data, load_example_story

# MODEL = "siebert/sentiment-roberta-large-english"  # size: 1.3GB Oo
MODEL = "j-hartmann/sentiment-roberta-large-english-3-classes"  # size: 1.3GB, too
# MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # size: 700 MB, 1-5 star ratings

TIMER = Timer()

logging.basicConfig(level=logging.INFO)


def create_sentiment_anaysis_model() -> pipeline:
    """ Instantiates a sentiment analyzer that can process individual sentences.
    Attention: This operation is costly and can take between 30s and 90s time.
    """
    logging.info(f"Loading sentiment analysis model `{MODEL}`, this will take some time")
    with TIMER:
        sentiment_classifier = pipeline(
            task="sentiment-analysis",
            model=MODEL
        )
    logging.info(f"Finished loading model in {TIMER.time:.3f}s")
    return sentiment_classifier


def analyze_sentiment(text: str, sentiment_classifier: pipeline) -> str:
    """ Takes an instantiated transformers pipeline and a text, which can be both
    a sentence or a paragraph. Then, performs sentiment analysis, determines a label
    as well as the classifier's prediction confidence, and returns the label.
    Label can be either one of {"positive", "neutral", "negative"}.
    """
    logging.info(f"Start analyzing sentiment of {text=}")
    with TIMER:
        sentiment = sentiment_classifier(text)
    logging.info(f"Finished inference in {TIMER.time:.3f}s")

    label, score = sentiment[0]["label"], sentiment[0]["score"]
    logging.info(
        f"Identified sentiment: {label} confidence: ({score * 100:.2f}%)"
    )
    return label


def _evaluate_sentiment_models():
    labeled_sentences = load_labeled_data()
    sentiment_classifier = create_sentiment_anaysis_model()

    correct_labels = 0
    for sentence, true_label in labeled_sentences:
        label = analyze_sentiment(sentence, sentiment_classifier)
        if label.lower() == true_label.lower():
            correct_labels += 1

    print(f"Overall accuracy: {correct_labels / (len(labeled_sentences))}")
    # results:
    # siebert/sentiment-roberta-large-english -> loading model: 37s, inference: 0.35s, accuracy: 0.875
    # j-hartmann/sentiment-roberta-large-english-3-classes -> loading model: 16s, inference: 0.35s, accuracy: 0.75
    # distilbert-base-uncased-finetuned-sst-2-english -> loading model: 12s, inference: 0.04s, accuracy: 0.8125
    # nlptown/bert-base-multilingual-uncased-sentiment: 1-5 star ratings, its not too great


def _infer_example_story():
    story_fragments = load_example_story()
    sentiment_classifier = create_sentiment_anaysis_model()

    for fragment in story_fragments:
        label = analyze_sentiment(fragment, sentiment_classifier)
        print(f"{fragment} -> SENTIMENT: {label}")


if __name__ == "__main__":
    # _evaluate_sentiment_models()
    _infer_example_story()
