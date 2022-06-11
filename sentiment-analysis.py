from transformers import pipeline

from utils import Timer, load_labeled_data, load_example_story

# MODEL = "siebert/sentiment-roberta-large-english"  # size: 1.3GB Oo
MODEL = "j-hartmann/sentiment-roberta-large-english-3-classes"  # size: 1.3GB, too
# MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # size: 700 MB, 1-5 star ratings


def evaluate_sentiment_models():
    timer = Timer()
    labeled_sentences = load_labeled_data()

    print("Start loading model")
    with timer:
        # sentiment_classifier = pipeline("sentiment-analysis")
        sentiment_classifier = pipeline(
            task="sentiment-analysis",
            model=MODEL
        )

    print(f"Finished loading model in {timer.time:.3f}s")

    correct_labels = 0
    for sentence, true_label in labeled_sentences:
        print(sentence)
        with timer as t:
            sentiment = sentiment_classifier(sentence)
        label, score = sentiment[0]["label"], sentiment[0]["score"]
        if label.lower() == true_label.lower():
            correct_labels += 1
        print(
            f"sentiment: {label} ({score * 100:.2f}%) -- classification time: {t.time:.3f}s"
        )
        print()

    print(f"Overall accuracy: {correct_labels / (len(labeled_sentences))}")
    # results:
    # siebert/sentiment-roberta-large-english -> loading model: 37s, inference: 0.35s, accuracy: 0.875
    # j-hartmann/sentiment-roberta-large-english-3-classes -> loading model: 16s, inference: 0.35s, accuracy: 0.75
    # distilbert-base-uncased-finetuned-sst-2-english -> loading model: 12s, inference: 0.04s, accuracy: 0.8125
    # nlptown/bert-base-multilingual-uncased-sentiment: 1-5 star ratings, its not too great


def infer_example_story():
    timer = Timer()
    story_fragments = load_example_story()

    print("Start loading model")
    with timer:
        # sentiment_classifier = pipeline("sentiment-analysis")
        sentiment_classifier = pipeline(
            task="sentiment-analysis",
            model=MODEL,
        )
    print(f"Finished loading model in {timer.time:.3f}s")
    print()

    for fragment in story_fragments:
        print(fragment)
        with timer as t:
            sentiment = sentiment_classifier(fragment)
        label, score = sentiment[0]["label"], sentiment[0]["score"]
        print(
            f"sentiment: {label} ({score * 100:.2f}%) -- classification time: {t.time:.3f}s"
        )
        print()


if __name__ == "__main__":
    infer_example_story()
