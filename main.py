import json
import logging

import zmq

from sentiment_analysis import SentimentClassifier

logging.basicConfig(level=logging.INFO)


class Runner:
    def __init__(self):
        # initialize zmq connection
        context = zmq.Context()
        self.server = context.socket(zmq.REP)
        self.server.bind("tcp://*:8107")

        # initialize sentiment classifier
        self.sentiment_classifier = SentimentClassifier()

    def start_main_loop(self):
        logging.info("Starting sentiment analysis loop")
        while True:
            request = self.server.recv_multipart()
            logging.info(f"Received request: {request}")

            data = json.loads(request[1])
            user_continuation = data["userContinuation"]
            ana_continuation = data["anaContinuation"]

            user_sentiment = self.sentiment_classifier.analyze_sentiment(user_continuation)
            ana_sentiment = self.sentiment_classifier.analyze_sentiment(ana_continuation)

            topic = b"GPT3 REP from demo_turn_server"
            content = json.dumps( {
                "userSentiment": user_sentiment,
                "anaSentiment:":  ana_sentiment,
            })
            response = [topic, content.encode()]

            self.server.send_multipart(response)
            logging.info(f"Done, response: {response}")


if __name__ == "__main__":
    r = Runner()
    r.start_main_loop()
