import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel

class TopicModeling:
    def __init__(self, documents, num_topics=2, passes=10):
        self.documents = documents
        self.num_topics = num_topics
        self.passes = passes
        self.stop_words = set(stopwords.words('english'))
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def preprocess(self, text):
        tokens = [word for word in text.lower().split() if word.isalpha() and word not in self.stop_words]
        return tokens

    def prepare_corpus(self):
        processed_docs = [self.preprocess(doc) for doc in self.documents]
        self.dictionary = corpora.Dictionary(processed_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

    def build_lda_model(self):
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, random_state=42, passes=self.passes)

    def print_topics(self, num_words=4):
        topics = self.lda_model.print_topics(num_words=num_words)
        for topic in topics:
            print(topic)

def main():
    # Download the stopwords from NLTK
    nltk.download('stopwords')

    # Sample data
    documents = [
        "Cats are small, usually furry, carnivorous mammals.",
        "Dogs are domesticated mammals, not natural wild animals.",
        "The domestic dog has been selectively bred for millennia for various behaviors, sensory capabilities, and physical attributes.",
        "The cat is similar in anatomy to the other felids, with strong, flexible bodies, quick reflexes, sharp retractable claws, and teeth adapted to killing small prey.",
        "Dogs have been bred by humans for a long time.",
        "Cats are known for their agility and stealth."
    ]

    # Initialize and run the topic modeling
    topic_modeling = TopicModeling(documents, num_topics=2, passes=10)
    topic_modeling.prepare_corpus()
    topic_modeling.build_lda_model()
    topic_modeling.print_topics(num_words=4)

if __name__ == "__main__":
    main()
