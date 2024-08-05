import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample data
documents = [
    "Cats are small, usually furry, carnivorous mammals.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "The domestic dog has been selectively bred for millennia for various behaviors, sensory capabilities, and physical attributes.",
    "The cat is similar in anatomy to the other felids, with strong, flexible bodies, quick reflexes, sharp retractable claws, and teeth adapted to killing small prey.",
    "Dogs have been bred by humans for a long time.",
    "Cats are known for their agility and stealth."
]

# Preprocessing the text data
def preprocess(text):
    tokens = [word for word in text.lower().split() if word.isalpha() and word not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Build the LDA model
num_topics = 2
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)

# Print the topics
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)
