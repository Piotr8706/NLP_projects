import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

"""
This program summarizes text using TextRank Algorithm
"""

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    sent1 = [w.lower() for w in word_tokenize(sent1)]
    sent2 = [w.lower() for w in word_tokenize(sent2)]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

def generate_summary(text, num_sentences=5):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    ranked_sentence_indexes = sorted(((scores[i], i) for i, s in enumerate(sentences)), reverse=True)
    selected_sentences = sorted(ranked_sentence_indexes[:num_sentences], key=lambda x: x[1])
    
    summary = " ".join([sentences[idx] for (score, idx) in selected_sentences])
    return summary

def main():
    df = pd.read_csv('C:\Users\piotr\OneDrive\Pulpit\Kaggle\jigsaw-unintended-bias-in-toxicity-classification\all_data.csv')

if __name__=='__main__':
    main()