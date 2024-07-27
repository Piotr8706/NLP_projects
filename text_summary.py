import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

"""
This program summarizes text using TextRank Algorithm
"""

def wrap():
    """
    
    """
    pass

def summary():
    """
    Summarization of the text
    """
    pass

def main():
    df = pd.read_csv('C:\Users\piotr\OneDrive\Pulpit\Kaggle\jigsaw-unintended-bias-in-toxicity-classification\all_data.csv')

if __name__=='__main__':
    main()