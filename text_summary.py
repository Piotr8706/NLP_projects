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
    #df = pd.read_csv('C:\Users\piotr\OneDrive\Pulpit\Kaggle\jigsaw-unintended-bias-in-toxicity-classification\all_data.csv')
    text = """Many of us, which are working in tech industry, are asking the same question, back in 2023 and now in 2024. When the tech jobs are coming back and to what extent?

    Long story short, the most simple and probably accurate answer is, when interest rates drop and money gets cheap again. But, there are couple of additional arguments that we need to take into account, that were not present before, like in previous crises that happened in 2001 and 2008.

    Beggining of the year 2024, does not have good news, as we all saw and is continues the same trend as past year 2023. We all testify, that getting job in 2024, especially junor roles is much worse and harder as it was back in 2023.

    When we ask ourselves the question in title, we mostly think that tech jobs refer to software development jobs in general, but that is true to some extent. Things have changed in past year. Technology is not only about software development, but it is generally about process of things getting better. Like things getting cheaper or new innovations kick in, like in hardware world.

    So you might ask yourself, what are those things, that have changed tech jobs industry, beside cheap money and high interest rates? Let me ask you a question, what technology have skyrocketed in past year or year and half? Yes you know the answer, it is AI.

    So, the classical or traditional way of investments in software development, may not play huge role to improve technology as we used to see in past two decades. Like me or anybody else, huge excitement and expectation comes from AI, either from software development side or new AI no code tools, robotics etc.

    When we talk about software development jobs, back in 1990s, most of web development jobs considered writing HTML and that was the whole career of web developer before. Manual QA was also impacted with automation tools, Server management for IT where we managed individual servers, now is swapped out with cloud, and what was manual work is swapped out with of the shelf stuff offered by cloud providers.

    When we talk about low code or nocode tools, they are now supercharged with AI tools or extensions like FlutterFlow, Supabase AI, GitHub Copilot etc., that are lowering the amount of work that needs to be done from our developer side. When we look at this from other side, this means that tech skills needed to be productive are shrinking. This probably means that this shrinking will also put price pressure on salaries that we used to see before in tech job market.

    So, what this all means for us as software developers and our future?

    First hings first, all this does not mean that all jobs are going away like you may read nowdays. It might, take years and decades to implement, improve and then deploy all those advancements in AI to reality, either software advancements or robotics.

    Second, this crisis in tech industry, layoffs and advancements in AI means we need to acquire new skillset for future tech jobs, that might not look the same way as we are used to. New skillset might be skillset in data science, robotics or new roles that AI might create in future. I know, that going from software development role in web or mobile to data science or robotics is not trivial, but it is what it is.

    So, the conclusion is, do not get depressed about current environment of market. We all are in the same boat. Thinking about our future and paying attention to the job market and checking what is going on is not a bad idea, just to get insight of what will be new trends, when this market will recover from current crisis, do we need and what are our transition plans. Hopefully we will be on track by the end of this year.

    If you liked this story, leave a clap, leave a comment and stay tuned for more."""
    source = "https://mehobega.medium.com/when-will-the-tech-jobs-come-back-a563d5ba45ba"
    result = generate_summary(text)
    print(result)
    # Save the summary to a text file
    with open("summary.txt", "w") as file:
        file.write(result)

if __name__=='__main__':
    main()