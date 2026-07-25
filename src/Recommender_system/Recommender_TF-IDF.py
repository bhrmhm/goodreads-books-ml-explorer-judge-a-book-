#1
import os
import nltk
import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_books

#https://github.com/anuragjain-git/text-classification
#https://www.kaggle.com/code/uthamkanth/beginner-tf-idf-and-cosine-similarity-from-scratch

query = "A mystery murder in an inn/hotel" #Users input


DIR = '/Users/bahar/Documents/Projects/Books_geek/resources'
df_books = load_books(os.path.join(DIR, 'goodreads_cleaned_7000.parquet'))
descriptions = df_books['description'] #all descriptions
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000
)
X = vectorizer.fit_transform(descriptions)
vectorizer.get_feature_names_out()


def get_recommendations(query, X):
    query_vector = vectorizer.transform([query])

    sim_scores = cosine_similarity(query_vector, X)

    #sort and take top similar books
    top_indices = np.argsort(sim_scores[0])[::-1][:10] #Keepy top 5 similar books

    recommendations = df_books.iloc[top_indices][["title", "description","genre"]]

    return recommendations


#print(get_recommendations("a romance before dying/dead/death", X))

recoms = get_recommendations("Percy jackson", X)
recoms.to_csv("../../results/recommendations_TFIDF.csv", index=False)