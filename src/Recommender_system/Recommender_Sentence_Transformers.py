#2
import os

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.Recommender_system.embedder import BookEmbedder
from src.data_loader import load_books
from src.retriever import BookRetriever

DIR = '/Users/bahar/Documents/Projects/Books_geek/resources'
#df_books = load_books(os.path.join(DIR, 'goodreads_cleaned_7000.parquet'))

# Load a pretrained Sentence Transformer model
#model = SentenceTransformer("all-mpnet-base-v2")


# Calculate embeddings by calling model.encode()
#embeddings = model.encode(descriptions, show_progress_bar=True)

def get_recommendations(query, path):
    #Load books
    df_books = load_books(path)

    # Load a pretrained Sentence Transformer model
    bookEmbedder = BookEmbedder()
    #Get all descriptions
    descriptions = df_books['description'].tolist()  # all descriptions

    # Calculate embeddings
    dscr_embeddings = bookEmbedder.encode_books_descriptions(descriptions)

    query_embedding = bookEmbedder.encode_query(query)

    bookRetriever = BookRetriever(dscr_embeddings, df_books)

    recommendation = bookRetriever.search(query_embedding)

    return recommendation

#print(get_recommendations("A young wizard discovers he has magical powers and goes to a special school of wizardly", embeddings))

recoms = get_recommendations("They meet just before Christmas", os.path.join(DIR, 'goodreads_cleaned_7000.parquet'))
recoms.to_csv("../../results/recommendations_ST.csv", index=False)




#TODO: see last prompt of chatgpt about making a complete pipeline
#Todo: when making parquet, do something to take better books like random choice