"""This script is for preprocessing, cleaning and analysing the dataset"""
import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
import Globals

MIN_SAMPLES = 90
MAX_PER_CLASS = 700

def extract_genre(shelves):
    for shelf in shelves:  # already sorted by count descending
        name = shelf['name'].lower()
        #if shelf['name'].lower() in Globals.GENRE_SHELVES:
            #return shelf['name'].lower()
        if name in Globals.GENRE_MAPPING:
            return Globals.GENRE_MAPPING[name]
    return None

def load_data(file_name, head=7000):
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)

            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data
DIR = '/Users/bahar/Documents/Projects/Books_geek/resources'

books = load_data(os.path.join(DIR, 'goodreads_books.json.gz'))

#make a dataframe
df_books = pd.DataFrame(books)

text_features = ['title', 'authors', 'genre', 'description', 'language_code']
num_features = ['ratings_count', 'text_reviews_count','num_pages', 'publication_year', 'average_rating']

#some info
#print(df_books.columns.tolist())
print(df_books.shape)
#print(df_books.head())
#print(books[0])

#list of all features for a book
books_features = df_books.columns.tolist()

#nulls
#print(df_books.isnull().sum()) no null
#print((df_books['description'] == '').sum())

text_cols = ['description', 'title', 'authors']
'''for col in text_cols:
    print(f"{col}: {(df_books[col] == '').sum()} empty")'''

"""description: 104 empty -> might need to drop"""

print(df_books['popular_shelves'][0])

#add a top shelf as main category based on more popular shelf
all_shelves = Counter()
for shelves in df_books['popular_shelves'][:100]:
    for shelf in shelves:
        all_shelves[shelf['name'].lower()] += int(shelf['count'])


# see most common shelves
'''for shelf, count in all_shelves.most_common(40):
    print(f"{shelf}: {count}")'''

#extract genres from common shelves
df_books['genre'] = df_books['popular_shelves'].apply(extract_genre)

print(df_books['genre'].value_counts())
#print(df_books['genre'].unique())
print(df_books['genre'].isnull().sum(), 'books with no matching genre')

# drop empty descriptions and books with no genre
df_books = df_books[df_books['description'] != '']
df_books = df_books[df_books['genre'].notna()]
print(df_books.shape)

#replace bad values with median
for col in num_features:
    df_books[col] = pd.to_numeric(df_books[col], errors='coerce')  # converts bad values to NaN
    df_books[col] = df_books[col].fillna(df_books[col].median()) #replace it with the median


# drop small classes first
valid_genres = df_books['genre'].value_counts()
valid_genres = valid_genres[valid_genres >= MIN_SAMPLES].index
df_books = df_books[df_books['genre'].isin(valid_genres)]

# then cut large classes
df_books = df_books.groupby('genre').apply(
    lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=42)
).reset_index(drop=True)

print(df_books['genre'].value_counts())
#---cleaning done---


#print(df_books['title'])
#print(df_books['authors']) #Only has author's ID

# save cleaned data as a parquet
df_books.to_parquet(os.path.join(DIR, 'goodreads_cleaned_7000.parquet'))





"""Genres:
['biography' 'fiction' 'fantasy' 'nonfiction' 'history' 'romance'
 'philosophy' 'mystery' None 'non-fiction' 'young-adult' 'children'
 'horror' 'poetry' 'crime' 'classics' 'graphic-novels' 'science-fiction'
 'self-help' 'sci-fi' 'thriller' 'comics' 'science' 'ya' 'adventure']"""