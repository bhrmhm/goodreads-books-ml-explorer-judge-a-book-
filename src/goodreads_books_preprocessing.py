import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd


GENRE_SHELVES = {
    'fiction', 'non-fiction', 'nonfiction', 'fantasy', 'science-fiction', 'sci-fi',
    'mystery', 'thriller', 'romance', 'horror', 'biography', 'history',
    'self-help', 'classics', 'young-adult', 'ya', 'children', 'poetry',
    'comics', 'graphic-novels', 'crime', 'adventure', 'philosophy', 'science'
}
def extract_genre(shelves):
    for shelf in shelves:  # already sorted by count descending
        if shelf['name'].lower() in GENRE_SHELVES:
            return shelf['name'].lower()
    return None

def load_data(file_name, head=5000):
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
print(books[0])
#list of all features for a book
books_features = df_books.columns.tolist()

#nulls
#print(df_books.isnull().sum()) no null
print((df_books['description'] == '').sum())

text_cols = ['description', 'title', 'authors']
'''for col in text_cols:
    print(f"{col}: {(df_books[col] == '').sum()} empty")'''

"""description: 104 empty -> might need to drop"""

print(df_books['popular_shelves'][0])
#add a top shelf as main category based on more popular shelf


df_books['genre'] = df_books['popular_shelves'].apply(extract_genre)

print(df_books['genre'].value_counts())
print(df_books['genre'].isnull().sum(), 'books with no matching genre')

"""110 books with no matching genre -> might need to drop"""
# drop empty descriptions and books with no genre
df_books = df_books[df_books['description'] != '']
df_books = df_books[df_books['genre'].notna()]
print(df_books.shape)

for col in num_features:
    df_books[col] = pd.to_numeric(df_books[col], errors='coerce')  # converts bad values to NaN
    df_books[col] = df_books[col].fillna(df_books[col].median()) #replace it with the median
#---cleaning done---


print(df_books['title'])
# all your cleaning code here
df_books.to_parquet(os.path.join(DIR, 'goodreads_cleaned_5000.parquet'))
