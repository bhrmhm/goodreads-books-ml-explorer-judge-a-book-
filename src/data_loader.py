import os

import pandas as pd
def load_books(path):
    df_books = pd.read_parquet(path)
    df = df_books.dropna(subset=["description"])  #normally in every parquet the null description are already dropped
    return df


def search_book_in_parquet(path, book_title):
    df_books = pd.read_parquet(path)
    df_books = df_books[df_books["title"].str.contains(book_title)]
    return df_books["title"]
