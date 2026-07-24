import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class BookRetriever:
    def __init__(self, book_embeddings, df_books):
        self.embeddings = book_embeddings
        self.df = df_books

    def search(self, query_embedding, top_k=10):
        cosine_scors = cosine_similarity(query_embedding, self.embeddings)[0] #a matrix of shape (1, nb_books), a score for each book -> that's why we take the one and only line with all scores
        top_idx = np.argsort(cosine_scors)[::-1][:top_k] #cause ascending order

        results = self.df.iloc[top_idx][["title", "description", "genre"]].copy()
        results["score"] = cosine_scors[top_idx]
        return results


