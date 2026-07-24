from sentence_transformers import SentenceTransformer


class BookEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def encode_books_descriptions(self, texts):
        return self.model.encode(texts,  show_progress_bar=True)
    def encode_query(self, query):
        return self.model.encode([query])