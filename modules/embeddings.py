import sentence_transformers as st
from sklearn.metrics.pairwise import cosine_similarity


def get_model(model_name: str):
    return st.SentenceTransformer(model_name)


def embed_text(model, text_list: list):
    return model.encode(text_list, normalize_embeddings=True)


def compare_similarity(model, embed1, embed2):
    return model.similarity(embed1, embed2)


def calculate_cosine_similarity_by_matrix(matrix):
    return cosine_similarity(matrix)


def calculate_cosine_similarity(vector_a, vector_b):
    return cosine_similarity([vector_a], [vector_b])[0][0]
