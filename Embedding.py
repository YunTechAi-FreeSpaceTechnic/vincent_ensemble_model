import os

# from functools import cache

# AI
from sentence_transformers import SentenceTransformer


def get_embedding_model(
    model_name: str = "BAAI/bge-m3", device: str = "cpu"
) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def main():
    model = get_embedding_model()
    texts = ["我喜歡吃蛋餅", "有些人喜歡吃蛋餅", "個人喜好", "眾人喜好"]
    embd = model.encode(texts, show_progress_bar=True)
    print(embd.shape)
    return
    for ft, fe in zip(texts, embd):
        for tt, te in zip(texts, embd):
            s = model.similarity(fe, te)
            print(ft, "->", tt, "相似度:", s)


if __name__ == "__main__":
    main()
