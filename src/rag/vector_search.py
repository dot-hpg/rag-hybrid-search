import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.rag.embed_index import COLLECTION, MODEL_NAME, QDRANT_PATH

# bge models work best when queries (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorSearcher:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient(path=QDRANT_PATH)

    def search(self, query: str, k: int = 5):
        vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        result = self.client.query_points(
            COLLECTION, query=vec.tolist(), limit=k, with_payload=True
        )
        return [(p.score, p.payload) for p in result.points]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How does LoRA reduce the number of trainable parameters?"
    searcher = VectorSearcher()
    for score, payload in searcher.search(query):
        print(f"{score:.3f} | {payload['title']} (p.{payload['page']})")
        print("   ", payload["text"][:160], "...\n")
