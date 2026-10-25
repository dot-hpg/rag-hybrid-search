import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
QDRANT_PATH = "data/qdrant"
COLLECTION = "papers"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH = 64


def load_chunks():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(path=QDRANT_PATH)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    for start in tqdm(range(0, len(chunks), BATCH), desc="Embedding"):
        batch = chunks[start : start + BATCH]
        vectors = model.encode(
            [c["text"] for c in batch],
            normalize_embeddings=True,
            batch_size=BATCH,
        )
        points = [
            PointStruct(id=start + i, vector=vec.tolist(), payload=chunk)
            for i, (chunk, vec) in enumerate(zip(batch, vectors))
        ]
        client.upsert(COLLECTION, points=points)

    print("Indexed:", client.count(COLLECTION).count, "points")
    client.close()


if __name__ == "__main__":
    main()
