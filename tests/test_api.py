import json

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api.app import create_app


@pytest.fixture
def client(pipeline):
    with TestClient(create_app(pipeline)) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/api/info").json()
    assert info["chunks"] == 6 and info["papers"] == 6
    assert info["default_mode"] == "hybrid_rerank"
    assert info["generator"] == "extractive"


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200 and "Hybrid-Search RAG" in resp.text


def test_search(client):
    body = client.post(
        "/api/search", json={"query": "U-shaped curve middle of context", "mode": "hybrid", "k": 2}
    ).json()
    assert body["mode"] == "hybrid"
    assert body["hits"][0]["doc_id"] == "2307.03172"
    assert set(body["hits"][0]) >= {"chunk_id", "title", "page", "text", "score", "ranks"}
    assert body["total_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "k": 0},
        {"query": "x", "k": 99},
        {"query": "x", "mode": "nope"},
    ],
)
def test_search_validation(client, payload):
    assert client.post("/api/search", json=payload).status_code == 422


def test_compare_runs_every_mode(client):
    body = client.post("/api/compare", json={"query": "next sentence prediction", "k": 3}).json()
    assert set(body) == {"bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank"}


def test_ask(client):
    body = client.post("/api/ask", json={"query": "What does BERT mask?"}).json()
    assert body["model"] == "extractive"
    assert body["answer"] and body["citations"]


def test_ask_stream_events(client):
    with client.stream(
        "POST", "/api/ask/stream", json={"query": "What is tiling in FlashAttention?"}
    ) as resp:
        raw = "".join(resp.iter_text())
    events = [
        (
            block.split("\n")[0].removeprefix("event: "),
            json.loads(block.split("\n")[1].removeprefix("data: ")),
        )
        for block in raw.strip().split("\n\n")
    ]
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and "token" in names
    assert events[0][1]["hits"][0]["doc_id"] == "2205.14135"
