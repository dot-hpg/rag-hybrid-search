.PHONY: install install-cpu data index eval test lint format serve docker

install:          ## dev install with models (uses your existing torch, or installs the default wheel)
	pip install -e ".[ml,dev]"

install-cpu:      ## same, but with the small CPU-only torch wheel
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[ml,dev]"

data:             ## download PDFs and rebuild data/processed/chunks.jsonl
	rag download
	rag ingest

index:            ## embed chunks into Qdrant
	rag index

eval:             ## benchmark all retrieval modes and refresh the README table
	rag eval --update-readme

test:
	pytest --cov

lint:
	ruff check .
	ruff format --check .
	mypy src

format:
	ruff check --fix .
	ruff format .

serve:
	rag serve --reload

docker:
	docker build -t rag-hybrid-search .
	docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY rag-hybrid-search
