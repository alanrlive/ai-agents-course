# ingest.py
import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "elevator_fs_kb"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 600
OVERLAP = 75

KB_FILES = [
    "Global_Meeting_Scheduling_Policy.txt",
    "Office_London.txt",
    "Office_New_York.txt",
    "Office_Singapore.txt",
    "Office_Dubai.txt",
    "Office_Sydney.txt",
    "Office_Warsaw.txt",
]


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - OVERLAP
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    base_dir = Path(__file__).parent

    print(f"Initialising ChromaDB at ./{CHROMA_PATH}/")
    chroma_client = chromadb.PersistentClient(path=str(base_dir / CHROMA_PATH))

    # Delete existing collection so re-runs start clean
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Existing collection '{COLLECTION_NAME}' cleared before re-ingestion.")
    except Exception:
        pass  # Collection didn't exist yet

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' created (distance: cosine).\n")

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    total_docs = 0
    total_chunks = 0

    for filename in KB_FILES:
        filepath = base_dir / filename
        print(f"[Loading]   {filename}")
        text = filepath.read_text(encoding="utf-8")
        total_docs += 1

        stem = Path(filename).stem  # e.g. "Office_London"
        chunks = chunk_text(text)
        print(f"[Chunking]  {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")
        total_chunks += len(chunks)

        print(f"[Embedding] calling {EMBEDDING_MODEL} for {len(chunks)} chunk(s)...")
        embeddings = embed_texts(openai_client, chunks)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            all_ids.append(f"{stem}_chunk_{i}")
            all_embeddings.append(embedding)
            all_documents.append(chunk)
            all_metadatas.append({"source": filename})

        print(f"[Done]      {filename} — {len(chunks)} chunks ready.\n")

    print(f"[Storing]   Writing {total_chunks} chunks to ChromaDB...")
    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    stored_count = collection.count()

    print(f"\n{'='*50}")
    print("INGESTION SUMMARY")
    print(f"{'='*50}")
    print(f"  Documents processed : {total_docs}")
    print(f"  Chunks created      : {total_chunks}")
    print(f"  Chunks stored       : {stored_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
