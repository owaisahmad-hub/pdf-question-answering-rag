import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("docs")

if __name__ == "__main__":
    chunks = load_chunks()
    batch = 200
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [c["text"] for c in part]
        col.upsert(
            ids=[str(i + j) for j in range(len(part))],
            documents=texts,
            embeddings=model.encode(texts).tolist(),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )
        print("Done", i + len(part))

    q = "What is the fiscal deficit target?"
    res = col.query(query_embeddings=model.encode([q]).tolist(), n_results=3)
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(meta, doc[:200], "\n")