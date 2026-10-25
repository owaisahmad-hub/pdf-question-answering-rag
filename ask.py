import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
llm = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask(question, k=4):
    res = col.query(query_embeddings=model.encode([question]).tolist(), n_results=k)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(
        f"[{m['source']}, page {m['page']}] {d}" for d, m in zip(docs, metas)
    )
    prompt = (
        "Answer ONLY using the context below. Cite the source file and page "
        "for every fact. If the answer is not in the context, say "
        "'I could not find this in the documents.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    out = llm.chat.completions.create(
       model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    sources = [{"source": m["source"], "page": m["page"]} for m in metas]
    return out.choices[0].message.content, sources

if __name__ == "__main__":
    answer, sources = ask("What is the fiscal deficit target for 2026-27?")
    print(answer)
    print(sources)