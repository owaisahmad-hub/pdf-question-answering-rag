import chromadb

col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
print("Total chunks:", col.count())
print(set(m["source"] for m in col.get()["metadatas"]))