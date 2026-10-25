import os
from pypdf import PdfReader

def load_chunks(folder="data", size=800, overlap=100):
    chunks = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(folder, name))
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\n", " ").strip()
            start = 0
            while start < len(text):
                piece = text[start:start + size]
                if len(piece) > 100:
                    chunks.append({"text": piece, "source": name, "page": page_no})
                start += size - overlap
    return chunks

if __name__ == "__main__":
    chunks = load_chunks()
    print("Total chunks:", len(chunks))
    print(chunks[0])