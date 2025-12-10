from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF


# -------------------------
# 1. Extract text from PDF
# -------------------------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")

    return text

pdf_path = "ethereum_whitepaper.pdf"
pdf_text = extract_pdf_text(pdf_path)

# -------------------------
# 2. Split into chunks
# -------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

chunks = chunk_text(pdf_text)

# -------------------------
# 3. Embed each chunk
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = model.encode(chunks)

# -------------------------
# 4. Build FAISS index
# -------------------------
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# -------------------------
# 5. Ask a question
# -------------------------
query = "What is Ethereum?"
q_vec = model.encode([query])

D, I = index.search(q_vec, 3)

# print("\nTop Results:\n")
# for idx in I[0]:
#     print(chunks[idx])
#     print("---------")

with open("search_results.txt", "w", encoding="utf-8") as f:
    f.write("Top Results:\n\n")
    for idx in I[0]:
        f.write(chunks[idx] + "\n")
        f.write("---------\n")

print("Results saved to search_results.txt")


