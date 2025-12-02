# Save this as: rag_one_pdf.py
import fitz                  # pip install pymupdf
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers
import faiss
import numpy as np
import ollama                # you already have Ollama installed
 
# ========================= CONFIG =========================
PDF_PATH = "ethereum_whitepaper.pdf"          # ← CHANGE THIS to your PDF name
OLLAMA_MODEL = "llama3.2:3b"      # or "qwen2.5:7b" if you pulled it
# ========================================================
 
# 1. Extract all text from the single PDF
print("Reading your PDF...")
doc = fitz.open(PDF_PATH)
text = ""
for page in doc:
    text += page.get_text()
print(f"PDF has {len(text):,} characters")
 
# 2. Split into small chunks (better retrieval)
def chunk_text(text, size=600):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size//6):          # ~600 chars per chunk
        chunk = " ".join(words[i:i+size//6])
        chunks.append(chunk)
    return chunks
 
chunks = chunk_text(text)
print(f"Split into {len(chunks)} chunks")
 
# 3. Create embeddings + FAISS index (happens instantly)
print("Creating vector database...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
 
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("Ready!")
 
# 4. Ask questions forever
def ask(question):
    q_vec = model.encode([question])
    D, I = index.search(q_vec, k=4)          # top 4 relevant chunks
    context = "\n\n".join([chunks[i] for i in I[0]])
    prompt = f"""Use ONLY this context to answer the question:
 
{context}
 
Question: {question}
Answer:"""
 
    result = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return result['response']
 
# ========================= LOOP =========================
print("\nAsk anything about your PDF (type 'quit' to stop)\n")
while True:
    q = input("You: ").strip()
    if q.lower() in ["quit", "exit", "bye"]:
        break
    if q == "":
        continue
    print("Answer:", ask(q), "\n")