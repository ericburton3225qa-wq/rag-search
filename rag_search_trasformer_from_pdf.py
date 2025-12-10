
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import fitz  # pip install pymupdf
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
import faiss
import numpy as np

# ========================= CONFIG =========================
PDF_PATH = "ethereum_whitepaper.pdf"  # ← CHANGE THIS to your PDF name
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"  # Your model choice
# =========================================================

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
transformer_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # uses GPU if available, CPU if not
    load_in_4bit=True  # fits perfectly in 8 GB RAM
)

# Create a pipeline for text generation
chat = pipeline("text-generation", model=transformer_model, tokenizer=tokenizer)

# 1. Extract all text from the PDF
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
    for i in range(0, len(words), size // 6):  # ~600 chars per chunk
        chunk = " ".join(words[i:i + size // 6])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)
print(f"Split into {len(chunks)} chunks")

# 3. Create embeddings + FAISS index
print("Creating vector database...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("Ready!")

# 4. Ask questions forever
def ask(question):
    q_vec = model.encode([question])
    D, I = index.search(q_vec, k=4)  # top 4 relevant chunks
    context = "\n\n".join([chunks[i] for i in I[0]])
    prompt = f"""Use ONLY this context to answer the question:

{context}

Question: {question}
Answer:"""

    # Generate the answer using the model with your preferred format
    response = chat([{"role": "user", "content": prompt}], 
                    max_new_tokens=500, 
                    temperature=0.7, 
                    do_sample=True)[0]["generated_text"][-1]["content"]
    return response

# ========================= LOOP =========================
print("\nAsk anything about your PDF (type 'quit' to stop)\n")
while True:
    q = input("You: ").strip()
    if q.lower() in ["quit", "exit", "bye"]:
        break
    if q == "":
        continue
    print("Answer:", ask(q), "\n")
