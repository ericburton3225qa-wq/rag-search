from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
 
docs = [
    "Transformers use attention to understand context.",
    "RAG combines search and generation.",
    "BERT is an encoder-only transformer.",
    "Blockchain is a decentralized digital database or ledger that securely stores records across a network of computers in a way that is transparent, immutable, and resistant to tampering",
    "Each 'block' contains data, and the blocks are linked in a chronological 'chain.'"
]
 
embedder = SentenceTransformer('all-MiniLM-L6-v2')
vectors = embedder.encode(docs)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
 
query = "What is BERT?"
q_vec = embedder.encode([query])
 
D, I = index.search(q_vec, 1)
print("Most relevant:", docs[I[0][0]])