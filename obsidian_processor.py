from llama_index.readers.obsidian import ObsidianReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def setup_knowledge_base(vault_path, index_path, docs_path):
    # 1. Load Notes
    reader = ObsidianReader(input_dir=vault_path)
    documents = reader.load_data() # This returns LlamaIndex Documents
    # Convert LlamaIndex Documents to Langchain Documents or extract text
    texts_with_metadata = [{"text": doc.get_content(), "metadata": {"source": doc.metadata.get("file_name", "Unknown")}} for doc in documents]

    # 2. Chunk Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for item in texts_with_metadata:
        doc_chunks = text_splitter.split_text(item["text"])
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({"text": chunk_text, "metadata": {**item["metadata"], "chunk_id": i}})

    # 3. Embed and Store
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_texts_only = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(chunk_texts_only, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    faiss.write_index(index, index_path)
    with open(docs_path, 'wb') as f:
        pickle.dump(chunks, f) # Save chunks with metadata corresponding to embeddings
    print(f"Knowledge base setup complete. Index saved to {index_path}, docs to {docs_path}")