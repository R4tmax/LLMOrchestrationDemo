"""
This script is responsible for vectorizing your Obsidian vault into form processable by RAG system.

If you are completely unfamiliar with text processing techniques, It might be hard to grasp the flow of the code and I would recommend some quick reading first.
Personally I am big fan of IBM and Google Cloud docs for quick top level grasp, e.g. https://www.ibm.com/think/topics/retrieval-augmented-generation.

Understanding what a "vector" is in this context is a slightly different beast but a good summary can be discerned from Google Docs
(https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview), alternatively, my elevator pitch explanation:

Instead of trying to "learn" context of all text inputs, you create an Index (a lookup structure, very similar to how book indexes work in nature),
which creates a datatype which can be queried for reference, in process of doing so we store which document/file sourced which pieces of text, think of this as citing
the information.

You then parse all the different documents into smaller subsections -> chunks.

These chunks are then "vectorized" that is, converted to some numerical representation - same way how word2vec works for LLMs - in a
multidimensional space. Each "chunk" then is as such associated with some direction represented by a matrix, chunks (pieces of text) talking about the same thing
will be "pointing in similar direction" and similarity between the vectors (e.g. cosine similarity) is used as decision boundary for which chunks should be passed to LLM as context
to shape its answer.
"""

from llama_index.readers.obsidian import ObsidianReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from config import OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH


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


if __name__ == "__main__":
    print("Starting Knowledge Base generation...")

    # Run the function
    try:
        setup_knowledge_base(OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH)
        print("SUCCESS: Index and Docs created successfully.")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()