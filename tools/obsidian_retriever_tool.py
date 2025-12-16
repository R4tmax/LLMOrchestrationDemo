from crewai.tools import BaseTool
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from typing import Type, Any
from pydantic import BaseModel, Field

from config import VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH, EMBEDDING_MODEL_NAME


class ObsidianSearchToolInput(BaseModel):
    query: str = Field(description="The search query to find relevant information in Obsidian notes.")


class ObsidianSearchTool(BaseTool):
    name: str = "Obsidian Vault Search"
    description: str = "Searches the Obsidian knowledge base for information relevant to the query. Returns top 3 relevant chunks."
    args_schema: Type[BaseModel] = ObsidianSearchToolInput
    index: Any = None
    docs: Any = None
    embedding_model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.index:
            self.index = faiss.read_index(VECTOR_STORE_INDEX_PATH)
        if not self.docs:
            with open(VECTOR_STORE_DOCS_PATH, 'rb') as f:
                self.docs = pickle.load(f)
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _run(self, query: str) -> str:
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype('float32'), k=3)  # Get top 3 results

        results = []
        for i in I[0]:
            if i != -1:  # FAISS can return -1 if less than k results found
                results.append(
                    f"Source: {self.docs[i]['metadata'].get('source', 'N/A')}\nContent: {self.docs[i]['text']}\n---")

        if not results:
            return "No relevant information found in Obsidian notes."
        return "\n".join(results)