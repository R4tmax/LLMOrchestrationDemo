import faiss
import pickle
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import streamlit as st

from config import VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH


def render_vector_space():
    try:
        # 1. Load the Math (FAISS) and Text (Pickle)
        index = faiss.read_index(VECTOR_STORE_INDEX_PATH)
        with open(VECTOR_STORE_DOCS_PATH, 'rb') as f:
            chunks = pickle.load(f)

        # 2. Extract the raw embeddings from FAISS
        # FAISS stores vectors in a flat array, we reconstruct it
        num_vectors = index.ntotal
        if num_vectors == 0:
            st.warning("Your vector store is empty.")
            return

        # Reconstruct vectors (this works for IndexFlatL2)
        embeddings = np.array([index.reconstruct(i) for i in range(num_vectors)])

        # 3. Squash dimensions from 384D to 3D using PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)

        # 4. Prepare data for Plotly
        # We only take the first 50 characters of text for the hover label
        hover_texts = [f"Source: {c['metadata']['source']}<br>{c['text'][:50]}..." for c in chunks]
        sources = [c['metadata']['source'] for c in chunks]

        # 5. Create the Interactive 3D Scatter Plot
        fig = px.scatter_3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            color=sources,  # Color dots by their source file
            hover_name=hover_texts,
            title="Obsidian Vault Semantic Map",
            opacity=0.7
        )

        # Hide the messy axis numbers
        fig.update_layout(scene=dict(xaxis=dict(showticklabels=False),
                                     yaxis=dict(showticklabels=False),
                                     zaxis=dict(showticklabels=False)))

        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load vector visualization: {e}")