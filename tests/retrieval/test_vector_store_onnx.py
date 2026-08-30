import os
import pytest
import numpy as np
import faiss
from backend.app.retrieval.vector_store import vector_store
from backend.app.retrieval.graph_retrieve import graph_retrieve
from backend.app.retrieval.hybrid_retrieve import hybrid_retrieve

@pytest.fixture(autouse=True)
def restore_production_vector_store():
    orig_dir = "./data/vector_index"
    vector_store.index_dir = orig_dir
    vector_store.index_file = os.path.join(orig_dir, "index.faiss")
    vector_store.metadata_file = os.path.join(orig_dir, "chunks_metadata.json")
    vector_store._initialized = False
    yield

def test_onnx_vector_store_properties():
    # 1. Initialize VectorStore (which will load fastembed if installed)
    vector_store.initialize()
    
    # 2. Check that the dimension is 384
    assert vector_store.dimension == 384
    
    # 3. Check model outputs
    if getattr(vector_store, "is_onnx", False):
        emb = list(vector_store.model.embed(["Test query"]))[0]
        assert len(emb) == 384
        assert isinstance(emb, np.ndarray) or isinstance(emb, list)
    else:
        emb = vector_store.model.encode(["Test query"])[0]
        assert len(emb) == 384
        assert isinstance(emb, np.ndarray)

def test_load_existing_faiss_index():
    # Verify the existing FAISS index on disk is loaded correctly
    vs = vector_store
    vs.initialize()
    assert vs.index is not None
    assert vs.index.ntotal > 0
    assert len(vs.metadata) == vs.index.ntotal

def test_representative_query_agreement():
    vs = vector_store
    vs.initialize()
    
    query = "What is the volume of distribution of Naproxen?"
    
    # Execute query search
    results = vs.search(query, top_k=5)
    
    assert len(results) > 0
    # The top retrieved chunk should be from the Naproxen document
    top_chunk = results[0]
    assert top_chunk.type == "chunk"
    assert "Naproxen" in top_chunk.content or "naproxen" in top_chunk.content.lower()

def test_other_retrieval_imports():
    # Ensure imports and basics of GraphRAG and Hybrid retrieval pipelines still function
    assert graph_retrieve is not None
    assert hybrid_retrieve is not None
