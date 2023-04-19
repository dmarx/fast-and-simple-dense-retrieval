import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import scipy

from fasdr import (
    #create_kdtree_index,
    create_kdtree,
    search_index,
    create_nlp_pipeline,
    Document,
    DocumentIndex,
)

# Dummy sentence embeddings
sentences = ["This is a test.", "Another test sentence.", "The third test sentence."]
embeddings = np.random.rand(3, 768)

# Temporary directory for storing test files
temp_dir = TemporaryDirectory()
doc_path = Path(temp_dir.name) / "test.txt"
doc_path.write_text("\n".join(sentences))

def test_create_kdtree_index():
    index = create_kdtree(embeddings)
    assert isinstance(index, scipy.spatial.KDTree)

def test_search_index():
    index = create_kdtree(embeddings)
    query = np.random.rand(1, 768)
    distances, indices = search_index(index, query, k=2)
    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)

def test_create_nlp_pipeline():
    nlp = create_nlp_pipeline()
    assert "sentencizer" in nlp.pipe_names
    assert "sentence_bert" in nlp.pipe_names

def test_document():
    doc = Document(doc_path)
    assert len(doc.sentences) == 3
    query = np.random.rand(1, 768)
    results = doc.search_sentences(query, k=2)
    assert len(results) == 2

def test_document_index():
    index = DocumentIndex(temp_dir.name)
    assert len(index.documents) == 1
    query = "test query"
    doc_results = index.search_documents(query, k=1)
    assert len(doc_results) == 1
    sentence_results = index.search_sentences(query, k=1)
    assert len(sentence_results) == 1
    assert len(sentence_results[0]) == 3
    assert len(sentence_results[0][-1]) == 1

temp_dir.cleanup()
