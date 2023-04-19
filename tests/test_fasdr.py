import numpy as np
import pytest
from fasdr import FASDR

@pytest.fixture
def sample_embeddings():
    return [
        (np.random.rand(300), "This is a sample sentence."),
        (np.random.rand(300), "Another example sentence."),
        (np.random.rand(300), "Yet another sentence."),
    ]

def test_add_item(sample_embeddings):
    fasdr = FASDR(dimensions=300)
    for embedding, text in sample_embeddings:
        fasdr.add_item(embedding, text)
    assert len(fasdr._items) == len(sample_embeddings)

def test_build(sample_embeddings):
    fasdr = FASDR(dimensions=300)
    for embedding, text in sample_embeddings:
        fasdr.add_item(embedding, text)
    fasdr.build()
    assert fasdr._index is not None

def test_query(sample_embeddings):
    fasdr = FASDR(dimensions=300)
    for embedding, text in sample_embeddings:
        fasdr.add_item(embedding, text)
    fasdr.build()

    query_embedding = np.random.rand(300)
    k = 2
    results = fasdr.query(query_embedding, k)
    assert len(results) == k

def test_save_and_load(tmpdir, sample_embeddings):
    fasdr = FASDR(dimensions=300)
    for embedding, text in sample_embeddings:
        fasdr.add_item(embedding, text)
    fasdr.build()

    file_path = tmpdir.join("fasdr_index.bin")
    fasdr.save(file_path)

    loaded_fasdr = FASDR(dimensions=300)
    loaded_fasdr.load(file_path)

    query_embedding = np.random.rand(300)
    k = 2
    original_results = fasdr.query(query_embedding, k)
    loaded_results = loaded_fasdr.query(query_embedding, k)
    assert original_results == loaded_results
