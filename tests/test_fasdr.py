import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock
from typing import List
import numpy as np
import spacy
from scipy.spatial import KDTree
import scipy

from fasdr import (
    build_targeted_index,
    Document,
    DocumentIndex,
    EmptyDocument,
    create_nlp_pipeline,
    create_kdtree,
    update_kdtree,
    search_index,
)

D_EMBED = 384

# Dummy sentence embeddings
sentences = ["This is a test.", "Another test sentence.", "The third test sentence."]
embeddings = np.random.rand(3, D_EMBED)

# Temporary directory for storing test files
temp_dir = tempfile.TemporaryDirectory()
doc_path = Path(temp_dir.name) / "test.txt"
doc_path.write_text("\n".join(sentences))


def create_sample_files(root_dir: Path, extensions: List[str]):
    for ext in extensions:
        subdir = root_dir / ext.strip(".")
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / f"sample{ext}").        write_text(f"This is a sample {ext} file.")


def test_create_kdtree():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    tree = create_kdtree(data)
    assert isinstance(tree, KDTree)

def test_update_kdtree():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    new_data = np.array([[7, 8]])
    tree = create_kdtree(data)
    updated_tree = update_kdtree(tree, new_data)
    assert isinstance(updated_tree, KDTree)
    assert updated_tree.data.shape == (4, 2)

def test_search_index():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    tree = create_kdtree(data)
    query = np.array([[1, 2]])
    k=1
    distances, indices = search_index(tree, query, k=k)
    assert distances.shape == (k,)
    assert indices.shape == (k,)

    assert np.array_equal(distances, np.array([0]))
    assert np.array_equal(indices, np.array([0]))

def test_create_nlp_pipeline():
    nlp = create_nlp_pipeline()
    assert isinstance(nlp, spacy.language.Language)
    assert nlp.has_pipe("sentencizer")
    assert nlp.has_pipe("sentence_bert")

def test_document():
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_path = Path(temp_dir) / "test.txt"
        doc_path.write_text("\n".join(sentences))
        doc = Document(doc_path)
        assert len(doc.sentences) == 3
        query = np.random.rand(1, D_EMBED)
        results = doc.search_sentences(query, k=2)
        assert len(results) == 2

def test_document_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_path = Path(temp_dir) / "test.txt"
        doc_path.write_text("\n".join(sentences))
        index = DocumentIndex(temp_dir)
        assert len(index.documents) == 1
        query = "test query"
        doc_results = index.search_documents(query, k=1)
        assert len(doc_results) == 1
        sentence_results = index.search_sentences(query, k=1)
        assert len(sentence_results) == 1
        assert len(sentence_results[0]) == 3

def test_document_empty_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "empty.txt"
        temp_file.write_text("")

        with pytest.raises(EmptyDocument):
            Document(temp_file)

def test_document_simple_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "simple.txt"
        temp_file.write_text("This is a test.")

        doc = Document(temp_file)
        assert len(doc.sentences) == 1
        assert doc.sentences[0] == "This is a test."
        assert doc.sentence_embeddings is not None

def test_document_save_and_load_embeddings():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "simple.txt"
        temp_file.write_text("This is a test.")

        doc = Document(temp_file)
        doc.save_embeddings()

        loaded_doc = Document(temp_file)
        loaded_doc.load_embeddings(temp_file.with_name(f"{temp_file.stem}_embeddings.pkl"))

        assert loaded_doc.sentences == doc.sentences
        assert np.array_equal(loaded_doc.sentence_embeddings, doc.sentence_embeddings)

def test_document_search_sentences():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "simple.txt"
        temp_file.write_text("This is a test.")

        doc = Document(temp_file)
        nlp = create_nlp_pipeline()
        query_embedding = nlp("This is a query.").vector

        results = doc.search_sentences(query_embedding)
        assert len(results) == 1
        assert results[0][1] == "This is a test."

def test_document_index_empty_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(EmptyDocument):
            DocumentIndex(Path(temp_dir), extensions=[".unknown"])

def test_document_index_simple_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])

        index = DocumentIndex(temp_dir)
        assert len(index.documents) == 2

def test_document_index_ignored_patterns():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])
        ignored_subdir = temp_dir / ".ignored"
        ignored_subdir.mkdir(parents=True, exist_ok=True)
        (ignored_subdir / "sample.txt").write_text("This is a sample txt file in the ignored folder.")

        index = DocumentIndex(temp_dir)
        assert len(index.documents) == 2

def test_document_index_save_and_load():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])

        index = DocumentIndex(temp_dir)
        index.save()

        loaded_index = DocumentIndex(temp_dir)
        loaded_index.load()

        assert loaded_index.model_name == index.model_name
        assert len(loaded_index.documents) == len(index.documents)
        assert loaded_index.summary_index is not None

def test_document_index_search_documents():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])

        index = DocumentIndex(temp_dir)
        nlp = create_nlp_pipeline()
        query_embedding = nlp("This is a query.").vector

        results = index.search_documents(query_embedding=query_embedding)
        assert len(results) == 1
        assert isinstance(results[0][1], Path)

def test_document_index_search_sentences():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])

        index = DocumentIndex(temp_dir)
        nlp = create_nlp_pipeline()
        query_embedding = nlp("This is a query.").vector

        results = index.search_sentences("This is a query.")
        assert len(results) > 0
        assert isinstance(results[0], dict)

def test_document_index_search_sentences_targeted():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])

        index = DocumentIndex(temp_dir)
        nlp = create_nlp_pipeline()
        query_embedding = nlp("This is a query.").vector

        results = index.search_sentences_targeted("This is a query.", n_docs=1, n_sents=1)
        assert len(results) > 0
        assert isinstance(results[0], dict)

def test_build_targeted_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_sample_files(temp_dir, [".txt", ".md"])

        index = DocumentIndex(temp_dir)
        docs = list(index.documents.values())

        targeted_index = build_targeted_index(docs)
        assert isinstance(targeted_index, dict)
        assert len(targeted_index["sentences"]) > 0
        assert isinstance(targeted_index["index"], KDTree)
