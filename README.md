# FASDR: Fast and Simple Dense Retrieval

🚧 WORK IN PROGRESS 🚧

FASDR is a simple and lightweight library for fast and efficient document retrieval. It is designed to be easy to setup and use, built on top of popular, trusted components (`scipy`, `spacy`, `transformers`) to ensure it can be seamlessly integrated into existing projects and "just work". It's especially well suited for small-to-medium corpora, such as retrieval-augmented prompting of FOSS documentation.

### Features

* Fast and efficient dense retrieval using KDTree data structures.
* Simple interface for indexing and searching documents and sentences.
* Support for various file formats and customizable indexing options.
* Integration with the SpaCy and Sentence-BERT libraries for natural language processing and sentence embeddings.

## Installation

First, install `fasdr` via pip:

```
pip install fasdr
```

Next, download sentence tokenization language model for
spacy:

<!--To do: language agnostic sentencizer? language detection?-->
```
python -m spacy download en_core_web_trf
```

## Quick Start

### Indexing documents

### Quick Start

To get started with FASDR, you can create a `DocumentIndex` object by passing in the root directory containing the documents you want to index:

```python
from fasdr import DocumentIndex

index = DocumentIndex("/path/to/documents")
```

Once you have created the DocumentIndex object, you can search for documents or sentences using the search_documents and search_sentences methods:

```python
# Find the top five documents relevant to the query "climate change"
results = index.search_documents("climate change", k=5)

# Find the top 10 sentences after filtering on the top 5 documents
results = index.search_sentences_targeted("climate change", n_docs=5, n_sents=10)
```

You can customize the behavior of the DocumentIndex object by specifying options such as the model name and the file extensions to include in the index:

```python
index = DocumentIndex(
    "/path/to/documents",
    model_name="all-MiniLM-L6-v2",
    extensions=[".txt", ".md", ".pdf"]
)
```

## Design

FASDR is designed to be fast and simple, with a focus on ease of use and minimal setup. It uses FAISS for similarity search, which is a highly optimized library for dense vector search, and SpaCy with the Sentence-BERT component for embedding text. The library is built around two main classes:

* `Document`: Represents a single document and its embeddings.  
* `DocumentIndex`: Represents an index of documents and their embeddings.  

`Document` objects are created by passing in the path to the document file, and can be used to search for similar sentences within the document. `DocumentIndex` objects are created by passing in the root directory containing the documents to index, and can be used to search for similar documents or sentences across all the indexed documents.