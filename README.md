# FASDR: Fast and Simple Dense Retrieval

FASDR is a lightweight library for fast and efficient retrieval of using sentence embeddings. It is designed to be easy to use and integrate into your projects, especially applications such as document search and information retrieval on small corpora, such as retrieval-augmented prompting of FOSS documentation. FASDR is built on top of spaCy, Sentence-Transformers, and Scipy.

### Features

* Fast and efficient dense retrieval using KDTree data structures.
* Simple interface for indexing and searching documents and sentences.
* Support for various file formats and customizable indexing options.
* Integration with the SpaCy and Sentence-BERT libraries for natural language processing and sentence embeddings.

## Installation

To use FASDR, you will need to install the required packages:

```
pip install git+https://github.com/dmarx/fast-and-simple-dense-retrieval
```

Then, simply download or clone the FASDR library to your project folder.

## Quick Start

### Indexing documents

```python
from FASDR import DocumentIndex

doc_index = DocumentIndex("path/to/your/documents")
```

### Searching for documents

```python
query = "What is the meaning of life?"
doc_results = doc_index.search_documents(query, k=3)
```

### Searching for sentences

```python
sentence_results = doc_index.search_sentences(query, k=3)
```
