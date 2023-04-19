# FASDR: Fast And Simple Dense Retrieval

FASDR is a lightweight and efficient library for fast and simple dense retrieval. It is designed to facilitate searching and retrieval of dense vector representations while keeping the storage and search overhead minimal. FASDR is particularly well-suited for projects that require fast, efficient searching of embeddings and their corresponding textual data.

## Motivation

The motivation behind FASDR is to provide a solution for projects that require efficient storage and retrieval of embeddings alongside their textual representations. While existing solutions like FAISS and Annoy are powerful and versatile, they can be overkill for smaller-scale projects or use cases where simplicity is a priority. FASDR aims to address these needs by providing a streamlined and easy-to-use interface for working with embeddings and their corresponding text data.

## Use Cases

FASDR is particularly well-suited for the following use cases:

1. Searching and retrieval of precomputed embeddings in small to medium-sized datasets.
2. Efficient storage and retrieval of embeddings alongside their textual representations.
3. Rapid prototyping and development of applications that require embedding-based search and retrieval.

## Installation

```bash
pip install fasdr
```

## Basic Usage

Here's a simple example demonstrating how to use FASDR for storing and retrieving embeddings alongside their textual representations:

```python
from fasdr import FASDR

# Initialize FASDR with the desired dimensions for your embeddings
fasdr = FASDR(dimensions=300)

# Add embeddings and their corresponding text data
embedding1 = [0.1, 0.2, 0.3, ...]
text1 = "This is a sample sentence."
fasdr.add_item(embedding1, text1)

embedding2 = [0.4, 0.5, 0.6, ...]
text2 = "Another example sentence."
fasdr.add_item(embedding2, text2)

# Build the index
fasdr.build()

# Query the index for nearest neighbors
query_embedding = [0.15, 0.25, 0.35, ...]
k = 2  # Number of nearest neighbors to retrieve
results = fasdr.query(query_embedding, k)

# Print the results
for score, text in results:
    print(f"Score: {score}, Text: {text}")
```

## Core Functionality

FASDR provides the following core functionality:

add_item(embedding, text): Add an embedding and its corresponding textual representation to the index.
build(): Build the index after adding all items. This step is required before querying the index.
query(query_embedding, k): Query the index for the k nearest neighbors of a given query_embedding.
save(file_path): Save the FASDR index to a file for later use.
load(file_path): Load a previously saved FASDR index from a file.
For more details on FASDR's API, please refer to the documentation.

## Contributing

We welcome contributions to FASDR! Please submit issues, bug reports, or feature requests through the GitHub repository, and feel free to open pull requests for improvements or bug fixes.

