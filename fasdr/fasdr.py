import pickle
import spacy
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
from scipy.spatial import KDTree
from spacy.language import Language
import fnmatch
from collections import OrderedDict


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class EmptyDocument(ValueError):
    pass

def create_kdtree(data: np.ndarray) -> KDTree:
    """
    Create a KDTree for the given data.

    Args:
        data (np.ndarray): A 2D array containing the embeddings.

    Returns:
        KDTree: A KDTree index.
    """
    return KDTree(data)

def update_kdtree(tree: KDTree, new_data: np.ndarray) -> KDTree:
    """
    Update an existing KDTree index with new data.

    Args:
        tree (KDTree): An existing KDTree index.
        new_data (np.ndarray): A 2D array containing the new embeddings to add to the index.

    Returns:
        KDTree: The updated KDTree index.
    """
    #new_data = new_data.reshape(1, -1)
    combined_data = np.concatenate((tree.data, new_data))
    return create_kdtree(combined_data)

def create_nlp_pipeline(model_name: str = DEFAULT_MODEL_NAME) -> Language:
    """
    Create a SpaCy pipeline with a Sentence-BERT component.

    Args:
        model_name (str): The name of the Sentence-BERT model to use.

    Returns:
        SentenceTransformerLanguage: A SpaCy pipeline with the Sentence-BERT component.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("sentence_bert", config={"model_name": model_name})
    return nlp

def search_index(index: KDTree, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the index for the nearest neighbors of the given query.

    Args:
        index (KDTree): A KDTree index.
        query (np.ndarray): A 2D array containing the query embeddings.
        k (int): The number of nearest neighbors to return.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The distances and indices of the nearest neighbors.
    """
    distances, indices = index.query(query, k)
    return distances.ravel(), indices.ravel()


class Document:
    """
    A class to represent a document and its embeddings.

    Attributes:
        file_path (Union[str, Path]): The path to the document file.
        nlp (Optional[spacy.Language]): A SpaCy pipeline with a Sentence-BERT component.
        model_name (str): The name of the Sentence-BERT model to use.
        force_reindex (bool): If True, force reindexing the document even if it was previously indexed.

    Example:
        >>> doc = Document("path/to/your/file.txt")
        >>> query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        >>> results = doc.search_sentences(query_embedding, k=3)
    """
    def __init__(
        self,
        file_path: Union[str, Path],
        nlp: Optional[spacy.Language] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.file_path = Path(file_path)
        self.model_name = model_name
        self.nlp = nlp

        if nlp is None:
            nlp = create_nlp_pipeline(model_name)
            self.nlp = nlp
        else:
            self.model_name = self.nlp.get_pipe("sentence_bert").model_name

        self.sentences = []
        self.sentence_embeddings = None
        self.summary_embedding = None
        self.sentence_index = None

        self.embeddings_loaded = False
        self._construct()

    def save_embeddings(self):
        """
        Save the sentence embeddings and sentences of the document to a file.
        """
        base_name = self.file_path.stem
        save_path = self.file_path.with_name(f"{base_name}_embeddings.pkl")

        data = {
            "sentences": self.sentences,
            "sentence_embeddings": self.sentence_embeddings,
        }

        with save_path.open("wb") as file:
            pickle.dump(data, file)

    def load_embeddings(self, save_path: Path):
        """
        Load the sentence embeddings and sentences of the document from the given file.

        Args:
            save_path (Path): The file path to load the embeddings and sentences from.
        """
        with save_path.open("rb") as file:
            data = pickle.load(file)
            self.sentences = data["sentences"]
            self.sentence_embeddings = data["sentence_embeddings"]
            self.sentence_index = create_kdtree(np.array(self.sentence_embeddings))

    def _construct(self):
        with self.file_path.open("r", encoding="utf-8") as file:
            text = file.read()
        text=text.strip()
        if not text:
            raise EmptyDocument

        doc = self.nlp(text)
        self.sentences = [str(sent) for sent in doc.sents]
        if not self.sentences:
            raise EmptyDocument
        self.sentence_embeddings = [sent.vector for sent in doc.sents]
        self.summary_embedding = doc.vector #np.mean(self.sentence_embeddings, axis=0)
        # to do: don't need to build document index here. make a lazy property
        self.sentence_index = create_kdtree(np.array(self.sentence_embeddings))

    def search_sentences(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[float, str]]:
        # ensure query is a 2d array
        query_embedding = query_embedding.reshape(1, -1)

        if not self.embeddings_loaded:
            self._construct()
            self.embeddings_loaded = True

        distances, indices = search_index(self.sentence_index, query_embedding, k)

        results = []
        for dist, idx in zip(distances, indices):
            idx = int(idx)
            results.append((dist, self.sentences[idx]))

        return results
    
    
class DocumentIndex:
    """
    A class to represent an index of documents and their embeddings.

    Attributes:
        root_directory (Union[str, Path]): The root directory containing the documents.
        model_name (str): The name of the Sentence-BERT model to use.
        nlp (Optional[spacy.language.Language]): A SpaCy pipeline with a Sentence-BERT component.
        whitelisted_extensions (Optional[List[str]]): A list of file extensions to include in the index.
        ignored_patterns (Optional[List[str]]): A list of directory patterns to ignore.
        force_reindex (bool): If True, force reindexing the documents even if they were previously indexed.

    Example:
        >>> doc_index = DocumentIndex("path/to/your/documents")
        >>> query = "What is the meaning of life?"
        >>> doc_results = doc_index.search_documents(query, k=3)
        >>> sentence_results = doc_index.search_sentences(query, k=3)
    """
    def __init__(
        self,
        root_directory: Union[str, Path],
        model_name: str = DEFAULT_MODEL_NAME,
        nlp: Optional[spacy.language.Language] = None,
        extensions: Optional[List[str]] = None,
        ignored_patterns: Optional[List[str]] = None,
        force_reindex: bool = False,
    ):
        root_directory = root_directory or Path.cwd()
        self.root_directory = Path(root_directory)
        self.save_location = self.root_directory / ".embeddings"
        self.save_location.mkdir(parents=True, exist_ok=True)
        self.index_save_path = self.save_location / "index.pkl"
        
        self.extensions = extensions or ['.py', '.md', '.txt', '.yaml', '.yml', '.toml']

        if ignored_patterns is None:
            self.ignored_patterns = [".*"]  # Ignore folders starting with a period by default
        else:
            self.ignored_patterns = ignored_patterns
        
        self.model_name = model_name

        if nlp is None:
            self.nlp = create_nlp_pipeline(model_name)
        else:
            self.nlp = nlp
            self.model_name = self.nlp.get_pipe("sentence_bert").model_name

        self.summary_index = None # populated by .load() or ._construct_from_root_directory()
            
        self.documents = OrderedDict()

        if not force_reindex and self.root_directory.joinpath(".embeddings", "index.pkl").exists():
            self.load()
        else:
            self._construct_from_root_directory(force_reindex=force_reindex)
            self.save()

    def _construct_from_root_directory(self, directory: Optional[Path] = None, force_reindex: bool=False):
        if directory is None:
            directory = self.root_directory
            
        for ext in self.extensions:
            for file_path in directory.glob(f"*{ext}"):
                existing_document = self.documents.get(file_path)

                if (existing_document is not None) and \
                    (existing_document.last_modified >= file_path.stat().st_mtime) and \
                    (not force_reindex):
                    continue

                try:
                    document = Document(file_path, nlp=self.nlp)
                    self.documents[file_path] = document
                except EmptyDocument:
                    continue

        # Recursive search for text files in subdirectories
        for subdir in directory.iterdir():
            if (
                subdir.is_dir()
                and subdir != self.save_location
                and not any(fnmatch.fnmatch(subdir.name, pattern) for pattern in self.ignored_patterns)
            ):
                try:
                    subdir_index = DocumentIndex(
                        root_directory=subdir,
                        nlp=self.nlp,
                        extensions=self.extensions,
                        ignored_patterns=self.ignored_patterns,
                        force_reindex=force_reindex,
                    )
                    self.documents.update(subdir_index.documents)
                except EmptyDocument:
                    continue

        if len(self.documents) > 0:
            summary_embeddings = np.array([doc.summary_embedding for doc in self.documents.values()])
            self.summary_index = create_kdtree(summary_embeddings)
        else:
            raise EmptyDocument

    def save(self):
        data = {
            "model_name": self.model_name,
            "documents": self.documents,
            "summary_index": self.summary_index,
        }

        with self.index_save_path.open("wb") as file:
            pickle.dump(data, file)

    def load(self):
        with self.index_save_path.open("rb") as file:
            data = pickle.load(file)
            self.model_name = data["model_name"]
            self.documents = data["documents"]
            self.summary_index = data["summary_index"]

    def search_documents(
        self,
        query: Optional[str]=None,
        k: int = 1,
        query_embedding: np.ndarray = None
    ) -> List[Tuple[float, Path]]:
        """
        Search for the documents containing the most similar summary embeddings to the given query.

        Args:
            query (Optional[str]): The query to search for. If a query embedding is provided, this argument will be ignored.
            k (int): The number of most similar documents to return.
            query_embedding (np.ndarray): An optional query embedding to use instead of a query string.

        Returns:
            List[Tuple[float, Path]]: A list of tuples containing the distances and file paths of the most similar documents.
        """
        if query_embedding is None:
            query_embedding = self.nlp(query).vector
        distances, indices = search_index(self.summary_index, np.array([query_embedding]), k)

        results = []
        for dist, idx in zip(distances, indices):
            document_key = list(self.documents.keys())[idx]
            results.append((dist, document_key))

        return results

    def search_sentences(self, query: str, k: int = 1, query_embedding: np.ndarray = None):
        """
        Search for the most similar sentences across all documents to the given query.

        Args:
            query (str): The query to search for.
            k (int): The number of most similar sentences to return.
            query_embedding (np.ndarray): An optional query embedding to use instead of a query string.

        Returns:
            List[Tuple[float, Path, List[Tuple[float, str]]]]: A list of tuples containing the distances, file paths, and
            the most similar sentences for each file.
        """
        if query_embedding is None:
            query_embedding = self.nlp(query).vector
        doc_subset = list(self.documents.values())
        return self.search_in_documents(query_embedding=query_embedding, docs=doc_subset, k=k)

    def search_sentences_targeted(
        self, 
        query: str, 
        n_docs: int = 1,
        n_sents: int = 1,
        query_embedding: np.ndarray = None,
    ) -> List[Tuple[float, Path, List[Tuple[float, str]]]]:
        """
        Search for sentences in the documents that match a query string.

        Args:
        - query (str): The query string to search for.
        - n_docs (int): The number of top documents to consider.
        - n_sents (int): The number of top sentences to return.
        - query_embedding (np.ndarray): The precomputed query embedding, if any.

        Returns:
        - List[Tuple[float, Path, List[Tuple[float, str]]]]: A list of tuples containing:
        - float: The relevance score of the sentence to the query.
        - Path: The path to the file where the sentence was found.
        - List[Tuple[float, str]]: A list of tuples containing:
            - float: The relevance score of the sentence to the query.
            - str: The sentence text.
        """
        if query_embedding is None:
            query_embedding = self.nlp(query).vector

        if n_docs < len(self.documents):
            doc_results = self.search_documents(query_embedding=query_embedding, k=n_docs)
            doc_subset = [self.documents[file_path] for _, file_path in doc_results]
        else:
            doc_subset = list(self.documents.values())
        return self.search_in_documents(query_embedding=query_embedding, docs=doc_subset, k=n_sents)

    def search_in_documents(self, docs: List[Document], query_embedding: Optional[np.ndarray] = None, k: int = 1):
        """
        Search for sentences in a set of documents that match a query embedding.

        Args:
        - docs (List[Document]): A list of documents to search in.
        - query_embedding (Optional[np.ndarray]): The precomputed query embedding, if any.
        - k (int): The number of top sentences to return.

        Returns:
        - List[Dict[str, Union[float, str]]]: A list of dictionaries containing:
        - str: The path to the file where the sentence was found.
        - str: The text of the sentence.
        - float: The relevance score of the sentence to the query.
        """
        filtered = build_targeted_index(docs)
        distances, indices = search_index(filtered['index'], np.array([query_embedding]), k)

        results = []
        for dist, idx in zip(distances, indices):
            idx = int(idx)
            fpath = filtered['file_paths'][idx]
            sent = filtered['sentences'][idx]
            rec = {'fpath':fpath, 'text':sent, 'score':dist}
            results.append(rec)

        return results


def build_targeted_index(docs: List[Document]):
    """
    Build a KD-Tree index over the sentence embeddings in a set of documents.

    Args:
    - docs (List[Document]): A list of documents to build the index from.

    Returns:
    - Dict[str, Union[List[str], KDTree]]: A dictionary containing:
      - List[str]: The text of all sentences in the documents.
      - List[str]: The file paths of all documents in the set, repeated for each of their sentences.
      - KDTree: The KD-Tree index built from the sentence embeddings.
    """
    assert len(docs) >0
    sentences = []
    sentence_embeddings = []
    file_paths = []
    for doc in docs:
        assert len(doc.sentences) == len(doc.sentence_embeddings)
        n = len(doc.sentences)
        sentences.extend(doc.sentences)
        sentence_embeddings.extend(doc.sentence_embeddings)
        file_paths.extend([str(doc.file_path.absolute()) for _ in range(n)])
    return dict(
        sentences=sentences,
        file_paths=file_paths,
        index=KDTree(np.array(sentence_embeddings)),
    )

