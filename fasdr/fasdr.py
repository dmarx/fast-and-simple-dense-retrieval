import pickle
import spacy
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import faiss
#from spacy_sentence_bert import SentenceTransformer
from spacy.language import Language
import fnmatch


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def create_faiss_index(data: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index for the given data.

    Args:
        data (np.ndarray): A 2D array containing the embeddings.

    Returns:
        faiss.IndexFlatL2: A FAISS index.
    """
    try:
        index = faiss.IndexFlatL2(data.shape[1])
    except IndexError:
        index = faiss.IndexFlatL2(len(data))
        data = data[:,None]
    index.add(data)
    return index

def update_faiss_index(index: faiss.IndexFlatL2, new_data: np.ndarray) -> faiss.IndexFlatL2:
    """
    Update an existing FAISS index with new data.

    Args:
        index (faiss.IndexFlatL2): An existing FAISS index.
        new_data (np.ndarray): A 2D array containing the new embeddings to add to the index.

    Returns:
        faiss.IndexFlatL2: The updated FAISS index.
    """
    index.add(new_data)
    return index

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

def search_index(index: faiss.IndexFlatL2, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the index for the nearest neighbors of the given query.

    Args:
        index (faiss.IndexFlatL2): A FAISS index.
        query (np.ndarray): A 2D array containing the query embeddings.
        k (int): The number of nearest neighbors to return.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The distances and indices of the nearest neighbors.
    """
    distances, indices = index.search(query, k)
    return distances, indices

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
        force_reindex: bool = False,
    ):
        self.file_path = Path(file_path)
        #self.last_modified = self.file_path.stat().st_mtime
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

        self._load_or_construct_summary_embedding(force_reindex=force_reindex)

#     def _load_or_construct_summary_embedding(self, force_reindex: bool=False):
#         base_name = self.file_path.stem
#         summary_save_path = self.file_path.with_name(f"{base_name}_summary_embedding.pkl")
#         out_of_date = self.file_path.stat().st_mtime > summary_save_path.stat().st_mtime
        
#         if any([out_of_date, force_reindex, not summary_save_path.exists()]):
#             self._construct()
#             self._save_summary_embedding(summary_save_path)
#             self._save_embeddings()
#         else:
#             with summary_save_path.open("rb") as file:
#                 self.summary_embedding = pickle.load(file)

    def _load_or_construct_summary_embedding(self, force_reindex: bool = False):
        summary_save_path = self.file_path.with_stem(self.file_path.stem + "_summary_embedding")
            
        if self.file_path.exists() and summary_save_path.exists():
            out_of_date = self.file_path.stat().st_mtime > summary_save_path.stat().st_mtime
        else:
            out_of_date = True

        if out_of_date or force_reindex:
            #self.summary_embedding = self._construct_summary_embedding()
            with self.file_path.open('r') as f:
                doc = self.nlp(f.read())
                self._spacy_doc = doc
                self.sentences = [str(s) for s in doc]
                self.sentence_embeddings = [s.vector for s in doc.sents]
                self.summary_embedding = doc.vector
                self.sentence_index = create_faiss_index(np.array(self.sentence_embeddings))
                self.embeddings_loaded = True
                
            with summary_save_path.open("wb") as f:
                pickle.dump(self.summary_embedding, f)
        else:
            with open(summary_save_path, "rb") as f:
                self.summary_embedding = pickle.load(f)
            
            
    def _save_summary_embedding(self, save_path: Path):
        with save_path.open("wb") as file:
            pickle.dump(self.summary_embedding, file)

    def _save_embeddings(self):
        base_name = self.file_path.stem
        save_path = self.file_path.with_name(f"{base_name}_embeddings.pkl")

        data = {
            "sentences": self.sentences,
            "sentence_embeddings": self.sentence_embeddings,
        }

        with save_path.open("wb") as file:
            pickle.dump(data, file)

    def _load_all_embeddings(self, save_path: Path):
        with save_path.open("rb") as file:
            data = pickle.load(file)
            self.sentences = data["sentences"]
            self.sentence_embeddings = data["sentence_embeddings"]
            self.sentence_index = create_faiss_index(np.array(self.sentence_embeddings))

    def _construct(self):
        with self.file_path.open("r", encoding="utf-8") as file:
            text = file.read()

        doc = self.nlp(text)
        self.sentences = [str(sent) for sent in doc.sents]
        self.sentence_embeddings = [sent.vector for sent in doc.sents]
        self.summary_embedding = np.mean(self.sentence_embeddings, axis=0)
        self.sentence_index = create_faiss_index(np.array(self.sentence_embeddings))

    def search_sentences(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[float, str]]:
        # ensure query is a 2d array
        query_embedding = query_embedding.reshape(1, -1)

        if not self.embeddings_loaded:
            self._load_all_embeddings()
            self.embeddings_loaded = True

        distances, indices = search_index(self.sentence_index, query_embedding, k)
        #results = [(distances[0][i], self.sentences[indices[0][i]]) for i in range(len(indices[0]))]
        results = []
        for dist, idx in zip(distances[0], indices[0]):
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
        whitelisted_extensions: Optional[List[str]] = None,
        ignored_patterns: Optional[List[str]] = None,
        force_reindex: bool = False,
    ):
        root_directory = root_directory or Path.cwd()
        self.root_directory = Path(root_directory)
        self.save_location = self.root_directory / ".embeddings"
        self.save_location.mkdir(parents=True, exist_ok=True)
        self.index_save_path = self.save_location / "index.pkl"
        
        self.whitelisted_extensions = whitelisted_extensions or ['.py', '.md', '.txt', '.yaml', '.yml', '.toml']

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

        
        #self.summary_index = create_faiss_index(np.empty((0, self.nlp.get_pipe("sentence_bert").model.vector_size)))
        #self.summary_index = create_faiss_index(np.empty((0, self.nlp.get_pipe("sentence_bert").vector_size)))
#         dummy_doc = self.nlp("a")
#         embedding_dim = dummy_doc.vector.shape[0]
#         self.summary_index = create_faiss_index(np.empty((0, embedding_dim)))
        self.summary_index = None # populated by .load() or ._construct_from_root_directory()
            
        self.documents = []

        if not force_reindex and self.root_directory.joinpath(".embeddings", "index.pkl").exists():
            self.load()
        else:
            self._construct_from_root_directory(force_reindex=force_reindex)
            self.save()

    def _construct_from_root_directory(self, directory: Optional[Path] = None, force_reindex: bool=False):
        if directory is None:
            directory = self.root_directory
            
        for ext in self.whitelisted_extensions:
            for file_path in directory.glob(f"*{ext}"):
                existing_document = next((doc for doc in self.documents if doc.file_path == file_path), None)

                if existing_document is not None and existing_document.last_modified >= file_path.stat().st_mtime:
                    continue

                document = Document(file_path, nlp=self.nlp, force_reindex=force_reindex)
                self.documents.append(document)

        # Recursive search for text files in subdirectories
        for subdir in directory.iterdir():
            if (
                subdir.is_dir()
                and subdir != self.save_location
                and not any(fnmatch.fnmatch(subdir.name, pattern) for pattern in self.ignored_patterns)
            ):
                subdir_index = DocumentIndex(
                    root_directory=subdir,
                    nlp=self.nlp,
                    whitelisted_extensions=self.whitelisted_extensions,
                    ignored_patterns=self.ignored_patterns,
                    force_reindex=force_reindex,
                )
                self.documents.extend(subdir_index.documents)

        # Build summary index with new embeddings
        dummy_doc = self.nlp("a")
        embedding_dim = dummy_doc.vector.shape[0]
        self.summary_index = create_faiss_index(np.empty((0, embedding_dim)))

        summary_embeddings = np.array([doc.summary_embedding for doc in self.documents])
        self.summary_index = update_faiss_index(self.summary_index, summary_embeddings)




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

    def search_documents(self, query: str, k: int = 1) -> List[Tuple[float, Path]]:
        query_embedding = self.nlp(query).vector
        distances, indices = search_index(self.summary_index, np.array([query_embedding]), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((dist, self.documents[idx].file_path))

        return results

    def search_sentences(self, query: str, k: int = 1) -> List[Tuple[float, Path, List[Tuple[float, str]]]]:
        doc_results = self.search_documents(query, k)

        query_embedding = self.nlp(query).vector

        results = []
        for dist, file_path in doc_results:
            document = next(doc for doc in self.documents if doc.file_path == file_path)
            sentence_results = document.search_sentences(np.array([query_embedding]), k)
            results.append((dist, file_path, sentence_results))

        return results