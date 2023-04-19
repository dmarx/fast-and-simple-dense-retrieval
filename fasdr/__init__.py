#from .fasdr import Document, DocumentIndex, create_kdtree, create_nlp_pipeline, update_kdtree, search_index
#FASDR = DocumentIndex
from .fasdr import (
    build_targeted_index,
    Document,
    DocumentIndex,
    EmptyDocument,
    create_nlp_pipeline,
    create_kdtree,
    update_kdtree,
    search_index,
)