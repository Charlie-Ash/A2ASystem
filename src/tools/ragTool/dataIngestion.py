# Llama Index for chunking and linking up with other parts
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
# Qdrant DB client
import qdrant_client
# PDF parser
import pymupdf
# Pytorch to empty CUDA cache (after embedding)
import torch
# Garbage collector
import gc

from tools.ragTool.config import COLLECTION_NAME, QDRANT_DB_PATH, EMBEDDING_MODEL, DATA_DIR


def load_documents() -> list[Document]:  # Probably add a path argument here in the future for better modification

    documents = []

    # Find all PDF files recursively
    pdf_files = DATA_DIR.rglob("*.pdf")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")
        try:
            pdf = pymupdf.open(pdf_file)
            full_text = ""

            for page in pdf:
                full_text += page.get_text()

            pdf.close()
        except Exception as e:
            print(f"Skipping {pdf_file.name}: failed to parse ({e})")
            continue

        # Create LlamaIndex Document object
        documents.append(

            Document(text= full_text, metadata= {"file_name": pdf_file.name})

        )

    return documents


def build_vector_index() -> tuple[VectorStoreIndex, qdrant_client.QdrantClient]:

    # Document loading
    documents = load_documents()

    # Chunk documents
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    chunks = splitter.get_nodes_from_documents(documents=documents)  # All PDF file content's chunks

    # Embed chunks
    try:
        embedding_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,  # BAAI/bge-m3: multilingual embedding model, ~568M params
            trust_remote_code=True,
            device="cpu"  # Embedding to run on CPU since it interferes with LLM in the GPU (resource-wise)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {e}") from e

    # Setup Qdrant
    db_client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)  # DB client

    if db_client.collection_exists(COLLECTION_NAME):  # Delete duplications of vector stores
        db_client.delete_collection(COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Build persisting indexes (will re-embed if ran again)
    index = VectorStoreIndex(
        chunks,
        storage_context= storage_context,
        embed_model= embedding_model
    )

    # Clean-up (Delete embedding model & empty CUDA cache)
    del embedding_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return Index
    return index, db_client


def load_vector_index() -> tuple[VectorStoreIndex, qdrant_client.QdrantClient]:  # To load the index if the DB is already created

    db_client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)  # DB client

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Embedding model is still needed when loading existing DB. Still used in user query.
    try:
        embedding_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
            device="cpu"  # Embedding to run on CPU since it interferes with LLM in the GPU (resource-wise)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {e}") from e

    # Persisting indexes
    index = VectorStoreIndex.from_vector_store(
        vector_store= vector_store,
        storage_context= storage_context,
        embed_model= embedding_model,
    )

    # Clean-up (Delete embedding model & empty CUDA cache)
    del embedding_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return Index
    return index, db_client
