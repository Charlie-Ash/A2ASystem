from llama_index.core import VectorStoreIndex


def user_query(index: VectorStoreIndex, input_string: str) -> list[str]:

    # Retriever
    retriever = index.as_retriever(similarity_top_k= 3)  # Top 3 similar chunks to return

    # Results
    # "retriever" is already connected to "QdrantVectorStore". This'll automatically embed the input, go to the DB and find relevant chunks
    results_node = retriever.retrieve(input_string)

    # Tranfer relevant text into a list
    # Note: returning chunk size is already determined as 512 during ingestion
    results_text_list = [node.text for node in results_node]

    return results_text_list
