from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Dict, Any
import json
import logging
import os

logger = logging.getLogger(__name__)

def safe_filter_metadata(metadata):
    if isinstance(metadata, dict):
        return {k: str(v) for k, v in metadata.items() if not isinstance(v, (list, dict))}
    elif isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return safe_filter_metadata(parsed)
        except json.JSONDecodeError:
            pass
    return {}

def create_documents(combined_data: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for item in combined_data:
        if isinstance(item, dict) and 'content' in item and 'metadata' in item:
            content = item['content']
            metadata = safe_filter_metadata(item['metadata'])
            documents.append(Document(page_content=content, metadata=metadata))
        else:
            logger.warning(f"Skipping invalid item: {item}")
    
    logger.info(f"Created {len(documents)} documents from {len(combined_data)} data items")
    return documents

def create_vector_store(combined_data: List[Dict[str, Any]], embeddings, persist_directory: str) -> Chroma:
    documents = create_documents(combined_data)
    logger.info(f"Created {len(documents)} documents from {len(combined_data)} data items")
    
    if not documents:
        raise ValueError("No valid documents created from combined data")
    
    batch_size = 500
    
    try:
        # Create a new Chroma instance
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
        
        # Persist the vector store
        vector_store.persist()
        
        # Get the count of documents
        count = len(vector_store.get())
        logger.info(f"Vector store created and persisted with {count} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def load_vector_store(persist_directory: str, embeddings) -> Chroma:
    try:
        logger.info(f"Attempting to load vector store from {persist_directory}")
        
        # Check if the directory exists
        if not os.path.exists(persist_directory):
            logger.error(f"Persist directory does not exist: {persist_directory}")
            raise FileNotFoundError(f"Persist directory not found: {persist_directory}")
        
        # List contents of the directory
        dir_contents = os.listdir(persist_directory)
        logger.info(f"Contents of {persist_directory}: {dir_contents}")
        
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Get the count of documents
        doc_count = len(vector_store.get())
        logger.info(f"Loaded vector store with {doc_count} documents")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise

def perform_similarity_search(vector_store, query, k=3):
    try:
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Similarity search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        raise

def create_retriever(vector_store: Chroma, search_kwargs: dict = {"k": 20}):
    default_kwargs = {
        "score_threshold": 0.7,
        "k": 20
    }
    # Update default_kwargs with any provided search_kwargs
    default_kwargs.update(search_kwargs)
    
    return vector_store.as_retriever(
        search_kwargs=default_kwargs,
        search_type="similarity_score_threshold"
    )