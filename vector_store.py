# vector_store.py
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Any
import json

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
    return documents

def create_vector_store(combined_data: List[Dict[str, Any]], embeddings) -> Chroma:
    documents = create_documents(combined_data)
    
    if not documents:
        raise ValueError("No valid documents created from combined data")
    
    batch_size = 5000
    vector_store = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(batch, embeddings, collection_name="mtg_data")
        else:
            vector_store.add_documents(batch)
    
    return vector_store

def perform_similarity_search(vector_store: Chroma, query: str, k: int = 3):
    return vector_store.similarity_search(query, k=k)