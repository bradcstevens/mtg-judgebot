from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Any

def create_documents(data: List[Dict[str, Any]], data_type: str) -> List[Document]:
    documents = []
    for item in data:
        if data_type == 'cards':
            content = f"{item['metadata']['name']}: {item['metadata']['oracle_text']}"
            metadata = {
                'oracle_id': item['content'],
                'name': item['metadata']['name'],
                'mana_cost': item['metadata'].get('mana_cost', ''),
                'type_line': item['metadata'].get('type_line', ''),
                'data_type': data_type
            }
        elif data_type == 'rulings':
            content = item['content']
            metadata = {
                'oracle_id': item['metadata'].get('oracle_id', ''),
                'published_at': item['metadata'].get('published_at', ''),
                'data_type': data_type
            }
        elif data_type == 'rules':
            content = item['content']
            metadata = {
                'id': item['id'],
                'rule_number': item['rule_number'],
                'base_rule': item['base_rule'],
                'parent_rule': item['parent_rule'],
                'data_type': data_type
            }
        elif data_type == 'glossary':
            content = f"{item['term']}: {item['definition']}"
            metadata = {
                'id': item['id'],
                'term': item['term'],
                'rule_refs': ','.join(item['rule_refs']),
                'data_type': data_type
            }
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def create_vector_store(data: List[Dict[str, Any]], embeddings, data_type: str) -> Chroma:
    documents = create_documents(data, data_type)
    return Chroma.from_documents(documents, embeddings, collection_name=data_type)

def perform_similarity_search(vector_store: Chroma, query: str, k: int = 1):
    return vector_store.similarity_search(query, k=k)