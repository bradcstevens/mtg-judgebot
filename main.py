import os
import logging
import re
from typing import List, Dict, Any
from nltk import ngrams
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from data_processor import process_cards_for_database, prepare_cards_for_vector_store, process_rules_and_glossary_data
from mtg_cards_api import fetch_card_details_by_oracle_id
from embeddings import initialize_embeddings
from vector_store import create_vector_store, load_vector_store, create_retriever
from langchain_core.runnables import RunnablePassthrough
from config import load_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and tokenizer
model_path = "models/mtg_card_name_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

print(model.config)  # Print model configuration

# Assuming the label map is defined as follows (adjust if different):
id_to_label = {0: "O", 1: "B-CARD", 2: "I-CARD"}

def perform_ner(query: str) -> List[str]:
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    card_names = []
    current_card = []
    
    for token, pred in zip(tokens, predictions[0]):
        if id_to_label[pred.item()] == "B-CARD":
            if current_card:
                card_names.append(tokenizer.convert_tokens_to_string(current_card))
                current_card = []
            current_card.append(token)
        elif id_to_label[pred.item()] == "I-CARD":
            current_card.append(token)
        elif current_card:
            card_names.append(tokenizer.convert_tokens_to_string(current_card))
            current_card = []
    
    if current_card:
        card_names.append(tokenizer.convert_tokens_to_string(current_card))
    
    return [name.strip() for name in card_names if name.strip()]

def create_or_load_sqlite_db(database_path: str, cards_file_path: str, rulings_file_path: str):
    if not os.path.exists(database_path):
        logger.info(f"Creating new SQLite database at {database_path}")
        process_cards_for_database(cards_file_path, rulings_file_path, database_path)
    else:
        logger.info(f"SQLite database already exists at {database_path}")

def create_or_load_vector_store(persist_directory: str, embeddings, data_processor, file_paths: List[str]):
    if os.path.exists(persist_directory):
        logger.info(f"Loading existing vector store from {persist_directory}")
        return load_vector_store(persist_directory, embeddings)
    else:
        logger.info(f"Creating new vector store in {persist_directory}")
        data = data_processor(*file_paths)
        return create_vector_store(data, embeddings, persist_directory)

def process_query(query: str, mtg_chain, database_path: str):
    print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
    try:
        # Perform NER on the query
        card_names = perform_ner(query)
        
        print(f"Detected card names: {card_names}")

        if not card_names:
            print("No card names detected, using original query.")
            modified_query = query
        else:
            modified_query = f"Information about {', '.join(card_names)}: {query}"

        print(f"Modified query: {modified_query}")
        retrieved_items = mtg_chain.invoke(modified_query)
        print_search_results(retrieved_items)
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")

def print_search_results(results):
    print("Retrieved Cards:")
    # for i, result in enumerate(results, 1):
    #     if result.metadata.get('document_type') == 'card':
    #         print(f"{i}. Card Name: {result.page_content}")
    #         print(f"   Oracle ID: {result.metadata['oracle_id']}")
    #     else:
    #         print(f"{i}. Content: {result.page_content}")
    #     print()

def main():
    cards_file_path = 'data/oracle-cards-20240722210341.json'
    rulings_file_path = 'data/rulings-20240722210039.json'
    rules_file_path = 'data/official-rules.txt'
    glossary_file_path = 'data/glossary.txt'
    
    database_path = "db/mtg_cards.sqlite"
    cards_vector_store_path = "db/chroma_db_cards"
    rules_glossary_vector_store_path = "db/chroma_db_rules_glossary"

    file_paths = [cards_file_path, rulings_file_path, rules_file_path, glossary_file_path]
    if not all(os.path.exists(path) for path in file_paths):
        logger.error("One or more required files not found.")
        return

    try:
        api_key = load_api_key()
        embeddings = initialize_embeddings(api_key)
    except ValueError as e:
        logger.error(f"Error loading API key or initializing embeddings: {e}")
        return

    create_or_load_sqlite_db(database_path, cards_file_path, rulings_file_path)

    cards_vector_store = create_or_load_vector_store(
        cards_vector_store_path, 
        embeddings, 
        prepare_cards_for_vector_store,
        [cards_file_path, rulings_file_path]
    )

    # rules_glossary_vector_store = create_or_load_vector_store(
    #     rules_glossary_vector_store_path,
    #     embeddings,
    #     process_rules_and_glossary_data,
    #     [rules_file_path, glossary_file_path]
    # )

    cards_retriever = create_retriever(cards_vector_store)
    # rules_glossary_retriever = create_retriever(rules_glossary_vector_store)

    mtg_chain = RunnablePassthrough() | (cards_retriever
                                        #  , rules_glossary_retriever
                                         )

    queries = [
        "How much mana does Black Lotus cost to play?",
        "What is the effect of Time Walk?",
        "How many cards do you draw with Brainstorm?",
        "Can you use Mana Drain in Modern?",
        "My name is Michael"
    ]

    for query in queries:
        process_query(query, mtg_chain, database_path)

if __name__ == "__main__":
    main()
