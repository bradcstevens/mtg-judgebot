from langchain_openai import OpenAIEmbeddings

def initialize_embeddings(api_key):
    """Initialize and return the OpenAI embeddings."""
    return OpenAIEmbeddings(openai_api_key=api_key)