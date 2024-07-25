import os
from dotenv import load_dotenv

def load_api_key():
    """Load the OpenAI API key from .env.local file."""
    load_dotenv('.env.local')
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env.local file")
    return api_key