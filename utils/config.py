import os
from dotenv import load_dotenv

def load_api_key():
    """Loads the Google API key from .env file."""
    # Construct the path to the .env file relative to this config.py file
    # Assumes .env is in ../environment/ relative to utils/
    script_dir = os.path.dirname(__file__) # Directory of config.py
    env_dir = os.path.abspath(os.path.join(script_dir, '..', 'environment'))
    dotenv_path = os.path.join(env_dir, '.env')

    print(f"Attempting to load .env from: {dotenv_path}") # Debug print

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            print("API Key loaded successfully from .env!")
            return api_key
        else:
            print("API Key (GOOGLE_API_KEY) not found in .env file.")
            return None
    else:
        # Fallback to environment variable if .env not found or key missing
        print(".env file not found at specified path. Trying environment variable.")
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
             print("API Key loaded successfully from environment variable!")
             return api_key
        else:
            print("API Key also not found as environment variable.")
            return None


# --- ChromaDB Configuration ---
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db_store'))
COLLECTION_NAME = "patient_reports"

# Ensure the ChromaDB directory exists
if not os.path.exists(CHROMA_PATH):
    try:
        os.makedirs(CHROMA_PATH)
        print(f"Created ChromaDB directory: {CHROMA_PATH}")
    except OSError as e:
        print(f"Error creating ChromaDB directory {CHROMA_PATH}: {e}")

