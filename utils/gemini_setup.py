from google import genai
from .config import load_api_key # Import from config.py in the same directory

def initialize_gemini_client():
    """Initializes and returns the Gemini client."""
    api_key = load_api_key()
    if not api_key:
        # Raise an error or return None to be handled by the caller
        raise ValueError("Google API Key not found. Please set it in environment/.env or as an environment variable.")

    try:
        client = genai.Client(api_key=api_key)
        print("Gemini client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        raise # Re-raise the exception to indicate failure

