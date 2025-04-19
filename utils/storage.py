import json
from datetime import datetime
import chromadb
import google.genai as genai
from google.genai import types
import typing_extensions as typing
import json
from datetime import datetime
import numpy as np

# Note: Assumes 'collection' object is initialized elsewhere and passed in,
# or initialized globally (less ideal for modularity)
# For simplicity here, we'll assume it's passed or globally available.
# A better approach uses dependency injection or a class structure.

def generate_embedding(text_to_embed, client_instance, task="retrieval_document"):
    """Generates an embedding. Returns list of floats or None."""
    if not text_to_embed or not isinstance(text_to_embed, str): return None
    if not client_instance: return None
    print(f"[Embedding] Generating embedding for task: {task}...")
    try:
        embedding_response = client_instance.models.embed_content(
            model="models/text-embedding-004", contents=text_to_embed, config=types.EmbedContentConfig(task_type=task)
        )
        if (embedding_response and hasattr(embedding_response, 'embeddings') and
            isinstance(embedding_response.embeddings, list) and len(embedding_response.embeddings) > 0 and
            hasattr(embedding_response.embeddings[0], 'values') and isinstance(embedding_response.embeddings[0].values, list)):
             print("[Embedding] Success.")
             return [float(x) for x in embedding_response.embeddings[0].values]
        else:
             print("[Embedding] Failed - Unexpected response structure.")
             return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def store_report_data_chroma(collection, patient_id, metrics_json, client_instance): # Changed 4th arg
    """
    Stores the report data (metrics) in ChromaDB.
    Generates embedding internally using the provided client instance.
    """
    if not collection:
        print("Error: ChromaDB collection not provided to store_report_data_chroma.")
        return False
    # Ensure metrics_json is a dictionary before proceeding
    if not isinstance(metrics_json, dict):
        print(f"Error: Invalid format for metrics_json for patient {patient_id}. Expected dict.")
        return False
    # Ensure client_instance is provided for embedding
    if not client_instance:
        print(f"Error: client_instance not provided to store_report_data_chroma for patient {patient_id}. Cannot generate embedding.")
        return False # Cannot proceed without client instance if embedding is generated here

    report_date = metrics_json.get('date', 'Unknown Date')
    # Sanitize date for use in ID if necessary, ensure it's a string
    report_date_str = str(report_date).replace(" ", "_").replace("/", "-")
    entry_id = f"{patient_id}_{report_date_str}"


    # --- Generate Embedding Inside This Function ---
    embedding_vector = None
    try:
        # Convert metrics dict to JSON string for embedding
        text_to_embed = json.dumps(metrics_json)
        if text_to_embed:
            embedding_vector = generate_embedding(text_to_embed, client_instance, task="retrieval_document")
            if not embedding_vector:
                print(f"Warning: Failed to generate embedding vector for report {entry_id}. Storing without embedding.")
        else:
             print(f"Warning: No text derived from metrics_json to embed for report {entry_id}.")

    except Exception as e:
        print(f"Error during embedding generation within store_report_data_chroma for {entry_id}: {e}")
        # Decide if you want to store without embedding or return False
        embedding_vector = None # Ensure it's None if generation fails
    # ---------------------------------------------

    metadata_to_store = {
        "patient_id": patient_id,
        "report_date": report_date, # Store original date format in metadata
        "metrics_data": json.dumps(metrics_json) # Store full metrics as JSON string
    }

    try:
        collection.upsert(
            # Use the generated embedding_vector (or handle if it's None/empty)
            # ChromaDB typically expects a list of embeddings, even if only one.
            # If embedding_vector is None, pass an empty list for the embedding for this ID.
            embeddings=[embedding_vector if embedding_vector else []],
            metadatas=[metadata_to_store],
            ids=[entry_id]
        )
        print(f"Stored/Updated report for patient {patient_id} on {report_date} in ChromaDB (ID: {entry_id}).")
        return True
    except Exception as e:
        # Catch specific ChromaDB errors if possible
        print(f"Error storing/updating report {entry_id} in ChromaDB: {e}")
        # Log the type of embedding_vector causing the issue if it persists
        print(f"Debug: Type of embedding_vector passed to upsert: {type(embedding_vector)}")
        if embedding_vector:
             print(f"Debug: First element type (if list): {type(embedding_vector[0]) if isinstance(embedding_vector, list) and embedding_vector else 'N/A'}")
        return False

def get_historical_data_chroma(collection, patient_id):
    """Retrieves historical data for a patient from ChromaDB using metadata filter."""
    if not collection:
        print("Error: ChromaDB collection not provided to get_historical_data_chroma.")
        return []
    print(f"Retrieving historical data for patient {patient_id} from ChromaDB...")
    try:
        results = collection.get(
            where={"patient_id": patient_id},
            include=["metadatas"]
        )

        historical_data = []
        if results and results.get('metadatas'):
            for metadata in results['metadatas']:
                try:
                    report_date = metadata.get('report_date', 'Unknown Date')
                    metrics_str = metadata.get('metrics_data')
                    if metrics_str:
                        metrics = json.loads(metrics_str)
                        historical_data.append({
                            "date": report_date,
                            "metrics": metrics,
                            "embedding": None # Not retrieving embedding here
                        })
                    else:
                         print(f"Warning: Missing 'metrics_data' in metadata for patient {patient_id}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding stored metrics JSON for patient {patient_id}: {e}")
                except Exception as e:
                    print(f"Error processing retrieved metadata for patient {patient_id}: {e}")

            # Sort by date after retrieval
            try:
                date_formats = ["%d/%m/%Y", "%d-%b-%Y", "%Y-%m-%d"]
                def parse_date(date_str):
                    for fmt in date_formats:
                        try:
                            date_part = date_str.split('/')[0].strip()
                            return datetime.strptime(date_part, fmt)
                        except ValueError:
                            continue
                    print(f"Warning: Could not parse date '{date_str}' with known formats.")
                    return datetime.min
                historical_data.sort(key=lambda x: parse_date(x['date']))
            except Exception as e:
                print(f"Warning: Could not sort reports by date reliably. Error: {e}")

        print(f"Retrieved {len(historical_data)} historical reports for patient {patient_id}.")
        return historical_data

    except Exception as e:
        print(f"Error retrieving data from ChromaDB for patient {patient_id}: {e}")
        return []

def get_patient_ids(collection):
    """Retrieves a list of unique patient IDs from the ChromaDB collection."""
    if not collection:
        print("Error: ChromaDB collection not provided to get_patient_ids.")
        return []
    try:
        # Get all items - might be inefficient for very large collections
        # Consider alternative ways to track patients if performance is an issue
        results = collection.get(include=["metadatas"])
        if results and results.get('metadatas'):
            patient_ids = set(meta['patient_id'] for meta in results['metadatas'] if 'patient_id' in meta)
            return sorted(list(patient_ids))
        return []
    except Exception as e:
        print(f"Error retrieving patient IDs from ChromaDB: {e}")
        return []

