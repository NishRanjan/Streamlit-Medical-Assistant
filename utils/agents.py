import google.genai as genai
from google.genai import types
import typing_extensions as typing
import json
from datetime import datetime
import numpy as np

# Import necessary functions from other utils modules
from .storage import store_report_data_chroma, get_historical_data_chroma
# Assuming generate_embedding is defined here or imported if needed separately
# For simplicity, let's redefine it here based on the last correction

# --- Define TypedDict Schemas (Match your extraction needs) ---
# Example Schema (Refined based on previous discussions)
class ExtractedMetrics(typing.TypedDict):
     patient_name: typing.Optional[str]
     date: typing.Optional[str]
     physical_metrics: typing.Optional[typing.Dict[str, typing.Any]]
     blood_parameters: typing.Optional[typing.Dict[str, typing.Any]]
     # Add other specific metrics or sections as needed

class AlertDetail(typing.TypedDict):
     metric: str
     value: str
     threshold: typing.Optional[str]
     severity: typing.Optional[str] # e.g., 'low', 'moderate', 'high'
     description: str

class ImmediateAnalysisResult(typing.TypedDict):
     patient_name: typing.Optional[str]
     date: typing.Optional[str]
     alerts: typing.Optional[typing.List[AlertDetail]]
     recommendations: typing.Optional[typing.List[str]]

class BloodReportMetricsUpdated(typing.TypedDict):
     patient_name: str 
     date: str       
     physical_metrics: list[str] 
     blood_parameters: list[str] 
     alerts: str            
     preliminary_recommendations: str

# --- Embedding Function ---
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

# --- Agent 1: Health Monitoring (JSON Extraction) ---
def extract_metrics_json(report_text, client_instance):
    """Uses LLM to extract structured JSON from report text."""
     # --- Prepare the Prompt (adjust description slightly if needed) ---
    prompt_for_json_updated = f"""
    Analyze the following blood report text. Extract the specified information.
    Format the output strictly as a JSON object matching the defined 'BloodReportMetricsUpdated' structure.
    - 'patient_name' and 'date' should be lists of strings found in the text.
    - 'physical_metrics' and 'blood_parameters' should be dictionaries containing key-value pairs of metrics.
    - 'alerts' should be a list of any explicitly mentioned alerts.
    - 'preliminary_recommendations' should be any summary or recommendations found.
    Only include information explicitly found in the text.
    Respond ONLY with the JSON object based on the BloodReportMetricsUpdated schema.

    Blood Report Text:
    ---
    """

    few_shot_examples = """
    Example Expected Output (structure may vary based on model interpretation):

    {
    "patient_name":[]
    "date":[]
    "physical_metrics": {"cholesterol": 180, "blood_sugar": 95},
    "blood_parameters": { "LDL cholesterol": 110, "HDL cholesterol": 50, "vitamin D": 35 },
    "alerts": [],
    "preliminary_recommendations": "Patient shows stable health metrics. Continue current lifestyle."
    }

    """
    prompt_for_json_updated1 = prompt_for_json_updated+" "+report_text+" "+few_shot_examples

    system_instruction = """
    You are a Health Monitoring Agent. Your task is to meticulously extract key medical measurements
    from blood report text and structure them into a predefined JSON format. Focus solely on
    extraction based on the provided text and schema. Do not infer or add information not present.
    """

    try:

        response = client_instance.models.generate_content(
            model='gemini-2.0-flash', # Use the initialized model
            config=types.GenerateContentConfig(
                temperature=0.1, # Lower temperature for less creative, more factual extraction
                response_mime_type="application/json",
                response_schema=BloodReportMetricsUpdated,
            system_instruction=system_instruction,
            ),
            contents=prompt_for_json_updated1
        )
        return response, None
    except Exception as e:
        error_msg = f"Error during JSON extraction: {e}"
        # print(f"Error during JSON extraction: {e}")
        return None, error_msg


# --- Agent 2: Early Warning (Risk Analysis & Alerts) ---
def generate_alerts_and_recommendations(parsed_json_metrics, client_instance):
    """Uses LLM to analyze extracted metrics and generate alerts/recommendations."""
    prompt_cot = f"""
    Analyze the following extracted health metrics. Think step-by-step to identify potential health risks based on standard thresholds (e.g., LDL > 130 is high, Fasting Blood Sugar > 100 is elevated, Vitamin D < 20 is deficient). Justify your reasoning before generating the final alert JSON.

    Metrics JSON:
    {parsed_json_metrics}

    Reasoning Steps:
    1. Cholesterol (245) is high (threshold generally < 200).
    2. Blood Sugar (130) is high (fasting threshold > 100).
    3. LDL Cholesterol (165) is high (threshold > 130).
    4. HDL Cholesterol (40) is low (threshold often desired > 40 or 50).
    5. Vitamin D (15) is deficient (threshold < 20).
    6. Multiple risk factors present: High Cholesterol/LDL, High Blood Sugar, Low HDL, Vit D deficiency.

    Final Alert JSON:
    """
    few_shot_examples = """
    Example Expected Output (structure may vary based on model interpretation):

    {
      "patient_name": "MR.TANNISH",
      "date": "11/08/2022",
      "alerts": [
        {
          "metric": "SERUM URIC ACID",
          "value": "8.1 mg/dL",
          "threshold": "3.5-7.2 mg/dL (Male)",
          "severity": "moderate",
          "description": "Elevated uric acid level. May indicate risk of gout or kidney stones. Consider dietary changes and follow-up with a physician."
        },
      "recommendations": [
        "Follow up with a physician for further evaluation of cardiovascular risk and management of lipid levels.",
        "Consider dietary changes to reduce triglycerides and increase HDL cholesterol (e.g., reduce saturated and trans fats, increase omega-3 fatty acids).",
        "Increase physical activity.",
        "Supplement with Vitamin D as recommended by a physician.",
        "Monitor uric acid levels and discuss management options with a physician if symptoms of gout develop.",
        "Further investigation of red blood cell indices, may be warranted."
      ]
    }

    """

    prompt_cot = prompt_cot+" "+few_shot_examples

    # Add instructions for the desired JSON output format for alerts
    # e.g., {"alert": true, "alert_reason": "...", "suggested_actions": "..."}

    try:

        response_cot = client_instance.models.generate_content(
            model='gemini-1.5-flash', # Or a fine-tuned model
            config=types.GenerateContentConfig(
                temperature=0.4, # Lower temperature for less creative, more factual extraction
                response_mime_type="application/json",
                # response_schema=Alerts,
                # system_instruction=system_instruction,
            ),
            contents=prompt_cot
        )
        return response_cot, None
    except Exception as e:
        error_msg = f"Error during alerts generation: {e}"
        return None, error_msg


# --- Agent 3: Time-Based Analytics ---
def generate_trend_analysis(patient_id, current_report_json, historical_reports, client_instance):
    """Uses LLM to analyze trends based on current and historical data."""
    if not client_instance: return "Error: Gemini client not initialized.", None
    if not current_report_json: return "Error: Current report data missing.", None
    if not historical_reports: return "No historical data available for trend analysis.", None # Not an error, just info

    report_date = current_report_json.get('date', 'Unknown Date')

    # Construct the Trend Analysis Prompt (adapt from your notebook)
    prompt = f"""
    Analyze the trends for patient '{patient_id}' based on their current and historical blood reports.
    Focus on key metrics found in the data (e.g., cholesterol, blood_sugar, specific blood parameters like SGPT, Urea).
    Identify significant trends over the available period (e.g., increasing, decreasing, stable, fluctuating). Note inconsistencies if metrics are missing in some reports.
    Provide a concise summary of the significant trends and suggest recommendations based *only* on these trends.

    Current Report ({report_date}):
    {json.dumps(current_report_json, indent=2)}

    Historical Reports (Oldest to Newest):
    """
    for report in historical_reports:
        prompt += f"\n---\nDate: {report.get('date', 'N/A')}\nMetrics:\n{json.dumps(report.get('metrics', {}), indent=2)}\n"

    prompt += """
    ---
    Trend Analysis Summary and Recommendations (based *only* on the trends observed above):
    """
    try:
        response = client_instance.models.generate_content(
            model='gemini-1.5-flash', # Or gemini-1.5-pro for potentially longer context
            config=types.GenerateContentConfig(
                temperature=0.4 # Balance factuality with generation
            ),
            contents=prompt
        )        
        trend_text = getattr(response, 'text', "LLM response format unexpected or missing text.")
        # Check for blocked response
        if not trend_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             trend_text = f"Trend analysis generation blocked due to: {response.prompt_feedback.block_reason}"

        return trend_text, None # Return text, no error
    except Exception as e:
        error_msg = f"Error during Trend Analysis LLM call: {e}"
        print(error_msg)
        return None, error_msg


# --- Agent 4: RAG for Symptoms ---
def suggest_next_steps_rag_chroma(collection, patient_id, symptoms, client_instance, top_n=3):
    """Suggests next steps based on symptoms and relevant patient history from ChromaDB."""
    if not client_instance: return "Error: Client instance missing.", None
    if not collection: return "Error: ChromaDB collection missing.", None

    print(f"\n--- Starting RAG for patient {patient_id} with symptoms: '{symptoms}' ---")

    # 1. Embed the input symptoms
    symptoms_embedding = generate_embedding(symptoms, client_instance, task="retrieval_query")
    if not symptoms_embedding:
        return "Error: Could not generate embedding for the symptoms.", None

    # 2. Query ChromaDB
    print(f"Querying ChromaDB for top {top_n} relevant reports for patient {patient_id}...")
    retrieved_context = []
    try:
        current_count = collection.count()
        if current_count == 0:
             print("Warning: ChromaDB collection is empty. Cannot query.")
             query_results = None
        else:
            query_results = collection.query(
                query_embeddings=[symptoms_embedding],
                n_results=min(top_n, current_count),
                where={"patient_id": patient_id},
                include=["metadatas", "distances"]
            )

        # 3. Extract Context
        if (query_results and query_results.get('ids') and query_results['ids'] and query_results['ids'][0]):
            print(f"\nTop {len(query_results['ids'][0])} relevant historical reports found:")
            for i, report_id in enumerate(query_results['ids'][0]):
                 if i < len(query_results['metadatas'][0]):
                    metadata = query_results['metadatas'][0][i]
                    distance = query_results['distances'][0][i]
                    report_date = metadata.get('report_date', 'Unknown Date')
                    print(f"  {i+1}. ID: {report_id}, Date: {report_date}, Distance: {distance:.4f}")
                    try:
                        metrics_str = metadata.get('metrics_data')
                        if metrics_str: retrieved_context.append({"date": report_date, "metrics": json.loads(metrics_str)})
                    except Exception as e: print(f"Error processing metadata for {report_id}: {e}")
                 else: print(f"Warning: Index mismatch for {report_id}")

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        # Continue without retrieved context if query fails, or return error

    # Fallback if no similar reports found
    if not retrieved_context:
         print("Could not find relevant reports via similarity. Falling back to latest reports.")
         all_history = get_historical_data_chroma(collection, patient_id) # Pass collection
         if all_history: retrieved_context = all_history[-top_n:]

    # 4. Construct the RAG Prompt (same as before)
    prompt_rag = f"""
    Patient ID: {patient_id}
    Current Symptoms: {symptoms}
    Relevant Patient History (Most similar or latest reports):
    ---
    """
    if retrieved_context:
        for context_report in retrieved_context:
            prompt_rag += f"\nReport Date: {context_report.get('date', 'N/A')}\n"
            try: metrics_dump = json.dumps(context_report.get('metrics', {}), indent=2)
            except Exception as dump_error: metrics_dump = f"Error formatting metrics: {dump_error}"
            prompt_rag += f"Extracted Data:\n{metrics_dump}\n---"
    else:
         prompt_rag += "\nNo relevant historical context could be retrieved.\n---"

    prompt_rag += """
    Based ONLY on the provided current symptoms and the relevant patient history above (if any), suggest potential next steps for this patient.
    Consider how the symptoms might relate to past findings. Be concise and action-oriented.
    Examples: specific tests, monitoring, specialist referral, lifestyle advice. Do NOT diagnose.
    Suggested Next Steps:
    """

    # 5. Call LLM
    print("\nRequesting suggestions from LLM using RAG...")
    try:
        response_rag = client_instance.models.generate_content(
            model='gemini-1.5-flash', # Or another suitable model
            contents=prompt_rag,
             config=types.GenerateContentConfig(
                temperature=0.5
            )
        )
        suggestion = getattr(response_rag, 'text', "LLM response format unexpected.")
        if not suggestion and hasattr(response_rag, 'prompt_feedback') and response_rag.prompt_feedback.block_reason:
             suggestion = f"Suggestion generation blocked: {response_rag.prompt_feedback.block_reason}"
        print("--- LLM RAG Response ---")
        print(suggestion)
        return suggestion, None # Return suggestion text, no error
    except Exception as e:
        error_msg = f"Error during RAG LLM call: {e}"
        print(error_msg)
        return None, error_msg


# --- Orchestrator Function ---
def process_report_and_analyze(patient_id, raw_report_text, client_instance, collection):
    """Orchestrates the full analysis pipeline for a new report."""
    if not client_instance or not collection:
        return {"error": "Client or DB Collection not initialized."}

    print(f"\n--- Starting Full Processing for Patient: {patient_id} ---")
    final_report = {"patient_info": {"id": patient_id}, "error": None} # Initialize final report structure

    # 1. Extract Metrics
    print("\nStep 1: Extracting JSON metrics...")
    parsed_json_metrics, error = extract_metrics_json(raw_report_text, client_instance)
    parsed_json_metrics = json.loads(parsed_json_metrics.text)
    if error:
        final_report["error"] = f"Extraction Error: {error}"
        return final_report # Stop processing if extraction fails
    final_report["extracted_metrics"] = parsed_json_metrics
    final_report["patient_info"]["name"] = parsed_json_metrics.get("patient_name", "N/A")
    final_report["patient_info"]["report_date"] = parsed_json_metrics.get("date", "N/A")
    print("Step 1 Complete.")

    # 2. Generate Alerts & Immediate Recommendations
    print("\nStep 2: Analyzing immediate risks...")
    immediate_analysis, error = generate_alerts_and_recommendations(parsed_json_metrics, client_instance)
    immediate_analysis = json.loads(immediate_analysis.text)
    if error:
        print(f"Warning: Alert generation failed: {error}")
        final_report["immediate_analysis"] = {"error": error}
    else:
        final_report["immediate_analysis"] = immediate_analysis
    print("Step 2 Complete.")

    # 3. Store Data and Generate Embedding (Embedding happens inside store function now)
    print("\nStep 3: Storing report and generating embedding...")
    # Pass the already parsed JSON
    success = store_report_data_chroma(collection, patient_id, parsed_json_metrics, client_instance)
    if not success:
        print("Warning: Failed to store report data in ChromaDB.")
        # Decide if you want to stop or continue without storing
    print("Step 3 Complete.")

    # 4. Time-Based Analysis (Retrieve history AFTER storing current)
    print("\nStep 4: Performing trend analysis...")
    historical_reports = get_historical_data_chroma(collection, patient_id)
    # Exclude the current report from history for trend comparison
    current_date = final_report["patient_info"]["report_date"]
    relevant_history = [r for r in historical_reports if r['date'] != current_date]

    trend_analysis_output, error = generate_trend_analysis(patient_id, parsed_json_metrics, relevant_history, client_instance)
    if error:
        print(f"Warning: Trend analysis failed: {error}")
        final_report["trend_analysis"] = {"error": error}
    else:
        final_report["trend_analysis"] = trend_analysis_output
    print("Step 4 Complete.")

    # 5. Aggregate Recommendations (Example - needs refinement)
    print("\nStep 5: Aggregating recommendations...")
    agg_recs = []
    if isinstance(final_report.get("immediate_analysis"), dict) and final_report["immediate_analysis"].get("recommendations"):
        agg_recs.extend(final_report["immediate_analysis"]["recommendations"])
    # TODO: Parse recommendations from trend_analysis_output text if needed
    # For now, just adding a note about the trend text:
    if isinstance(final_report.get("trend_analysis"), str):
         agg_recs.append(f"See Trend Analysis section for trend-based insights.")

    final_report["aggregated_recommendations"] = agg_recs
    final_report["disclaimer"] = "For Informational Purposes Only. Not Medical Advice. Consult your doctor."
    print("Step 5 Complete.")

    print(f"\n--- Full Processing Finished for Patient: {patient_id} ---")
    return final_report

