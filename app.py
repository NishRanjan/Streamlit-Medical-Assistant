import streamlit as st
import pandas as pd
import json
import io # For handling byte streams from file uploader

# Import utility functions
from utils.config import load_api_key
from utils.gemini_setup import initialize_gemini_client
from utils.chromadb_setup import initialize_chromadb_collection
from utils.data_processing import extract_text_from_pdf_bytes, read_text_file_bytes
from utils.storage import get_patient_ids, get_historical_data_chroma
from utils.agents import (
    process_report_and_analyze,
    suggest_next_steps_rag_chroma,
    # Import other agent functions if needed directly
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- Initialization & State Management ---
# Initialize Gemini client and ChromaDB collection only once
@st.cache_resource
def get_resources():
    try:
        client = initialize_gemini_client()
        collection = initialize_chromadb_collection()
        return client, collection
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop() # Stop execution if initialization fails
        return None, None

client, collection = get_resources()

# Initialize session state variables
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'rag_suggestion' not in st.session_state:
    st.session_state.rag_suggestion = None
if 'selected_patient_id' not in st.session_state:
    st.session_state.selected_patient_id = None
if 'historical_reports' not in st.session_state:
    st.session_state.historical_reports = []
if 'selected_history_date' not in st.session_state:
    st.session_state.selected_history_date = None

# --- Sidebar ---
with st.sidebar:
    st.title("🩺 Medical Assistant")
    st.markdown("---")

    # Patient Selection
    st.header("Patient Selection")
    patient_ids = get_patient_ids(collection) # Fetch patient IDs from DB
    if not patient_ids:
        st.info("No patient data found in the database yet.")
        # Allow adding a new patient ID when processing a report
        st.session_state.selected_patient_id = None
    else:
        # Use index=None for default empty selection if needed, or default to first patient
        default_index = patient_ids.index(st.session_state.selected_patient_id) if st.session_state.selected_patient_id in patient_ids else 0
        selected_patient = st.selectbox(
            "Select Patient:",
            options=patient_ids,
            index=default_index,
            key="patient_selector" # Assign a key to the widget
        )
        # Update session state when selection changes
        if selected_patient != st.session_state.selected_patient_id:
            st.session_state.selected_patient_id = selected_patient
            # Clear previous results when patient changes
            st.session_state.analysis_result = None
            st.session_state.rag_suggestion = None
            st.session_state.historical_reports = []
            st.session_state.selected_history_date = None
            st.rerun() # Rerun to update the UI based on new patient

    st.markdown("---")
    st.info("This app uses AI to analyze reports and symptoms. It is not a substitute for professional medical advice.")


# --- Main App Interface (Tabs) ---
tab1, tab2, tab3 = st.tabs(["Process New Report", "Symptom Analysis (RAG)", "View Patient History"])

# == Tab 1: Process New Report ==
with tab1:
    st.header("Process New Blood Report")

    # Input for Patient ID (allow adding new patients)
    new_patient_id_input = st.text_input(
        "Enter Patient ID (or select existing from sidebar):",
        value=st.session_state.selected_patient_id if st.session_state.selected_patient_id else ""
    )

    uploaded_file = st.file_uploader("Upload Blood Report (PDF or TXT)", type=["pdf", "txt"])

    if st.button("Analyze Report", key="analyze_button"):
        if uploaded_file is not None and new_patient_id_input:
            patient_id_to_process = new_patient_id_input.strip()
            st.session_state.selected_patient_id = patient_id_to_process # Update selected patient

            file_bytes = uploaded_file.getvalue()
            file_type = uploaded_file.type
            raw_report_text = None

            with st.spinner(f"Processing report for {patient_id_to_process}..."):
                # Extract text based on file type
                if file_type == "application/pdf":
                    raw_report_text = extract_text_from_pdf_bytes(file_bytes)
                elif file_type == "text/plain":
                    raw_report_text = read_text_file_bytes(file_bytes)

                if raw_report_text:
                    # Call the main orchestrator function
                    st.session_state.analysis_result = process_report_and_analyze(
                        patient_id=patient_id_to_process,
                        raw_report_text=raw_report_text,
                        client_instance=client,
                        collection=collection
                    )
                    st.success("Analysis complete!")
                     # Clear cached history for this patient as it's updated
                    st.session_state.historical_reports = []
                    st.session_state.selected_history_date = None
                else:
                    st.error("Could not extract text from the uploaded file.")
                    st.session_state.analysis_result = None
        elif not new_patient_id_input:
             st.warning("Please enter a Patient ID.")
             st.session_state.analysis_result = None
        else:
            st.warning("Please upload a report file.")
            st.session_state.analysis_result = None

    # Display Analysis Results
    if st.session_state.analysis_result:
        st.markdown("---")
        st.subheader("Analysis Results")

        result = st.session_state.analysis_result
        if result.get("error"):
            st.error(f"An error occurred during processing: {result['error']}")
        else:
            # Display Patient Info
            if "patient_info" in result:
                st.write(f"**Patient:** {result['patient_info'].get('name', 'N/A')} ({result['patient_info'].get('id', 'N/A')})")
                st.write(f"**Report Date:** {result['patient_info'].get('report_date', 'N/A')}")

            # Display Immediate Analysis (Alerts & Recs)
            if "immediate_analysis" in result and result["immediate_analysis"]:
                 with st.expander("Immediate Analysis & Alerts", expanded=True):
                     analysis = result["immediate_analysis"]
                     if isinstance(analysis, dict):
                         if analysis.get("error"):
                              st.warning(f"Could not generate immediate analysis: {analysis['error']}")
                         else:
                              alerts = analysis.get("alerts", [])
                              recs = analysis.get("recommendations", [])
                              if alerts:
                                   st.warning("Alerts Found:")
                                   # Create a simple dataframe for alerts
                                   alert_data = [{"Metric": a.get('metric'), "Value": a.get('value'), "Severity": a.get('severity', 'N/A'), "Description": a.get('description')} for a in alerts]
                                   st.dataframe(pd.DataFrame(alert_data), use_container_width=True)
                              else:
                                   st.info("No immediate critical alerts identified in this report.")

                              if recs:
                                   st.markdown("**Immediate Recommendations:**")
                                   for rec in recs:
                                        st.write(f"- {rec}")
                     else: # Handle case where analysis is just error text
                          st.warning(f"Could not generate immediate analysis: {analysis}")


            # Display Trend Analysis
            if "trend_analysis" in result and result["trend_analysis"]:
                with st.expander("Trend Analysis", expanded=True):
                     analysis = result["trend_analysis"]
                     if isinstance(analysis, dict) and analysis.get("error"):
                          st.warning(f"Could not generate trend analysis: {analysis['error']}")
                     elif isinstance(analysis, str):
                          st.markdown(analysis) # Display the text directly
                     else:
                          st.info("Trend analysis data not available or in unexpected format.")


            # Display Extracted Metrics (Optional)
            if "extracted_metrics" in result and result["extracted_metrics"]:
                with st.expander("View Extracted Metrics (JSON)"):
                    st.json(result["extracted_metrics"])


# == Tab 2: Symptom Analysis (RAG) ==
with tab2:
    st.header("Symptom Analysis (RAG)")

    if not st.session_state.selected_patient_id:
        st.warning("Please select a patient from the sidebar first.")
    else:
        st.write(f"Analyzing symptoms for patient: **{st.session_state.selected_patient_id}**")
        symptoms_input = st.text_area("Enter current patient symptoms:", height=150, key="symptoms_input")

        if st.button("Suggest Next Steps", key="rag_button"):
            if symptoms_input:
                with st.spinner("Analyzing symptoms and retrieving history..."):
                    suggestion, error = suggest_next_steps_rag_chroma(
                        collection=collection,
                        patient_id=st.session_state.selected_patient_id,
                        symptoms=symptoms_input,
                        client_instance=client,
                        top_n=3 # Retrieve top 3 relevant reports
                    )
                    if error:
                        st.error(f"RAG Error: {error}")
                        st.session_state.rag_suggestion = None
                    else:
                        st.session_state.rag_suggestion = suggestion
                        st.success("Suggestion generated!")
            else:
                st.warning("Please enter patient symptoms.")
                st.session_state.rag_suggestion = None

        # Display RAG Suggestion
        if st.session_state.rag_suggestion:
            st.markdown("---")
            st.subheader("Suggested Next Steps (based on symptoms and history):")
            st.markdown(st.session_state.rag_suggestion)


# == Tab 3: View Patient History ==
with tab3:
    st.header("View Patient History")

    if not st.session_state.selected_patient_id:
        st.warning("Please select a patient from the sidebar first.")
    else:
        st.write(f"Viewing history for patient: **{st.session_state.selected_patient_id}**")

        # Fetch history if not already in session state for this patient
        if not st.session_state.historical_reports:
             with st.spinner("Loading patient history..."):
                  st.session_state.historical_reports = get_historical_data_chroma(
                       collection=collection,
                       patient_id=st.session_state.selected_patient_id
                  )

        if not st.session_state.historical_reports:
            st.info("No historical reports found for this patient in the database.")
        else:
            report_dates = [report.get("date", f"Unknown_{i}") for i, report in enumerate(st.session_state.historical_reports)]
            # Ensure default index is valid if state exists but doesn't match options
            default_date_index = None
            if st.session_state.selected_history_date in report_dates:
                 default_date_index = report_dates.index(st.session_state.selected_history_date)

            selected_date = st.selectbox(
                "Select Report Date to View:",
                options=report_dates,
                index=default_date_index,
                key="history_date_selector"
            )

            # Update selected date in session state
            st.session_state.selected_history_date = selected_date

            # Find and display the selected report's metrics
            selected_report_data = next((report for report in st.session_state.historical_reports if report.get("date") == selected_date), None)

            if selected_report_data and selected_report_data.get("metrics"):
                st.markdown("---")
                st.subheader(f"Details for Report: {selected_date}")
                st.json(selected_report_data["metrics"])
            elif selected_date:
                 st.warning(f"Could not find details for report date: {selected_date}")

