# modules/ai_model.py

import streamlit as st
import google.generativeai as genai
import json

# --- Gemini API Setup ---
# This setup is used if you are using the Gemini API
try:
    genai.configure(api_key=st.secrets["google_api_key"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    model = None # Handle cases where API key is not set

def get_gemini_response(prompt):
    """Generic function to get a response from the Gemini model."""
    if not model:
        return "Gemini API not configured. Please add 'google_api_key' to your Streamlit secrets."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the AI model: {e}"

def get_relevant_info(patient_data, query):
    """Analyzes patient data to answer a specific query."""
    context = json.dumps(patient_data, indent=2)
    prompt = f"""
    You are an expert clinical assistant AI. Based ONLY on the provided patient's health record, answer the doctor's question concisely.
    The doctor's question is: "{query}"

    PATIENT RECORD (in FHIR JSON format):
    {context}
    """
    return get_gemini_response(prompt)

def proactive_risk_analysis(patient_data):
    """Proactively identifies potential clinical risks from a patient's record."""
    context = json.dumps(patient_data)
    prompt = f"""
    Analyze the provided patient's FHIR record. Identify the top 3 potential clinical risks
    (e.g., "Sepsis risk due to high WBC and fever," "Medication conflict between Warfarin and Aspirin").
    Present them as a brief, bulleted list. If no significant risks are apparent, state that.

    PATIENT RECORD:
    {context}
    """
    return get_gemini_response(prompt)