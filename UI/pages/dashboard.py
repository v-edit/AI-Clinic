# ui/pages/dashboard.py

import streamlit as st
import sys
import os

# This adds the project root to the Python path to allow importing from 'modules'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.data_loader import get_patient_list, load_patient_data, format_patient_summary
from modules.ai_model import get_relevant_info, proactive_risk_analysis

st.set_page_config(page_title="Copilot Dashboard", page_icon="🩺", layout="wide")
st.title("Clinical Copilot Dashboard")

# --- Sidebar for Patient Selection ---
st.sidebar.header("Patient Selection")
patient_list = get_patient_list()
if not patient_list:
    st.error("No patient data found. Make sure your `data/fhir_records` folder is populated with JSON files.")
    st.stop()

selected_patient_id = st.sidebar.selectbox("Choose a patient record:", patient_list)

# --- Main Dashboard ---
if selected_patient_id:
    patient_data = load_patient_data(selected_patient_id)
    
    # Proactive Risk Analysis
    with st.expander("🚨 Proactive AI Risk Analysis", expanded=True):
        with st.spinner("Scanning for potential risks..."):
            risks = proactive_risk_analysis(patient_data)
            st.warning(risks)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Summary")
        st.markdown(format_patient_summary(patient_data))

    with col2:
        st.subheader("Ask the AI Copilot")
        query = st.text_area("Enter your question about this patient:", height=100)

        if st.button("Get AI Insights"):
            if query:
                with st.spinner("Analyzing records..."):
                    answer = get_relevant_info(patient_data, query)
                    st.success("Analysis Complete!")
                    st.markdown(answer)
            else:
                st.warning("Please enter a question.")