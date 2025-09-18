import streamlit as st
import os
from backend.fhir_loading import load_all_fhir, summarize_bundle
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModel

# Model loader for BioBERT QA and Med-Gemma
@st.cache_resource(show_spinner=True)
def load_qa_pipeline(model_choice):
    if model_choice == "BioBERT QA":
        model_name = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return qa
    elif model_choice == "Med-Gemma (Medical LLM)":
        model_name = "google/med-gemma-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        qa = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        return qa
    else:
        raise ValueError("Unknown model choice")

st.set_page_config(page_title="AI Clinical Copilot", layout="wide")

# Custom CSS for a modern dark look
st.markdown("""
    <style>
    .main, .stApp {
        background-color: #111111 !important;
    }
    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #90caf9;
        margin-bottom: 0.5em;
        letter-spacing: 1px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #b3e5fc;
        margin-bottom: 1em;
    }
    .stButton>button {
        background-color: #3949ab;
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1976d2;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        background: #222;
        color: #fff;
    }
    .stExpanderHeader {
        font-weight: 600;
        color: #90caf9;
    }
    .stCodeBlock, .stMarkdown, .stText {
        color: #e3f2fd !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="big-title">🩺 AI Clinical Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your AI-powered assistant for patient data review and clinical decision support</div>', unsafe_allow_html=True)

# --- Model selection ---
with st.sidebar:
    st.markdown("## 🤖 Select AI Model")
    model_choice = st.selectbox(
        "Choose a model for clinical insights:",
        ["BioBERT QA", "Med-Gemma (Medical LLM)"]
    )

# Load FHIR data
bundle = load_all_fhir()

# --- Patient selection logic ---
# Find all Patient resources (robust extraction)
patients = []
for entry in bundle.get("entry", []):
    res = entry.get("resource", {})
    if res.get("resourceType") == "Patient":
        name = res.get("name", [{}])[0].get("text") or res.get("name", [{}])[0].get("family") or "Unknown"
        gender = res.get("gender", "?")
        dob = res.get("birthDate", "?")
        pid = res.get("id", "?")
        patients.append({"id": pid, "label": f"{name} ({gender}, {dob})", "resource": res})

if not patients:
    st.markdown('<div style="color:#b71c1c;font-weight:600;font-size:1.2rem;margin:2em 0;">⚠️ No Patient resources found in your FHIR data.<br>Check that your data files contain valid FHIR Bundles with Patient resources.</div>', unsafe_allow_html=True)
    st.stop()

patient_options = [p["label"] for p in patients]
with st.sidebar:
    st.markdown("## 👤 Select Patient")
    selected_idx = st.selectbox(
        "Select a patient",
        options=range(len(patient_options)),
        format_func=lambda i: patient_options[i],
        label_visibility="collapsed"
    )
selected_patient = patients[selected_idx]["resource"]
selected_patient_id = selected_patient.get("id")

# Filter bundle to only include selected patient and related resources
def filter_bundle_for_patient(bundle, patient_id):
    filtered = {"resourceType": "Bundle", "type": "collection", "entry": []}
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == "Patient" and res.get("id") == patient_id:
            filtered["entry"].append(entry)
        elif res.get("resourceType") in ["Condition", "MedicationRequest", "Observation", "Procedure", "Immunization", "AllergyIntolerance", "Encounter"]:
            subject = res.get("subject", {}).get("reference", "")
            if subject.endswith(f"Patient/{patient_id}"):
                filtered["entry"].append(entry)
    return filtered

patient_bundle = filter_bundle_for_patient(bundle, selected_patient_id)

# Show summary of selected patient data
with st.expander("Patient Data Summary", expanded=True):
    st.code(summarize_bundle(patient_bundle), language="text")

# --- Smarter Clinical Copilot Section ---
st.markdown("<div style='margin-top:1.5em; font-weight:600; color:#3949ab;'>Smarter Clinical Copilot</div>", unsafe_allow_html=True)
doctor_query = st.text_input("Enter a clinical question or complaint (e.g. 'What is the best treatment for this patient’s hypertension?')", key="doc_query")

def extract_relevant_data(bundle, question):
    # Use the QA model to extract only relevant data points for the query
    context = summarize_bundle(bundle)
    qa = load_qa_pipeline(model_choice)
    if model_choice == "BioBERT QA":
        result = qa(question=question, context=context)
        answer = result.get("answer", "")
    else:
        prompt = f"Patient summary: {context}\n\nQuestion: {question}\n\nRelevant data points:"
        outputs = qa(prompt)
        answer = outputs[0]["generated_text"][len(prompt):].strip() if outputs and isinstance(outputs, list) else ""
    return answer


# Suggest three generic care options (no citations)
def suggest_care_options(question, relevant_data):
    options = []
    for i in range(3):
        options.append({
            "option": f"Care option {i+1} for: {question} (based on extracted patient data)",
            "citation": ""
        })
    return options
if st.button("Get Clinical Copilot Suggestions") and doctor_query:
    with st.spinner("Analyzing patient records and medical evidence..."):
        relevant_data = extract_relevant_data(patient_bundle, doctor_query)
        care_options = suggest_care_options(doctor_query, relevant_data)
    st.markdown("<div style='font-size:1.1rem;font-weight:600;color:#1a237e;margin-bottom:0.5em;'>Relevant Patient Data</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#222;padding:1em;border-radius:8px;color:#fff'>{relevant_data}</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.1rem;font-weight:600;color:#1a237e;margin:1em 0 0.5em 0;'>Evidence-Based Care Options</div>", unsafe_allow_html=True)
    for idx, opt in enumerate(care_options, 1):
        st.markdown(f"<div style='background:#263238;padding:1em;border-radius:8px;margin-bottom:1em;'>"
                    f"<b>Option {idx}:</b> {opt['option']}"
                    f"</div>", unsafe_allow_html=True)

