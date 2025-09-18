import streamlit as st
import requests, base64, io, torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForVision2SeqLM
)

# ---------------------
# API Function
# ---------------------
def call_medgemma_api(text_prompt: str, image_file=None):
    endpoint = st.secrets["MEDGEMMA_ENDPOINT"]
    api_key = st.secrets["API_KEY"]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": text_prompt}

    if image_file is not None:
        buffered = io.BytesIO()
        image_file.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        payload["image_base64"] = img_b64

    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json().get("generated_text", "No response")

# ---------------------
# Model Loaders
# ---------------------
@st.cache_resource
def load_bart_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_medgemma_text_model():
    model_name = "google/medgemma-27b-text-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_medgemma_multimodal():
    model_name = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return processor, model

# ---------------------
# CSS Styling
# ---------------------
def custom_css():
    st.markdown(
        """
        <style>
        .main { background-color: #f8fbff; }
        .stTitle { color: #004080; font-size: 38px !important; font-weight: bold; }
        .card { background: white; padding: 20px; border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
