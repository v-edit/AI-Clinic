# dashboard.py
import sys
import os

# Add parent folder to path to access settings.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from settings import (
    call_medgemma_api,
    load_bart_pipeline,
    load_medgemma_text_model,
    load_medgemma_multimodal,
)

import streamlit as st
import plotly.express as px
from PIL import Image

def show_dashboard():
    # Logo
    try:
        logo = Image.open("logo.jpg")
        st.image(logo, width=120)
    except:
        st.write("🩺 Clinical Copilot")

    st.title("Clinical Copilot Dashboard")

    # Sidebar Inputs
    st.sidebar.header("Doctor's Query")
    text_prompt = st.sidebar.text_area("Enter patient's complaint, notes, or question:", height=150)
    image_input = st.sidebar.file_uploader("Upload scan/image (optional)", type=["png","jpg","jpeg"])

    # API Button
    if st.sidebar.button("Analyze with MedGemma API"):
        if not text_prompt and image_input is None:
            st.sidebar.warning("⚠️ Please enter text or upload image.")
        else:
            with st.spinner("🩺 Calling MedGemma API..."):
                try:
                    result_text = call_medgemma_api(text_prompt, image_input)
                    st.subheader("🧠 MedGemma Insight (via API)")
                    st.success(result_text)
                except Exception as e:
                    st.error(f"Error calling API: {e}")

    # Local Model Button
    if st.sidebar.button("Analyze with MedGemma Local"):
        if not text_prompt and image_input is None:
            st.sidebar.warning("⚠️ Need text or image.")
        else:
            with st.spinner("⚙️ Running MedGemma locally..."):
                if image_input:
                    processor, model = load_medgemma_multimodal()
                    img = Image.open(image_input).convert("RGB")
                    inputs = processor(images=img, text=text_prompt, return_tensors="pt").to("cuda")
                    outputs = model.generate(**inputs, max_length=512)
                    result_text = processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    text_model = load_medgemma_text_model()
                    out = text_model(text_prompt, max_length=256, do_sample=True, temperature=0.7)
                    result_text = out[0]['generated_text']
                st.subheader("🧠 MedGemma Local Insight")
                st.success(result_text)

    # BART Summarizer
    if st.sidebar.button("Summarize with BART"):
        if not text_prompt.strip():
            st.sidebar.warning("⚠️ Enter some patient notes first.")
        else:
            with st.spinner("🤖 Summarizing..."):
                summarizer = load_bart_pipeline()
                summary = summarizer(text_prompt, max_length=120, min_length=40, do_sample=False)
                st.subheader("🧠 BART Summary")
                st.info(summary[0]['summary_text'])

    # Example Dashboard Panels
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature", "98.6 °F")
        st.metric("BP", "120/80 mmHg")
    with col2:
        st.write("🧪 Labs")
        st.write("- Hb: 13.5 g/dL")
        st.write("- Sugar: 95 mg/dL")
    with col3:
        st.write("📝 Notes")
        st.write("Allergy: Penicillin")

    # Plotly 3D Chart
    st.markdown("---")
    df = px.data.iris()
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                        color='species', size='petal_width')
    st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown(
        "<hr><div style='text-align: center; color: grey;'>Built with ❤️ at Hackathon 2025</div>",
        unsafe_allow_html=True
    )
