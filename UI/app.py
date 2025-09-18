# app.py
import sys
import os

# Add current folder to Python path
sys.path.append(os.path.dirname(__file__))

from pages.dashboard import show_dashboard
from settings import custom_css

import streamlit as st

# Page Config
st.set_page_config(
    page_title="Clinical Copilot",
    page_icon="🩺",
    layout="wide"
)

# Inject CSS
custom_css()

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Dashboard", "⚙️ Settings"])

# Page Loader
if page == "🏠 Dashboard":
    show_dashboard()
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.info("Here you could add user profile, API configs, or model preferences.")
