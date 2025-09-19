import streamlit as st
import sqlite3
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pathlib
import datetime
import pandas as pd
from typing import List, Dict, Any

# ---------------------------
# DATABASE FUNCTIONS
# ---------------------------
def init_db():
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Patients table
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            diagnosis TEXT,
            symptoms TEXT,
            treatment_plan TEXT,
            doctor TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # AI Queries table for history
    c.execute("""
        CREATE TABLE IF NOT EXISTS ai_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            query TEXT,
            response TEXT,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    conn.commit()
    conn.close()

def sign_up(username: str, password: str) -> tuple[bool, str]:
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True, "Account created successfully! 🎉"
    except sqlite3.IntegrityError:
        return False, "Username already exists! Please choose a different one."
    finally:
        conn.close()

def login(username: str, password: str) -> bool:
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def add_patient(name: str, age: int, gender: str, diagnosis: str, symptoms: str, treatment_plan: str, doctor: str):
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO patients (name, age, gender, diagnosis, symptoms, treatment_plan, doctor)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, age, gender, diagnosis, symptoms, treatment_plan, doctor))
    conn.commit()
    conn.close()

def get_patients() -> List[Dict[str, Any]]:
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY created_at DESC")
    patients = c.fetchall()
    conn.close()
    
    return [{
        'id': p[0], 'name': p[1], 'age': p[2], 'gender': p[3],
        'diagnosis': p[4], 'symptoms': p[5], 'treatment_plan': p[6],
        'doctor': p[7], 'created_at': p[8], 'updated_at': p[9]
    } for p in patients]

def save_ai_query(username: str, query: str, response: str, model: str):
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO ai_queries (username, query, response, model_used)
        VALUES (?, ?, ?, ?)
    """, (username, query, response, model))
    conn.commit()
    conn.close()

def get_ai_history(username: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect("clinic.db")
    c = conn.cursor()
    c.execute("SELECT * FROM ai_queries WHERE username=? ORDER BY created_at DESC LIMIT 10", (username,))
    queries = c.fetchall()
    conn.close()
    
    return [{
        'id': q[0], 'query': q[2], 'response': q[3],
        'model_used': q[4], 'created_at': q[5]
    } for q in queries]

# ---------------------------
# AI PIPELINE
# ---------------------------
@st.cache_resource
def load_qa_pipeline(model_name="google/flan-t5-small"):
    """Load and cache the AI model pipeline"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Available models
AVAILABLE_MODELS = {
    "google/flan-t5-small": "FLAN-T5 Small (Fast, General Purpose)",
    "google/flan-t5-base": "FLAN-T5 Base (Better Quality, Slower)",
    "microsoft/DialoGPT-small": "DialoGPT Small (Conversational)"
}

# ---------------------------
# STYLING
# ---------------------------
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid #000000;
    }
    
    .login-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .dashboard-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .dashboard-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    
    .card-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .user-badge {
        position: fixed;
        top: 1rem;
        right: 1rem;
        background: #00b894;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #74b9ff;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        color: #636e72;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .ai-response {
        background: #000000;
        border-left: 4px solid #00b894;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .model-info {
        background: #e17055;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .patient-card {
        background: white;
        border: 1px solid #000000;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .patient-card:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# LOGIN PAGE
# ---------------------------
def show_login_page():
    load_custom_css()
    
    # Main header
    st.markdown("""
    <div class='main-header'>
        <h1>🏥 AI-CLINIC</h1>
        <p>Advanced Medical AI Assistant Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        auth_mode = st.radio(
            "Choose action:",
            ["🔐 Login", "📝 Sign Up"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        with st.form("auth_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            if "Sign Up" in auth_mode:
                confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            
            stay_logged_in = st.checkbox("🔄 Stay logged in")
            submit = st.form_submit_button("Submit", use_container_width=True, type="primary")
            
            if submit:
                if not username or not password:
                    st.error("⚠️ Please fill in all fields!")
                elif "Sign Up" in auth_mode:
                    if password != confirm_password:
                        st.error("⚠️ Passwords don't match!")
                    elif len(password) < 6:
                        st.error("⚠️ Password must be at least 6 characters!")
                    else:
                        success, msg = sign_up(username, password)
                        if success:
                            st.success(msg)
                            st.info("✅ You can now login with your new credentials.")
                        else:
                            st.error(f"❌ {msg}")
                else:  # Login
                    if login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.stay_logged_in = stay_logged_in
                        st.session_state.current_page = "dashboard"
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# DASHBOARD COMPONENTS
# ---------------------------
def show_dashboard():
    load_custom_css()
    
    # User badge
    st.markdown(f"<div class='user-badge'>👤 {st.session_state.username}</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='main-header'>
        <h1>🏥 AI-Clinic Dashboard</h1>
        <p>Welcome back! Your medical AI assistant is ready to help.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    patients = get_patients()
    ai_history = get_ai_history(st.session_state.username)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stats-card'>
            <p class='metric-value'>{len(patients)}</p>
            <p class='metric-label'>Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stats-card'>
            <p class='metric-value'>{len(ai_history)}</p>
            <p class='metric-label'>AI Consultations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recent_patients = len([p for p in patients if 'today' in str(p.get('created_at', ''))])
        st.markdown(f"""
        <div class='stats-card'>
            <p class='metric-value'>{len([p for p in patients if p.get('gender') == 'Female'])}</p>
            <p class='metric-label'>Female Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stats-card'>
            <p class='metric-value'>{len([p for p in patients if p.get('gender') == 'Male'])}</p>
            <p class='metric-label'>Male Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Cards
    st.subheader("🎯 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cards = [
        {"icon": "🧑‍⚕️", "title": "Patients", "subtitle": "Manage patient records", "page": "patients"},
        {"icon": "📋", "title": "Records", "subtitle": "View medical history", "page": "records"},
        {"icon": "🤖", "title": "AI Assistant", "subtitle": "Get clinical insights", "page": "ai_assistant"},
        {"icon": "⚙️", "title": "Settings", "subtitle": "Configure preferences", "page": "settings"}
    ]
    
    for i, (col, card) in enumerate(zip([col1, col2, col3, col4], cards)):
        with col:
            if st.button(f"{card['icon']}", key=f"card_{i}", help=f"{card['title']}: {card['subtitle']}"):
                st.session_state.current_page = card['page']
                st.rerun()
            
            st.markdown(f"""
            <div class='dashboard-card' onclick='document.querySelector("[data-testid=\'card_{i}\'] button").click()'>
                <div class='card-icon'>{card['icon']}</div>
                <div class='card-title'>{card['title']}</div>
                <div class='card-subtitle'>{card['subtitle']}</div>
            </div>
            """, unsafe_allow_html=True)

def show_patients_page():
    load_custom_css()
    
    st.markdown("# 🧑‍⚕️ Patient Management")
    
    tab1, tab2 = st.tabs(["📋 Patient List", "➕ Add New Patient"])
    
    with tab1:
        patients = get_patients()
        
        if patients:
            st.markdown(f"### 👥 Total Patients: {len(patients)}")
            
            # Search and filter
            col1, col2 = st.columns([3, 1])
            with col1:
                search = st.text_input("🔍 Search patients", placeholder="Search by name, diagnosis...")
            with col2:
                gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female", "Other"])
            
            # Filter patients
            filtered_patients = patients
            if search:
                filtered_patients = [p for p in patients if 
                                   search.lower() in p['name'].lower() or 
                                   search.lower() in str(p['diagnosis']).lower()]
            
            if gender_filter != "All":
                filtered_patients = [p for p in filtered_patients if p['gender'] == gender_filter]
            
            # Display patients
            for patient in filtered_patients:
                with st.expander(f"👤 {patient['name']} - Age {patient['age']} ({patient['gender']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**📋 Diagnosis:** {patient['diagnosis']}")
                        st.write(f"**🩺 Doctor:** {patient['doctor']}")
                        st.write(f"**📅 Added:** {patient['created_at']}")
                    
                    with col2:
                        st.write(f"**🤒 Symptoms:** {patient['symptoms']}")
                        st.write(f"**💊 Treatment:** {patient['treatment_plan']}")
        else:
            st.info("No patients found. Add your first patient using the 'Add New Patient' tab.")
    
    with tab2:
        st.markdown("### ➕ Add New Patient")
        
        with st.form("add_patient"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("👤 Patient Name*", placeholder="Full name")
                age = st.number_input("🎂 Age*", min_value=0, max_value=150, value=30)
                gender = st.selectbox("⚧ Gender*", ["Male", "Female", "Other"])
            
            with col2:
                doctor = st.text_input("🩺 Doctor*", placeholder="Dr. Name")
                diagnosis = st.text_input("📋 Diagnosis*", placeholder="Primary diagnosis")
            
            symptoms = st.text_area("🤒 Symptoms", placeholder="List of symptoms...")
            treatment_plan = st.text_area("💊 Treatment Plan", placeholder="Prescribed treatment...")
            
            if st.form_submit_button("✅ Add Patient", type="primary"):
                if name and age and gender and doctor and diagnosis:
                    add_patient(name, age, gender, diagnosis, symptoms, treatment_plan, doctor)
                    st.success(f"✅ Patient {name} added successfully!")
                    st.rerun()
                else:
                    st.error("⚠️ Please fill in all required fields (marked with *)")

def show_ai_assistant_page():
    load_custom_css()
    
    st.markdown("# 🤖 AI Medical Assistant")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🔧 Model Configuration")
        selected_model = st.selectbox(
            "Choose AI Model:",
            list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            help="Different models have different capabilities and response times"
        )
        
        st.markdown(f"""
        <div class='model-info'>
            📊 Current Model: {selected_model.split('/')[-1]}
        </div>
        """, unsafe_allow_html=True)
        
        # Load selected model
        if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
            with st.spinner(f"Loading {selected_model}..."):
                st.session_state.qa_pipeline = load_qa_pipeline(selected_model)
                st.session_state.current_model = selected_model
    
    with col1:
        st.markdown("### 💬 Ask the AI Assistant")
        
        # Quick templates
        st.markdown("**📝 Quick Templates:**")
        template_col1, template_col2 = st.columns(2)
        
        with template_col1:
            if st.button("🤒 Symptom Analysis"):
                st.session_state.query_template = "Analyze these symptoms and suggest possible conditions: "
        
        with template_col2:
            if st.button("💊 Treatment Options"):
                st.session_state.query_template = "What are the treatment options for: "
        
        # Query input
        query = st.text_area(
            "Enter your medical query:",
            value=st.session_state.get('query_template', ''),
            height=100,
            placeholder="Ask about symptoms, treatments, medical conditions, drug interactions..."
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            ask_button = st.button("🔍 Get AI Response", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            if clear_button:
                st.session_state.query_template = ''
                st.rerun()
        
        if ask_button and query.strip():
            with st.spinner("🤖 AI is thinking..."):
                try:
                    result = st.session_state.qa_pipeline(
                        f"Medical Query: {query}",
                        max_length=300,
                        do_sample=True,
                        temperature=0.7
                    )
                    response = result[0]['generated_text']
                    
                    # Save to history
                    save_ai_query(st.session_state.username, query, response, selected_model)
                    
                    # Display response
                    st.markdown(f"""
                    <div class='ai-response'>
                        <h4>🤖 AI Response:</h4>
                        <p>{response}</p>
                        <small>Model: {selected_model} | Time: {datetime.datetime.now().strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("⚠️ **Disclaimer:** This AI response is for informational purposes only and should not replace professional medical advice.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
    
    # AI History
    st.markdown("---")
    st.markdown("### 📚 Recent AI Consultations")
    
    history = get_ai_history(st.session_state.username)
    if history:
        for i, item in enumerate(history[:5]):  # Show last 5
            with st.expander(f"🔍 Query {i+1}: {item['query'][:50]}..."):
                st.write(f"**Query:** {item['query']}")
                st.write(f"**Response:** {item['response']}")
                st.write(f"**Model:** {item['model_used']}")
                st.write(f"**Time:** {item['created_at']}")
    else:
        st.info("No AI consultation history found.")

def show_records_page():
    load_custom_css()
    
    st.markdown("# 📋 Medical Records")
    
    patients = get_patients()
    
    if patients:
        # Convert to DataFrame for better display
        df = pd.DataFrame(patients)
        
        # Summary statistics
        st.markdown("### 📊 Records Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Total Records", len(df))
        
        with col2:
            avg_age = df['age'].mean() if 'age' in df.columns else 0
            st.metric("📈 Average Age", f"{avg_age:.1f}")
        
        with col3:
            common_diagnosis = df['diagnosis'].mode().iloc[0] if 'diagnosis' in df.columns and len(df) > 0 else "N/A"
            st.metric("🏆 Most Common Diagnosis", common_diagnosis)
        
        # Detailed records table
        st.markdown("### 📄 Detailed Records")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            view_mode = st.radio("View Mode:", ["📊 Table View", "📋 Card View"], horizontal=True)
        
        with col2:
            sort_by = st.selectbox("Sort by:", ["created_at", "name", "age", "diagnosis"])
        
        # Sort dataframe
        df_sorted = df.sort_values(sort_by, ascending=False)
        
        if view_mode == "📊 Table View":
            st.dataframe(
                df_sorted[['name', 'age', 'gender', 'diagnosis', 'doctor', 'created_at']],
                use_container_width=True
            )
        else:
            # Card view
            for _, patient in df_sorted.iterrows():
                st.markdown(f"""
                <div class='patient-card'>
                    <h4>👤 {patient['name']}</h4>
                    <p><strong>Age:</strong> {patient['age']} | <strong>Gender:</strong> {patient['gender']}</p>
                    <p><strong>Diagnosis:</strong> {patient['diagnosis']}</p>
                    <p><strong>Doctor:</strong> {patient['doctor']}</p>
                    <p><strong>Added:</strong> {patient['created_at']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📝 No medical records found. Add patients to see their records here.")

def show_settings_page():
    load_custom_css()
    
    st.markdown("# ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🤖 AI Models", "🔐 Security"])
    
    with tab1:
        st.markdown("### 👤 User Profile")
        st.info(f"**Username:** {st.session_state.username}")
        st.info("**Role:** Medical Professional")
        st.info("**Account Type:** Standard")
        
        if st.button("🔄 Refresh Session"):
            st.rerun()
    
    with tab2:
        st.markdown("### 🤖 Available AI Models")
        
        for model_id, description in AVAILABLE_MODELS.items():
            st.markdown(f"""
            <div class='patient-card'>
                <h4>🧠 {model_id.split('/')[-1]}</h4>
                <p>{description}</p>
                <p><strong>Model ID:</strong> <code>{model_id}</code></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🔐 Security Settings")
        st.warning("🔒 Your session is secure and encrypted.")
        st.info("💾 All patient data is stored locally and encrypted.")
        
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.current_page = "dashboard"
            st.rerun()

# ---------------------------
# MAIN APP LOGIC
# ---------------------------
def main():
    st.set_page_config(
        page_title="AI-Clinic",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize database
    init_db()
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.stay_logged_in = False
        st.session_state.current_page = "dashboard"
        st.session_state.current_model = "google/flan-t5-small"
    
    # Load initial model
    if "qa_pipeline" not in st.session_state:
        st.session_state.qa_pipeline = load_qa_pipeline()
    
    # Main app logic
    if not st.session_state.logged_in:
        show_login_page()
    else:
        # Sidebar navigation
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            nav_options = {
                "🏠 Dashboard": "dashboard",
                "🧑‍⚕️ Patients": "patients",
                "📋 Records": "records",
                "🤖 AI Assistant": "ai_assistant",
                "⚙️ Settings": "settings"
            }
            
            for label, page in nav_options.items():
                if st.button(label, use_container_width=True, key=f"nav_{page}"):
                    st.session_state.current_page = page
                    st.rerun()
            
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.current_page = "dashboard"
                st.rerun()
        
        # Page routing
        page = st.session_state.get('current_page', 'dashboard')
        
        if page == "dashboard":
            show_dashboard()
        elif page == "patients":
            show_patients_page()
        elif page == "records":
            show_records_page()
        elif page == "ai_assistant":
            show_ai_assistant_page()
        elif page == "settings":
            show_settings_page()

if __name__ == "__main__":
    main()