import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. PAGE SETUP & HUMAN-CENTRIC BRANDING
# ==========================================
st.set_page_config(page_title="Early Intervention Dashboard", page_icon="🌱", layout="wide")

# Custom CSS for a warm, clean layout  (FIX: unsafe_allow_html, not unsafe_index)
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌱 Early Student Support & Intervention Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Using predictive insights to build inclusive, equitable, and quality learning environments (Supporting UN SDG 4 & 5)</p>', unsafe_allow_html=True)

# ==========================================
# 2. ENGINE & DATA PROCESSING (Cached)
# ==========================================
@st.cache_data
def load_and_clean_engine():
    df = pd.read_csv("student-mat.csv")

    # Target: support needed if final grade G3 < 10
    df['needs_support'] = (df['G3'] < 10).astype(int)

    # NOTE: this version excludes G1, G2, G3 from the features.
    X = df.drop(columns=['G1', 'G2', 'G3', 'needs_support'])
    y = df['needs_support']

    raw_features = X.copy()
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, feature_columns, raw_features, acc

try:
    model, scaler, model_features, raw_X, model_accuracy = load_and_clean_engine()
except FileNotFoundError:
    st.error("❌ Could not locate 'student-mat.csv' in this directory. Please upload it to proceed.")
    st.stop()

# ==========================================
# 3. SIDEBAR: PROJECT INSIGHTS & PURPOSE
# ==========================================
with st.sidebar:
    st.markdown("### 📊 Dashboard Status")
    st.metric(label="Model Accuracy", value=f"{model_accuracy * 100:.1f}%")

    st.markdown("---")
    st.markdown("### 🎯 Core Objectives")
    st.markdown("""
    * **SDG 4 (Quality Education):** Ensuring no student quietly slips through the cracks.
    * **SDG 5 (Gender Equality):** Mapping socio-demographic variables like parental education to address systemic gaps.
    * **Early prediction:** This version uses lifestyle and background factors only.
    """)

# ==========================================
# 4. HUMANISED INPUT FORM
# ==========================================
st.markdown("### 🧑‍🎓 Student Profile Intake Form")
st.caption("Fill out the baseline and behavioural details below to run an academic wellness check.")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Socio-Demographics**")
        sex_choice = st.radio("Gender Profile", options=["Female", "Male"], horizontal=True, key="gender_radio")
        sex = "F" if sex_choice == "Female" else "M"
        age = st.slider("Current Age", 15, 22, 17)
        failures = st.selectbox("Previous Unpassed Classes", options=[0, 1, 2, 3],
                                help="Number of past class failures.")
        absences = st.number_input("Total Days Absent", min_value=0, max_value=90, value=0)

    with col2:
        st.markdown("**Home & Foundation**")
        medu_map = {"None": 0, "Primary (4th Grade)": 1, "Middle School (9th Grade)": 2,
                    "Secondary Education": 3, "Higher Education": 4}
        Medu_label = st.selectbox("Mother's Educational Background", options=list(medu_map.keys()), index=3)
        Medu = medu_map[Medu_label]

        study_map = {"Less than 2 hours": 1, "2 to 5 hours": 2, "5 to 10 hours": 3, "More than 10 hours": 4}
        study_label = st.selectbox("Weekly Self-Study Commitment", options=list(study_map.keys()), index=1)
        studytime = study_map[study_label]

        internet = st.radio("Reliable Web Access at Home?", options=["Yes", "No"], horizontal=True)

    with col3:
        st.markdown("**Social Lifestyle Balance**")
        goout = st.slider("Frequency of Going Out with Peers", 1, 5, 3,
                          help="1: Extremely isolated, 5: Highly social outside class hours.")
        health = st.slider("Self-Reported Health & Vitality", 1, 5, 5,
                           help="1: Severe exhaustion/illness, 5: Energetic and healthy.")
        schoolsup = st.radio("Receiving Extra Institutional Support?", options=["Yes", "No"], index=1, horizontal=True)
        famsup = st.radio("Receiving Educational Family Support?", options=["Yes", "No"], index=0, horizontal=True)

# Fill non-form columns with dataset modes/medians to keep the form light
input_data = {col: raw_X[col].mode()[0] if raw_X[col].dtype == 'object' else int(raw_X[col].median())
              for col in raw_X.columns}

# Overwrite with the user's choices  (FIX: sex was always becoming "M" before)
input_data['sex'] = sex
input_data['age'] = age
input_data['failures'] = failures
input_data['absences'] = absences
input_data['Medu'] = Medu
input_data['studytime'] = studytime
input_data['internet'] = internet.lower()
input_data['schoolsup'] = schoolsup.lower()
input_data['famsup'] = famsup.lower()
input_data['goout'] = goout
input_data['health'] = health

# ==========================================
# 5. EXECUTION & ACTIONABLE OUTPUT
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Evaluate Academic Wellness & Support Needs", type="primary", use_container_width=True):

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)

    final_input = pd.DataFrame(0, index=[0], columns=model_features)
    for col in model_features:
        if col in input_encoded.columns:
            final_input[col] = input_encoded[col]

    scaled_vector = scaler.transform(final_input)
    prediction = model.predict(scaled_vector)[0]
    probability = model.predict_proba(scaled_vector)[0][1] * 100

    st.markdown("---")
    st.markdown("### 🎯 Diagnostic Evaluation Results")

    if prediction == 1:
        st.error("⚠️ **Action Recommended: Higher Academic Stress Risk Detected**")

        m1, m2 = st.columns([1, 3])
        m1.metric("Vulnerability Index", f"{probability:.1f}%",
                  help="Probability that this profile needs proactive academic support.")
        m2.markdown(f"""
        **Educational Assessment:** This student shows a **{probability:.1f}%** alignment with profiles
        that struggle to pass final assessments under standard teaching conditions.
        The strongest stress signals are past unpassed classes (`failures`) and attendance irregularities (`absences`).
        """)

        st.markdown("#### 🛠️ Recommended Care Pathway")
        c1, c2, c3 = st.columns(3)
        c1.info("**1. Academic Coaching**\n\nInvite the student to peer-tutoring groups to review core subjects before major tests.")
        c2.info("**2. Flexibility Review**\n\nCheck whether high absences come from health, family, or transport issues.")
        c3.info("**3. Mentorship Connection**\n\nAssign a counsellor to build an early connection and rebuild classroom confidence.")

    else:
        st.success("✨ **Academic Standing: Stable & Thriving**")
        st.balloons()   # 🎉 celebrate a good result

        m1, m2 = st.columns([1, 3])
        m1.metric("Vulnerability Index", f"{probability:.1f}%")
        m2.markdown(f"""
        **Educational Assessment:** This student's current indicators reflect strong academic endurance.
        Their profile shows a very low (**{probability:.1f}%**) risk of falling behind.
        """)

        st.markdown("#### 🌾 Maintenance Plan")
        st.markdown("* Keep providing standard access to campus web services and resource centres.\n"
                    "* Monitor progress trends leading up to midterms without heavy intervention.")
