import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Group 6 — Student Performance & Gender Equity", page_icon="🌱", layout="wide")

# ==========================================================
# 2. VISUAL IDENTITY  (growth & support theme)
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Nunito+Sans:wght@400;600;700;800&display=swap');

:root{
  --ink:#1B2A2F; --muted:#5A6B62; --bg:#F6F8F5; --line:#E4EAE3;
  --thrive:#2E9E6B; --thrive-soft:#DCEEE3; --thrive-deep:#1E7A4F;
  --care:#E8765A; --care-soft:#FBE3DC; --care-deep:#C4502F;
  --gold:#E8A33D; --teal:#155E63;
}
.stApp{ background:var(--bg); }
html,body,[class*="css"],p,div,span,label,.stMarkdown,.stRadio,.stSelectbox,.stSlider{
  font-family:'Nunito Sans',-apple-system,Segoe UI,sans-serif; color:var(--ink);
}
h1,h2,h3,h4{ font-family:'Fraunces',Georgia,serif; }

/* Hero */
.hero{ background:linear-gradient(135deg,#155E63 0%,#2E9E6B 100%);
  border-radius:22px; padding:32px 36px; margin-bottom:18px;
  box-shadow:0 14px 34px rgba(21,94,99,.28); }
.hero .pill{ display:inline-block; background:rgba(255,255,255,.16);
  border:1px solid rgba(255,255,255,.32); color:#EAFBF2; padding:5px 14px;
  border-radius:999px; font-size:12px; font-weight:800; letter-spacing:1px;
  text-transform:uppercase; margin-bottom:14px; }
.hero h1{ color:#fff; font-size:36px; font-weight:700; margin:0 0 8px 0; letter-spacing:-.5px; }
.hero p{ color:#E8F5EE; font-size:15px; margin:0; max-width:780px; line-height:1.5; }

/* Section headers */
.eyebrow{ font-size:12px; font-weight:800; letter-spacing:1.6px; text-transform:uppercase;
  color:var(--thrive); margin:6px 0 0 0; }
.section-title{ font-size:23px; font-weight:600; margin:0 0 2px 0; }
.col-head{ font-size:13px; font-weight:800; letter-spacing:.4px; text-transform:uppercase;
  color:var(--teal); border-bottom:2px solid var(--thrive-soft); padding-bottom:6px; margin-bottom:8px; }

/* Result card */
.result{ border-radius:20px; padding:26px 30px; margin:8px 0 16px 0; }
.result.thrive{ background:var(--thrive-soft); border-left:7px solid var(--thrive); }
.result.care{ background:var(--care-soft); border-left:7px solid var(--care); }
.result h2{ margin:0 0 6px 0; font-size:27px; }
.result.thrive h2{ color:var(--thrive-deep); }
.result.care h2{ color:var(--care-deep); }
.result p{ font-size:15px; margin:0; line-height:1.55; color:var(--ink); }

/* Signature: support meter */
.meter-wrap{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 22px; margin-bottom:14px; }
.meter-title{ font-size:13px; font-weight:800; letter-spacing:.5px; text-transform:uppercase; color:var(--muted); }
.meter-track{ height:18px; border-radius:999px; margin:12px 0 6px 0; position:relative;
  background:linear-gradient(90deg,#2E9E6B 0%,#E8A33D 55%,#E8765A 100%); }
.meter-marker{ position:absolute; top:-7px; width:7px; height:32px; background:var(--ink);
  border-radius:4px; box-shadow:0 0 0 3px #fff,0 2px 6px rgba(0,0,0,.25); }
.meter-labels{ display:flex; justify-content:space-between; font-size:11px; font-weight:700; color:#90A096; }

/* Recommendation cards */
.rec{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 20px; height:100%; }
.rec .num{ font-family:'Fraunces',serif; font-size:22px; font-weight:700; color:var(--thrive); }
.rec h4{ margin:8px 0 5px 0; font-size:15px; }
.rec p{ font-size:13px; color:var(--muted); margin:0; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. ENGINE & DATA  (cached)
# ==========================================================
@st.cache_data
def load_engine():
    df = pd.read_csv("student-mat.csv")
    df["needs_support"] = (df["G3"] < 10).astype(int)

    X = df.drop(columns=["G1", "G2", "G3", "needs_support"])
    y = df["needs_support"]
    raw_features = X.copy()
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    model.fit(scaler.fit_transform(X_train), y_train)
    acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))

    cohort = {
        "absences": df["absences"].mean(),
        "failures": df["failures"].mean(),
        "studytime": df["studytime"].mean(),
        "n": len(df),
        "at_risk_rate": y.mean() * 100,
    }
    return model, scaler, feature_columns, raw_features, acc, cohort

try:
    model, scaler, model_features, raw_X, model_accuracy, cohort = load_engine()
except FileNotFoundError:
    st.error("Could not find 'student-mat.csv' in this folder. Add it to the repo and rerun.")
    st.stop()

# ==========================================================
# 4. HERO
# ==========================================================
st.markdown("""
<div class="hero">
  <span class="pill">Group 6 · ITC · SDG 4 &amp; 5</span>
  <h1>Gender-Based Analysis of Student Academic Performance</h1>
  <p>A machine-learning early-warning tool that flags students at risk of failing — part of our study
  into how gender and home background shape academic outcomes, supporting Quality Education (SDG 4)
  and Gender Equality (SDG 5).</p>
</div>
""", unsafe_allow_html=True)

# Status strip
s1, s2, s3 = st.columns(3)
s1.metric("Model accuracy", f"{model_accuracy*100:.1f}%")
s2.metric("Students learned from", f"{cohort['n']}")
s3.metric("Needed support in data", f"{cohort['at_risk_rate']:.0f}%")

# ==========================================================
# 5. SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### 📘 About this project")
    st.markdown(
        "Group 6 — Institute of Technology of Cambodia, Department of Applied Mathematics and "
        "Statistics. We use a Random Forest model on the Student Performance dataset to predict "
        "at-risk students and analyse gender-based differences in academic outcomes."
    )
    st.markdown("---")
    st.markdown("### 🎯 Why it matters")
    st.markdown(
        "- **SDG 4 — Quality Education:** no student slips through the cracks unnoticed.\n"
        "- **SDG 5 — Gender Equality:** surface where background and home factors create unequal odds.\n"
        "- **Early by design:** uses habits and home context, not exam scores."
    )
    with st.expander("How to read the result"):
        st.write(
            "The support meter shows how likely a student is to need help — green means thriving, "
            "coral means it's worth checking in. The comparison shows how this student sits next to "
            "the class average."
        )

# ==========================================================
# 6. INTAKE FORM
# ==========================================================
st.markdown('<p class="eyebrow">Step 1</p><p class="section-title">Tell us about the student</p>',
            unsafe_allow_html=True)
st.caption("A quick profile of habits and home context — no grades needed.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="col-head">Who they are</div>', unsafe_allow_html=True)
    sex_choice = st.radio("Gender", ["Female", "Male"], horizontal=True, key="gender_radio")
    sex = "F" if sex_choice == "Female" else "M"
    age = st.slider("Age", 15, 22, 17)
    failures = st.selectbox("Past classes not passed", [0, 1, 2, 3],
                            help="How many classes the student has failed before.")
    absences = st.number_input("Days absent this year", min_value=0, max_value=90, value=4)

with col2:
    st.markdown('<div class="col-head">Home &amp; foundation</div>', unsafe_allow_html=True)
    medu_map = {"None": 0, "Primary (Grade 4)": 1, "Middle (Grade 9)": 2,
                "Secondary": 3, "Higher education": 4}
    Medu = medu_map[st.selectbox("Mother's education", list(medu_map.keys()), index=3)]
    study_map = {"Under 2 hours": 1, "2-5 hours": 2, "5-10 hours": 3, "Over 10 hours": 4}
    studytime = study_map[st.selectbox("Weekly study time", list(study_map.keys()), index=1)]
    internet = st.radio("Internet at home?", ["Yes", "No"], horizontal=True)

with col3:
    st.markdown('<div class="col-head">Daily life</div>', unsafe_allow_html=True)
    goout = st.slider("Time out with friends", 1, 5, 3, help="1 = rarely, 5 = very often.")
    health = st.slider("Current health", 1, 5, 5, help="1 = unwell, 5 = energetic.")
    schoolsup = st.radio("Extra school support?", ["Yes", "No"], index=1, horizontal=True)
    famsup = st.radio("Family study support?", ["Yes", "No"], index=0, horizontal=True)

# Fill remaining columns with dataset typical values (is_numeric_dtype: pandas-3 safe)
input_data = {}
for c in raw_X.columns:
    input_data[c] = int(raw_X[c].median()) if pd.api.types.is_numeric_dtype(raw_X[c]) else raw_X[c].mode()[0]
input_data.update({
    "sex": sex, "age": age, "failures": failures, "absences": absences, "Medu": Medu,
    "studytime": studytime, "internet": internet.lower(), "schoolsup": schoolsup.lower(),
    "famsup": famsup.lower(), "goout": goout, "health": health,
})

# ==========================================================
# 7. RESULT
# ==========================================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🌱 Run wellness check", type="primary", use_container_width=True):
    row = pd.DataFrame([input_data])
    enc = pd.get_dummies(row)
    final = pd.DataFrame(0, index=[0], columns=model_features)
    for c in model_features:
        if c in enc.columns:
            final[c] = enc[c]
    scaled = scaler.transform(final)
    prob = model.predict_proba(scaled)[0][1] * 100
    needs_support = model.predict(scaled)[0] == 1

    st.markdown('<p class="eyebrow">Step 2</p><p class="section-title">The wellness check</p>',
                unsafe_allow_html=True)

    if needs_support:
        st.markdown("""
        <div class="result care">
          <h2>🤝 Worth a check-in</h2>
          <p>This student's habits and home context line up with profiles that often need a little
          extra support to stay on track. A short, caring conversation now can make a real difference.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result thrive">
          <h2>🌿 On a healthy path</h2>
          <p>This student's profile points to strong momentum and a low chance of falling behind.
          Keep doing what's working and check in occasionally.</p>
        </div>""", unsafe_allow_html=True)
        st.balloons()
        st.toast("Great news - this student is thriving!", icon="🌿")

    # Signature support meter
    marker = min(max(prob, 2), 98)
    st.markdown(f"""
    <div class="meter-wrap">
      <div class="meter-title">Support-need level — {prob:.0f}%</div>
      <div class="meter-track"><div class="meter-marker" style="left:{marker}%"></div></div>
      <div class="meter-labels"><span>Thriving</span><span>Watch</span><span>Needs support</span></div>
    </div>""", unsafe_allow_html=True)

    # Live comparison with the class
    st.markdown('<p class="eyebrow">How they compare</p><p class="section-title">This student vs the class</p>',
                unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("Days absent", f"{absences}",
              delta=f"{absences - cohort['absences']:+.0f} vs class", delta_color="inverse")
    k2.metric("Past failures", f"{failures}",
              delta=f"{failures - cohort['failures']:+.1f} vs class", delta_color="inverse")
    k3.metric("Weekly study", f"{studytime}",
              delta=f"{studytime - cohort['studytime']:+.1f} vs class", delta_color="normal")

    # Care pathway / maintenance
    if needs_support:
        st.markdown('<p class="eyebrow">Step 3</p><p class="section-title">A caring next step</p>',
                    unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.markdown('<div class="rec"><span class="num">01</span><h4>Coaching</h4>'
                    '<p>Invite them to a small peer-tutoring group to review core subjects before tests.</p></div>',
                    unsafe_allow_html=True)
        r2.markdown('<div class="rec"><span class="num">02</span><h4>Check the cause</h4>'
                    '<p>See whether absences come from health, family, or transport - then remove the barrier.</p></div>',
                    unsafe_allow_html=True)
        r3.markdown('<div class="rec"><span class="num">03</span><h4>A mentor</h4>'
                    '<p>Pair them with a counsellor for an early, friendly connection that rebuilds confidence.</p></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="eyebrow">Keep it going</p><p class="section-title">Light-touch plan</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "- Keep steady access to study resources and the things already working.\n"
            "- A gentle check-in around mid-term is enough - no heavy intervention needed."
        )
