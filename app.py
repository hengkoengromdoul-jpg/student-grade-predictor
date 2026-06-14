import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Group 6 — Student Performance & Gender Equity",
                   page_icon="🌱", layout="wide")

FEATURES = ["G1", "G2", "failures", "absences", "sex", "Medu", "famsup"]

# ==========================================================
# 2. VISUAL IDENTITY (growth & support theme)
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

.hero{ background:linear-gradient(135deg,#155E63 0%,#2E9E6B 100%);
  border-radius:22px; padding:32px 36px; margin-bottom:18px;
  box-shadow:0 14px 34px rgba(21,94,99,.28); }
.hero .pill{ display:inline-block; background:rgba(255,255,255,.16);
  border:1px solid rgba(255,255,255,.32); color:#EAFBF2; padding:5px 14px;
  border-radius:999px; font-size:12px; font-weight:800; letter-spacing:1px;
  text-transform:uppercase; margin-bottom:14px; }
.hero h1{ color:#fff; font-size:34px; font-weight:700; margin:0 0 8px 0; letter-spacing:-.5px; }
.hero p{ color:#E8F5EE; font-size:15px; margin:0; max-width:800px; line-height:1.5; }

.eyebrow{ font-size:12px; font-weight:800; letter-spacing:1.6px; text-transform:uppercase;
  color:var(--thrive); margin:6px 0 0 0; }
.section-title{ font-size:23px; font-weight:600; margin:0 0 2px 0; }
.col-head{ font-size:13px; font-weight:800; letter-spacing:.4px; text-transform:uppercase;
  color:var(--teal); border-bottom:2px solid var(--thrive-soft); padding-bottom:6px; margin-bottom:8px; }

.result{ border-radius:20px; padding:26px 30px; margin:8px 0 16px 0; }
.result.thrive{ background:var(--thrive-soft); border-left:7px solid var(--thrive); }
.result.care{ background:var(--care-soft); border-left:7px solid var(--care); }
.result h2{ margin:0 0 6px 0; font-size:27px; }
.result.thrive h2{ color:var(--thrive-deep); }
.result.care h2{ color:var(--care-deep); }
.result p{ font-size:15px; margin:0; line-height:1.55; color:var(--ink); }

.gradebox{ background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:18px 22px; text-align:center; }
.gradebox .label{ font-size:12px; font-weight:800; letter-spacing:.6px; text-transform:uppercase; color:var(--muted); }
.gradebox .value{ font-family:'Fraunces',serif; font-size:46px; font-weight:700; color:var(--teal); line-height:1.1; }
.gradebox .unit{ font-size:16px; color:var(--muted); }

.meter-wrap{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 22px; }
.meter-title{ font-size:12px; font-weight:800; letter-spacing:.6px; text-transform:uppercase; color:var(--muted); }
.meter-track{ height:18px; border-radius:999px; margin:12px 0 6px 0; position:relative;
  background:linear-gradient(90deg,#2E9E6B 0%,#E8A33D 55%,#E8765A 100%); }
.meter-marker{ position:absolute; top:-7px; width:7px; height:32px; background:var(--ink);
  border-radius:4px; box-shadow:0 0 0 3px #fff,0 2px 6px rgba(0,0,0,.25); }
.meter-labels{ display:flex; justify-content:space-between; font-size:11px; font-weight:700; color:#90A096; }

.rec{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 20px; height:100%; }
.rec .num{ font-family:'Fraunces',serif; font-size:22px; font-weight:700; color:var(--thrive); }
.rec h4{ margin:8px 0 5px 0; font-size:15px; }
.rec p{ font-size:13px; color:var(--muted); margin:0; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. MODELS & DATA (cached) — 7 features INCLUDING G1, G2
# ==========================================================
@st.cache_data
def load_engine():
    df = pd.read_csv("student-mat.csv")
    d = df.copy()
    d["sex"] = d["sex"].map({"F": 1, "M": 0})
    d["famsup"] = d["famsup"].map({"yes": 1, "no": 0})

    X = d[FEATURES]
    y_reg = d["G3"]
    y_clf = (d["G3"] < 10).astype(int)

    # Regression: forecast the final grade G3
    Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
    r2 = r2_score(yte, reg.predict(Xte))

    # Classification: detect at-risk students (stratified, balanced)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_clf, test_size=0.2,
                                                  random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                 class_weight="balanced").fit(Xc_tr, yc_tr)
    acc = accuracy_score(yc_te, clf.predict(Xc_te))

    cohort = {"G1": df["G1"].mean(), "G2": df["G2"].mean(),
              "absences": df["absences"].mean(), "failures": df["failures"].mean(),
              "n": len(df), "at_risk_rate": y_clf.mean() * 100}
    return reg, clf, r2, acc, cohort

try:
    reg, clf, r2, acc, cohort = load_engine()
except FileNotFoundError:
    st.error("Could not find 'student-mat.csv' in this folder. Add it to the repo and rerun.")
    st.stop()

def encode(g1, g2, failures, absences, gender, medu, famsup):
    return pd.DataFrame([{
        "G1": g1, "G2": g2, "failures": failures, "absences": absences,
        "sex": 1 if gender == "Female" else 0, "Medu": medu,
        "famsup": 1 if famsup == "Yes" else 0,
    }])[FEATURES]

# ==========================================================
# 4. HERO
# ==========================================================
st.markdown("""
<div class="hero">
  <span class="pill">Group 6 · ITC · SDG 4 &amp; 5</span>
  <h1>Gender-Based Analysis of Student Academic Performance</h1>
  <p>A machine-learning tool that forecasts a student's final grade and flags those at risk of failing —
  part of our study into how gender and home background shape academic outcomes, supporting
  Quality Education (SDG 4) and Gender Equality (SDG 5).</p>
</div>
""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
s1.metric("At-risk model accuracy", f"{acc*100:.1f}%")
s2.metric("Grade forecast (R²)", f"{r2:.3f}")
s3.metric("Students learned from", f"{cohort['n']}")

# ==========================================================
# 5. SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### 📘 About this project")
    st.markdown(
        "Group 6 — Institute of Technology of Cambodia, Department of Applied Mathematics and "
        "Statistics. We use Random Forest models on the Student Performance dataset to forecast "
        "final grades, detect at-risk students, and analyse gender-based differences."
    )
    st.markdown("---")
    st.markdown("### 🎯 Why it matters")
    st.markdown(
        "- **SDG 4 — Quality Education:** spot students who will pass weakly and help them improve before finals.\n"
        "- **SDG 5 — Gender Equality:** examine how gender and home factors shape outcomes.\n"
        "- **After mid-terms:** uses the early grades G1 and G2 — the strongest predictors — to forecast the final result."
    )
    with st.expander("How to read the result"):
        st.write(
            "You'll get two things: the predicted final grade (G3, out of 20) and the risk of failing. "
            "Green means thriving; coral means it's worth a check-in. The comparison shows how this "
            "student sits next to the class average."
        )

# ==========================================================
# 6. INPUT FORM — the 7 features
# ==========================================================
st.markdown('<p class="eyebrow">Step 1</p><p class="section-title">Enter the student\'s details</p>',
            unsafe_allow_html=True)
st.caption("Uses the seven features from our study, including the early-term grades G1 and G2.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="col-head">Recent grades</div>', unsafe_allow_html=True)
    g1 = st.slider("First-period grade (G1)", 0, 20, 11)
    g2 = st.slider("Second-period grade (G2)", 0, 20, 11)

with col2:
    st.markdown('<div class="col-head">Academic history</div>', unsafe_allow_html=True)
    failures = st.selectbox("Past classes not passed", [0, 1, 2, 3],
                            help="How many classes the student has failed before.")
    absences = st.number_input("Days absent this year", min_value=0, max_value=90, value=4)

with col3:
    st.markdown('<div class="col-head">Background</div>', unsafe_allow_html=True)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True, key="gender_radio")
    medu_map = {"None": 0, "Primary (Grade 4)": 1, "Middle (Grade 9)": 2,
                "Secondary": 3, "Higher education": 4}
    medu = medu_map[st.selectbox("Mother's education", list(medu_map.keys()), index=3)]
    famsup = st.radio("Family study support?", ["Yes", "No"], horizontal=True)

# ==========================================================
# 7. RESULT — predicts BOTH G3 and risk
# ==========================================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🌱 Predict grade & check risk", type="primary", use_container_width=True):
    X_in = encode(g1, g2, failures, absences, gender, medu, famsup)
    pred_g3 = float(reg.predict(X_in)[0])
    risk = float(clf.predict_proba(X_in)[0][1]) * 100
    at_risk = clf.predict(X_in)[0] == 1

    st.markdown('<p class="eyebrow">Step 2</p><p class="section-title">Prediction</p>',
                unsafe_allow_html=True)

    if at_risk:
        st.markdown("""
        <div class="result care">
          <h2>🤝 At risk — worth a check-in</h2>
          <p>This student's profile lines up with those who often struggle to pass the final exam.
          A short, caring conversation now can change the outcome before finals.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result thrive">
          <h2>🌿 On track to pass</h2>
          <p>This student's early grades and profile point to a healthy final result and a low chance
          of falling behind. Keep doing what's working.</p>
        </div>""", unsafe_allow_html=True)
        st.balloons()
        st.toast("On track to pass — nicely done!", icon="🌿")

    # Two headline outputs: predicted grade + risk meter
    g_col, m_col = st.columns([1, 2])
    with g_col:
        st.markdown(f"""
        <div class="gradebox">
          <div class="label">Predicted final grade (G3)</div>
          <div class="value">{pred_g3:.1f}<span class="unit"> / 20</span></div>
        </div>""", unsafe_allow_html=True)
    with m_col:
        marker = min(max(risk, 2), 98)
        st.markdown(f"""
        <div class="meter-wrap">
          <div class="meter-title">Risk of failing — {risk:.0f}%</div>
          <div class="meter-track"><div class="meter-marker" style="left:{marker}%"></div></div>
          <div class="meter-labels"><span>On track</span><span>Watch</span><span>At risk</span></div>
        </div>""", unsafe_allow_html=True)

    # Comparison with the class
    st.markdown('<p class="eyebrow">How they compare</p><p class="section-title">This student vs the class</p>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("G1", f"{g1}", delta=f"{g1 - cohort['G1']:+.1f} vs class", delta_color="normal")
    k2.metric("G2", f"{g2}", delta=f"{g2 - cohort['G2']:+.1f} vs class", delta_color="normal")
    k3.metric("Days absent", f"{absences}", delta=f"{absences - cohort['absences']:+.0f} vs class", delta_color="inverse")
    k4.metric("Past failures", f"{failures}", delta=f"{failures - cohort['failures']:+.1f} vs class", delta_color="inverse")

    # Next step
    if at_risk:
        st.markdown('<p class="eyebrow">Step 3</p><p class="section-title">A caring next step</p>',
                    unsafe_allow_html=True)
        r1, r2c, r3 = st.columns(3)
        r1.markdown('<div class="rec"><span class="num">01</span><h4>Coaching</h4>'
                    '<p>Invite them to a small peer-tutoring group to review core subjects before finals.</p></div>',
                    unsafe_allow_html=True)
        r2c.markdown('<div class="rec"><span class="num">02</span><h4>Check the cause</h4>'
                     '<p>See whether absences come from health, family, or transport — then remove the barrier.</p></div>',
                     unsafe_allow_html=True)
        r3.markdown('<div class="rec"><span class="num">03</span><h4>A mentor</h4>'
                    '<p>Pair them with a counsellor for an early, friendly connection that rebuilds confidence.</p></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="eyebrow">Keep it going</p><p class="section-title">Light-touch plan</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "- Keep steady access to study resources and the habits already working.\n"
            "- A gentle check-in before finals is enough — no heavy intervention needed."
        )
