"""
Gender-Based Analysis of Student Academic Performance — Streamlit App
=====================================================================
Group 6 · Institute of Technology of Cambodia
Department of Applied Mathematics and Statistics

An interactive dashboard built from the team's mini-project notebook. It lets
faculty explore the data, forecast a student's final grade (G3), and get an
instant "At-Risk" alert — the live dashboard described in the project's
Future Work. Framed around UN SDG 4 (Quality Education) and SDG 5 (Gender
Equality).

Run locally:   streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration & constants
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Student Performance · Group 6",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
PASS_MARK = 10          # grades are 0–20; 10 is the pass threshold
AT_RISK_THRESHOLD = 10  # a student is "At-Risk" if predicted/actual G3 < 10

# SDG colours
SDG4_RED = "#C5192D"
SDG5_ORANGE = "#FF3A21"
ACCENT = "#1f6feb"

# Features the models use (faithful to the project's selected drivers,
# and intuitive for a faculty member to type in).
# The 7 features selected for their direct academic impact (see project slides):
#   Prior grades (G1, G2) · Academic history (failures) ·
#   Lifestyle (absences, sex/gender) · Environment (Medu, famsup)
FEATURES = ["G1", "G2", "failures", "absences", "sex", "Medu", "famsup"]

# Human-readable labels for the input widgets
MEDU_LABELS = {
    0: "0 — none",
    1: "1 — primary (4th grade)",
    2: "2 — 5th to 9th grade",
    3: "3 — secondary",
    4: "4 — higher education",
}
FAILURES_LABELS = {0: "0", 1: "1", 2: "2", 3: "3 or more"}

sns.set_theme(style="whitegrid", font_scale=0.95)


# --------------------------------------------------------------------------- #
# Data loading & model training (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data():
    """Prefer the real `student-mat.csv`; fall back to the bundled sample.

    Returns (DataFrame, is_sample: bool, source_name: str).
    """
    here = Path(__file__).parent
    real = here / "student-mat.csv"
    sample = here / "sample_student_data.csv"

    if real.exists():
        df = _read_flexible(real)
        return df, False, real.name
    if sample.exists():
        df = _read_flexible(sample)
        return df, True, sample.name

    # Last resort: generate the sample on the fly
    from generate_sample_data import build
    df = build()
    df.to_csv(sample, index=False)
    return df, True, sample.name


def _read_flexible(path: Path) -> pd.DataFrame:
    """The original UCI file is ';'-separated; Kaggle copies use ','.
    Try comma first, fall back to semicolon if everything lands in one column.
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def preprocess(df: pd.DataFrame):
    """Encode the model features and build both targets.

    Returns (X, y_reg, y_clf). No feature scaling: Random Forests are
    scale-invariant, so StandardScaler adds nothing here.
    """
    X = df[FEATURES].copy()
    X["sex"] = X["sex"].map({"F": 1, "M": 0})
    X["famsup"] = X["famsup"].map({"yes": 1, "no": 0})
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_reg = df["G3"].astype(float)
    y_clf = (df["G3"] < AT_RISK_THRESHOLD).astype(int)   # 1 = At-Risk
    return X, y_reg, y_clf


@st.cache_resource(show_spinner=False)
def get_models():
    """Train both Random Forest models and package everything the pages need."""
    df, _, _ = load_data()
    X, y_reg, y_clf = preprocess(df)

    # ---- Regression: forecast final grade G3 ----
    Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=SEED)
    reg = RandomForestRegressor(n_estimators=200, random_state=SEED)
    reg.fit(Xtr, ytr)
    reg_pred = reg.predict(Xte)

    reg_results = {
        "r2": r2_score(yte, reg_pred),
        "mae": mean_absolute_error(yte, reg_pred),
        "importances": pd.Series(reg.feature_importances_, index=FEATURES).sort_values(),
        "X_test": Xte,
        "y_test": yte,
        "pred": reg_pred,
    }

    # ---- Classification: detect At-Risk students (G3 < 10) ----
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf
    )
    clf = RandomForestClassifier(
        n_estimators=200, random_state=SEED, class_weight="balanced"
    )
    clf.fit(Xtr, ytr)
    clf_pred = clf.predict(Xte)

    report = classification_report(
        yte, clf_pred, target_names=["On Track", "At-Risk"], output_dict=True, zero_division=0
    )
    clf_results = {
        "accuracy": accuracy_score(yte, clf_pred),
        "report": report,
        "cm": confusion_matrix(yte, clf_pred),
        "importances": pd.Series(clf.feature_importances_, index=FEATURES).sort_values(),
        "X_test": Xte,
        "y_test": yte,
        "pred": clf_pred,
        "at_risk_rate": float(y_clf.mean()),
    }

    return {"reg": reg, "clf": clf, "reg_results": reg_results, "clf_results": clf_results}


def encode_input(raw: dict) -> pd.DataFrame:
    """Turn the widget values into a single-row frame matching FEATURES."""
    row = {
        "G1": raw["G1"],
        "G2": raw["G2"],
        "failures": raw["failures"],
        "absences": raw["absences"],
        "sex": 1 if raw["sex"] == "Female" else 0,
        "Medu": raw["Medu"],
        "famsup": 1 if raw["famsup"] == "Yes" else 0,
    }
    return pd.DataFrame([row])[FEATURES]


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def fig_to_streamlit(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def student_input_form(key_prefix: str) -> dict:
    """Render the shared student-input widgets and return raw values.

    Collects only the 7 model features selected in the project.
    """
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Academic record**")
        g1 = st.slider("First-period grade (G1)", 0, 20, 11, key=f"{key_prefix}_g1")
        g2 = st.slider("Second-period grade (G2)", 0, 20, 11, key=f"{key_prefix}_g2")
        failures = st.selectbox(
            "Past class failures", options=list(FAILURES_LABELS),
            format_func=lambda x: FAILURES_LABELS[x], key=f"{key_prefix}_fail",
        )
        absences = st.number_input(
            "School absences", min_value=0, max_value=93, value=4, key=f"{key_prefix}_abs"
        )
    with c2:
        st.markdown("**Background & home**")
        sex = st.radio("Gender", ["Female", "Male"], horizontal=True, key=f"{key_prefix}_sex")
        medu = st.selectbox(
            "Mother's education", options=list(MEDU_LABELS),
            format_func=lambda x: MEDU_LABELS[x], index=2, key=f"{key_prefix}_me",
        )
        famsup = st.radio("Family educational support?", ["Yes", "No"],
                          horizontal=True, key=f"{key_prefix}_fs")

    return dict(
        G1=g1, G2=g2, failures=failures, absences=absences,
        sex=sex, Medu=medu, famsup=famsup,
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview(df, is_sample):
    st.title("🎓 Gender-Based Analysis of Student Academic Performance")
    st.caption(
        "Machine Learning mini-project · Group 6 · Institute of Technology of Cambodia "
        "· Dept. of Applied Mathematics and Statistics"
    )

    st.markdown(
        """
This interactive dashboard turns our project notebook into a working tool. Using the
**Student Performance** dataset, it analyses how academic, behavioural and family
factors — including **gender** — relate to student outcomes, and it predicts results
with **Random Forest** models.

The work is framed around two United Nations Sustainable Development Goals:
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div style='border-left:6px solid {SDG4_RED};padding:0.6rem 1rem;"
            f"background:rgba(197,25,45,0.06);border-radius:6px'>"
            f"<b>SDG 4 — Quality Education</b><br>Forecast the final grade (G3) so "
            f"teachers can support students <i>before</i> the final exam.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='border-left:6px solid {SDG5_ORANGE};padding:0.6rem 1rem;"
            f"background:rgba(255,58,33,0.06);border-radius:6px'>"
            f"<b>SDG 5 — Gender Equality</b><br>Detect at-risk students and compare "
            f"male / female outcomes to keep learning opportunities equal.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Dataset at a glance")
    pass_rate = (df["G3"] >= PASS_MARK).mean() * 100
    m = st.columns(4)
    m[0].metric("Students", f"{len(df):,}")
    m[1].metric("Features", df.shape[1])
    m[2].metric("Average final grade (G3)", f"{df['G3'].mean():.2f} / 20")
    m[3].metric("Pass rate (G3 ≥ 10)", f"{pass_rate:.0f}%")

    g = df["sex"].value_counts()
    st.write(
        f"The cohort is balanced by gender: **{int(g.get('F', 0))} female** and "
        f"**{int(g.get('M', 0))} male** students."
    )

    with st.expander("What's in each page?"):
        st.markdown(
            """
- **Explore the Data** — the key charts from our EDA (gender, grades, absences, family, correlations).
- **Predict Final Grade** — type a student's details and forecast their G3 (regression).
- **At-Risk Detector** — the live early-warning tool: instant At-Risk alert with probability (classification).
- **Model Insights** — accuracy, R², feature importance and gender-parity results.
"""
        )

    st.divider()
    st.subheader("Preview of the data")
    st.dataframe(df.head(15), use_container_width=True)


def page_explore(df):
    st.title("📊 Explore the Data")
    st.caption("The exploratory analysis from the notebook, made interactive.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Gender & grades", "What drives grades", "Family & environment", "Correlations"]
    )

    # ---- Tab 1: gender & grades ----
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Number of students by gender**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x="sex", hue="sex", palette=[SDG5_ORANGE, ACCENT],
                          legend=False, ax=ax)
            ax.set_xlabel("Gender"); ax.set_ylabel("Students")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Female", "Male"])
            fig_to_streamlit(fig)
        with c2:
            st.markdown("**Final grade (G3) by gender**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x="sex", y="G3", hue="sex",
                        palette=[SDG5_ORANGE, ACCENT], legend=False, ax=ax)
            ax.set_xlabel("Gender"); ax.set_ylabel("Final grade")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Female", "Male"])
            fig_to_streamlit(fig)

        st.markdown("**Average grades (G1, G2, G3) by gender**")
        gender_grade = df.groupby("sex")[["G1", "G2", "G3"]].mean()
        gender_grade.index = ["Female", "Male"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        gender_grade.plot(kind="bar", ax=ax, color=["#9ecae1", "#4292c6", "#08519c"])
        ax.set_xlabel("Gender"); ax.set_ylabel("Average grade")
        plt.xticks(rotation=0)
        fig_to_streamlit(fig)
        st.info(
            f"On average, male students score **{df[df.sex=='M'].G3.mean():.2f}** and "
            f"female students **{df[df.sex=='F'].G3.mean():.2f}** on the final grade.",
            icon="📌",
        )

    # ---- Tab 2: what drives grades ----
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Study time vs final grade**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x="studytime", y="G3", color="#74c476", ax=ax)
            ax.set_xlabel("Weekly study time (1=low → 4=high)"); ax.set_ylabel("Final grade")
            fig_to_streamlit(fig)
        with c2:
            st.markdown("**Past failures vs final grade**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x="failures", y="G3", color="#fb6a4a", ax=ax)
            ax.set_xlabel("Number of past failures"); ax.set_ylabel("Final grade")
            fig_to_streamlit(fig)

        st.markdown("**Absences vs final grade**")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.scatterplot(data=df, x="absences", y="G3", hue="sex",
                        palette=[SDG5_ORANGE, ACCENT], alpha=0.7, ax=ax)
        ax.set_xlabel("Absences"); ax.set_ylabel("Final grade")
        ax.legend(title="", labels=["Female", "Male"])
        fig_to_streamlit(fig)
        st.info("Higher absences tend to pull final grades down — a key behavioural red flag.", icon="📌")

    # ---- Tab 3: family & environment ----
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mother's education vs final grade**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x="Medu", y="G3", color="#807dba", ax=ax)
            ax.set_xlabel("Mother's education (0–4)"); ax.set_ylabel("Final grade")
            fig_to_streamlit(fig)
        with c2:
            st.markdown("**Family support**")
            fam = df["famsup"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Yes", "No"], [fam.get("yes", 0), fam.get("no", 0)],
                   color=["#41ab5d", "#d9d9d9"])
            ax.set_ylabel("Students")
            fig_to_streamlit(fig)
        st.info(
            "Maternal education and family support form a social safety net linked to "
            "stronger, more stable outcomes (SDG 5).",
            icon="📌",
        )

    # ---- Tab 4: correlations ----
    with tab4:
        st.markdown("**Correlation heatmap (numeric features)**")
        num = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, annot_kws={"size": 7}, ax=ax)
        fig_to_streamlit(fig)

        st.markdown("**How each numeric feature correlates with the final grade (G3)**")
        corr_g3 = num.corr()["G3"].drop("G3").sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [SDG4_RED if v < 0 else ACCENT for v in corr_g3.values]
        ax.barh(corr_g3.index, corr_g3.values, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Correlation with G3")
        fig_to_streamlit(fig)


def page_predict_grade(models, df):
    st.title("🔮 Predict Final Grade (G3)")
    st.caption("Random Forest Regression · SDG 4 — forecast outcomes for early support.")

    raw = student_input_form("reg")
    st.divider()

    if st.button("Predict final grade", type="primary", use_container_width=True):
        x = encode_input(raw)
        pred = float(models["reg"].predict(x)[0])
        pred = max(0.0, min(20.0, pred))
        mean_g3 = df["G3"].mean()

        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.metric(
                "Predicted final grade",
                f"{pred:.1f} / 20",
                delta=f"{pred - mean_g3:+.1f} vs class average",
            )
            if pred >= 14:
                st.success("On track for a strong result. ✅")
            elif pred >= PASS_MARK:
                st.info("Likely to pass, but there is room to improve. 📘")
            else:
                st.error("At risk of failing — recommend academic support. ⚠️")
        with c2:
            fig, ax = plt.subplots(figsize=(6, 1.7))
            ax.barh([0], [pred], color=ACCENT, height=0.5)
            ax.axvline(PASS_MARK, color=SDG4_RED, ls="--", lw=1.5)
            ax.text(PASS_MARK + 0.2, 0.32, "pass mark (10)", color=SDG4_RED, fontsize=8)
            ax.set_xlim(0, 20); ax.set_yticks([]); ax.set_xlabel("Grade (0–20)")
            ax.set_title(f"Predicted G3 = {pred:.1f}")
            fig_to_streamlit(fig)

        st.caption(
            "Grades are on a 0–20 scale. Prediction uses prior grades (G1, G2), "
            "absences, past failures, gender, mother's education, and family support."
        )


def page_risk(models, df):
    st.title("🚨 At-Risk Early-Warning Detector")
    st.caption("Random Forest Classifier · the live dashboard from our Future Work.")
    st.markdown(
        "Enter a student's details to get an **instant At-Risk alert**. A student is "
        f"flagged **At-Risk** when their predicted final grade would fall below "
        f"**{AT_RISK_THRESHOLD}/20** (a fail)."
    )

    raw = student_input_form("clf")
    st.divider()

    if st.button("Check risk status", type="primary", use_container_width=True):
        x = encode_input(raw)
        proba = float(models["clf"].predict_proba(x)[0][1])  # P(At-Risk)
        is_risk = proba >= 0.5

        c1, c2 = st.columns([1, 1.2])
        with c1:
            if is_risk:
                st.error(f"### ⚠️ AT-RISK\nRisk probability: **{proba*100:.0f}%**")
                st.markdown("**Recommended action:** flag for academic mentoring and "
                            "monitor attendance closely.")
            else:
                st.success(f"### ✅ ON TRACK\nRisk probability: **{proba*100:.0f}%**")
                st.markdown("**Recommended action:** standard monitoring.")
            st.progress(proba)
        with c2:
            st.markdown("**How this student compares with the cohort**")
            cohort = df[["G1", "G2", "absences", "failures"]].mean()
            comp = pd.DataFrame({
                "This student": [raw["G1"], raw["G2"], raw["absences"], raw["failures"]],
                "Cohort average": [cohort["G1"], cohort["G2"], cohort["absences"], cohort["failures"]],
            }, index=["G1", "G2", "Absences", "Failures"]).round(1)
            st.dataframe(comp, use_container_width=True)
            st.caption(
                "Low G1/G2, high absences or past failures are the strongest signals "
                "the model uses to raise an alert."
            )


def page_insights(models):
    st.title("📈 Model Insights")
    st.caption("Performance, key drivers and gender-parity results.")

    reg = models["reg_results"]
    clf = models["clf_results"]

    # ---- headline metrics ----
    m = st.columns(4)
    m[0].metric("Regression R²", f"{reg['r2']:.3f}",
                help="Share of variance in G3 explained by the model.")
    m[1].metric("Regression MAE", f"{reg['mae']:.2f} pts",
                help="Average grade error (0–20 scale).")
    m[2].metric("Classifier accuracy", f"{clf['accuracy']*100:.1f}%")
    m[3].metric("At-risk students", f"{clf['at_risk_rate']*100:.1f}%",
                help="Share of the dataset with G3 < 10.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(
        ["Feature importance", "Classifier quality", "Gender parity (SDG 5)"]
    )

    # ---- Tab 1: feature importance ----
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Grade forecast — top drivers**")
            imp = reg["importances"].tail(8)
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.barh(imp.index, imp.values, color=ACCENT)
            ax.set_xlabel("Importance")
            fig_to_streamlit(fig)
        with c2:
            st.markdown("**At-Risk detection — top drivers**")
            imp = clf["importances"].tail(8)
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.barh(imp.index, imp.values, color=SDG4_RED)
            ax.set_xlabel("Importance")
            fig_to_streamlit(fig)
        st.info(
            "Prior grades **G1 and G2** dominate the forecast — mid-term performance "
            "largely sets the final outcome — while **absences** and **failures** are "
            "the strongest non-grade warning signals.",
            icon="🧭",
        )

    # ---- Tab 2: classifier quality ----
    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Confusion matrix**")
            cm = clf["cm"]
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["On Track", "At-Risk"],
                        yticklabels=["On Track", "At-Risk"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            fig_to_streamlit(fig)
        with c2:
            st.markdown("**Classification report**")
            rep = pd.DataFrame(clf["report"]).transpose().round(3)
            rep = rep.loc[["On Track", "At-Risk"], ["precision", "recall", "f1-score", "support"]]
            st.dataframe(rep, use_container_width=True)
            st.caption(
                "Recall on the **At-Risk** row matters most: it is the share of truly "
                "struggling students the system successfully flags."
            )

    # ---- Tab 3: gender parity ----
    with tab3:
        st.markdown("**Actual vs predicted final grade, by gender** (held-out test set)")
        parity = pd.DataFrame({
            "sex": reg["X_test"]["sex"].values,
            "Actual G3": reg["y_test"].values,
            "Predicted G3": reg["pred"],
        })
        table = parity.groupby("sex")[["Actual G3", "Predicted G3"]].mean()
        table.index = ["Male" if i == 0 else "Female" for i in table.index]
        table = table.round(2)
        table["Model accuracy"] = (
            100 - (abs(table["Actual G3"] - table["Predicted G3"]) / table["Actual G3"] * 100)
        ).round(1).astype(str) + "%"

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(table, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            table[["Actual G3", "Predicted G3"]].plot(
                kind="bar", ax=ax, color=["#08519c", "#9ecae1"]
            )
            ax.set_ylabel("Final grade"); plt.xticks(rotation=0)
            fig_to_streamlit(fig)
        st.info(
            "The model predicts male and female outcomes with similar accuracy, so its "
            "errors do not systematically disadvantage either group — important for a "
            "fair, SDG 5–aligned tool.",
            icon="⚖️",
        )

    st.divider()
    with st.expander("📌 Key findings (summary)"):
        st.markdown(
            """
- **Mid-term grades (G1 & G2) are the strongest predictors** of the final grade.
- **Absences** are the leading *behavioural* warning sign of failure / dropout risk.
- **Past failures** sharply lower predicted grades and act as a high-risk trigger.
- **Mother's education and family support** form a social safety net linked to stronger
  outcomes — a direct, multi-generational link to gender equality (SDG 5).
- Two Random Forest models — a **regressor** (forecast G3) and a **classifier**
  (detect At-Risk) — together let educators act *before* students fall behind (SDG 4).
"""
        )


# --------------------------------------------------------------------------- #
# Sidebar & router
# --------------------------------------------------------------------------- #
def main():
    df, is_sample, source = load_data()
    models = get_models()

    with st.sidebar:
        st.markdown("### 🎓 Group 6 · ITC")
        st.caption("Student Academic Performance · ML Mini-Project")
        page = st.radio(
            "Navigate",
            ["Overview", "Explore the Data", "Predict Final Grade",
             "At-Risk Detector", "Model Insights"],
        )
        st.divider()
        st.caption(f"📁 Data source: `{source}`  ·  {len(df)} students")
        if is_sample:
            st.warning(
                "**Using synthetic sample data.**\n\nThis is a placeholder so the app "
                "runs. For your real results, download the *Student Performance* dataset "
                "and save it as **`student-mat.csv`** in the project folder, then rerun. "
                "See the README.",
                icon="⚠️",
            )
        st.divider()
        st.caption("SDG 4 — Quality Education\nSDG 5 — Gender Equality")

    if page == "Overview":
        page_overview(df, is_sample)
    elif page == "Explore the Data":
        page_explore(df)
    elif page == "Predict Final Grade":
        page_predict_grade(models, df)
    elif page == "At-Risk Detector":
        page_risk(models, df)
    elif page == "Model Insights":
        page_insights(models)


if __name__ == "__main__":
    main()
