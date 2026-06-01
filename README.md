# 🎓 Gender-Based Analysis of Student Academic Performance

An interactive **Streamlit** dashboard built from our Machine Learning mini-project.
It analyses the *Student Performance* dataset and uses **Random Forest** models to
forecast a student's final grade and flag students who are **at risk** — the live
early-warning tool described in our project's Future Work.

> **Group 6** · Institute of Technology of Cambodia · Department of Applied
> Mathematics and Statistics · Lecturer: Dr. Phauk Sokkhey · 2025–2026

The project is framed around two UN Sustainable Development Goals:
**SDG 4 (Quality Education)** and **SDG 5 (Gender Equality)**.

---

## What the app does

The dashboard has five pages (use the sidebar to navigate):

1. **Overview** — project summary, SDG framing, and dataset statistics.
2. **Explore the Data** — the key charts from our EDA: gender distribution, grades by
   gender, study time, absences, family support, mother's education, and a correlation
   heatmap.
3. **Predict Final Grade** — type a student's details and get a forecast of their final
   grade `G3` (Random Forest **regression**).
4. **At-Risk Detector** — get an instant **At-Risk / On-Track** alert with a risk
   probability (Random Forest **classification**). A student is flagged At-Risk when
   their predicted final grade would fall below 10/20.
5. **Model Insights** — R², accuracy, feature importance, a confusion matrix, and a
   gender-parity comparison of predicted vs actual grades.

---

## About the data

This project includes the real **`student-mat.csv`** dataset — 395 students from the
*Math* course of the UCI/Kaggle *Student Performance* dataset — so the app shows **real
results out of the box** (no sidebar warning). Source:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
· [Kaggle](https://www.kaggle.com/datasets/whenamancodes/student-performance).

It also ships with **`sample_student_data.csv`**, a *synthetic* fallback that is used
**only if `student-mat.csv` is missing** (in which case a yellow warning appears in the
sidebar). The app always prefers the real file. The loader handles both comma- and
semicolon-separated files — the original UCI file uses `;`.

---

## Project structure

```
student-performance-app/
├── app.py                    # The Streamlit app (run this)
├── student-mat.csv           # Real dataset (395 students) — used by default
├── generate_sample_data.py   # Builds the synthetic fallback dataset
├── sample_student_data.csv   # Synthetic fallback (used only if real data missing)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # App theme
├── .gitignore
└── README.md
```

---

## ▶️ Run it locally (VS Code)

You need **Python 3.10+** installed.

```bash
# 1. Open the folder in VS Code, then open a terminal (Terminal → New Terminal)

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Press `Ctrl+C` in the terminal to stop.

> **Tip:** install the **Python** extension in VS Code for the best experience.

---

## 🐙 Push it to GitHub

```bash
# from inside the project folder
git init
git add .
git commit -m "Student performance Streamlit app"

# create an EMPTY repo on github.com first, then connect it:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

You can also do all of this with VS Code's built-in **Source Control** panel
(the branch icon on the left) — click **Publish to GitHub**.

---

## ☁️ Deploy free on Streamlit Community Cloud

1. Make sure your code is pushed to a **public** GitHub repo (steps above).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `your-username/your-repo`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**. After a minute or two you'll get a public URL like
   `https://your-app.streamlit.app` that you can share with anyone.

Whenever you `git push` a change, the deployed app updates automatically.

---

## Models & method (summary)

- **Features used (7, selected for direct academic impact):** prior grades `G1` and
  `G2`; `absences`; past `failures`; `sex` (gender); `Medu` (mother's education); and
  `famsup` (family support).
- **Regression** (`RandomForestRegressor`) forecasts the final grade `G3`.
- **Classification** (`RandomForestClassifier`, `class_weight="balanced"`) flags
  At-Risk students (defined as `G3 < 10`).
- **80/20 train/test split**, `random_state=42` for reproducibility.

**On the real `student-mat.csv` dataset, the app achieves R² ≈ 0.86 (MAE ≈ 1.07 grade
points) for grade forecasting and ≈ 0.89 accuracy for At-Risk detection** (At-Risk
precision ≈ 0.81, recall ≈ 0.85). About 33% of students fall in the At-Risk band.

### Key findings (real data)
- **Prior grades dominate the forecast** — the second-period grade `G2` alone accounts
  for roughly 79% of the model's predictive power. This makes the tool an *after-midterm*
  early-warning system rather than a start-of-year one.
- **Absences** are the strongest *behavioural* signal once grades are accounted for.
- In the exploratory analysis, **past failures**, **lower mother's education**, and **no
  family support** all line up with weaker final grades — a social safety net with a
  multi-generational link to gender equality (SDG 5) — even though the Random Forest
  leans mostly on `G2`.
- **Gender:** in this Math cohort, **female students show a higher At-Risk rate (≈ 36% vs
  ≈ 29%)** and a slightly lower average final grade (≈ 9.97 vs ≈ 10.91) than males —
  despite studying *more* on average and having *fewer* past failures. That gap, against
  the grain of their study effort, is a compelling point to foreground for SDG 5.

---

## Notes on changes from the original notebook

A few issues in the original notebook were corrected when turning it into this app
(see the project hand-off notes for details):

- The dropout target was redefined from the arbitrary `(G3 < 5) & (absences > 10)`
  rule — which produced almost no positive cases and a misleading 100% accuracy — to
  the standard **at-risk** definition `G3 < 10`, giving a meaningful, balanced classifier.
- `StandardScaler` was removed from the Random Forest pipelines: tree models are
  scale-invariant, so scaling added nothing.
- Preprocessing and the feature set were unified into a single, consistent pipeline.

---

*Built for educational purposes, in support of SDG 4 (Quality Education) and
SDG 5 (Gender Equality).*
