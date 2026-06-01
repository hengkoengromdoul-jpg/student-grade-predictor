"""
generate_sample_data.py
------------------------
Creates `sample_student_data.csv`: a SYNTHETIC stand-in for the real UCI
"Student Performance" (Math) dataset so the Streamlit app runs out of the box.

It reproduces the full 33-column schema and the key relationships the project
relies on (G1 -> G2 -> G3 chain, absences/failures drag grades down, mother's
education lifts them). It is NOT real data. For your actual report, download the
real dataset and save it as `student-mat.csv` in this folder. See README.md.
"""

import numpy as np
import pandas as pd

N = 395
RNG = np.random.default_rng(42)


def pick(values, probs, size=N):
    return RNG.choice(values, size=size, p=probs)


def build() -> pd.DataFrame:
    df = pd.DataFrame()

    # --- Demographics / school context ---
    df["school"] = pick(["GP", "MS"], [0.88, 0.12])
    df["sex"] = pick(["F", "M"], [0.527, 0.473])          # 208 F / 187 M in real data
    df["age"] = pick([15, 16, 17, 18, 19, 20, 21, 22],
                     [0.20, 0.27, 0.25, 0.20, 0.05, 0.02, 0.005, 0.005])
    df["address"] = pick(["U", "R"], [0.78, 0.22])
    df["famsize"] = pick(["GT3", "LE3"], [0.71, 0.29])
    df["Pstatus"] = pick(["T", "A"], [0.90, 0.10])

    # --- Parents ---
    df["Medu"] = pick([0, 1, 2, 3, 4], [0.01, 0.15, 0.26, 0.25, 0.33])
    df["Fedu"] = pick([0, 1, 2, 3, 4], [0.01, 0.20, 0.29, 0.25, 0.25])
    jobs = ["teacher", "health", "services", "at_home", "other"]
    df["Mjob"] = pick(jobs, [0.14, 0.09, 0.26, 0.15, 0.36])
    df["Fjob"] = pick(jobs, [0.07, 0.05, 0.28, 0.05, 0.55])

    # --- School choice / logistics ---
    df["reason"] = pick(["home", "reputation", "course", "other"],
                        [0.28, 0.27, 0.37, 0.08])
    df["guardian"] = pick(["mother", "father", "other"], [0.69, 0.23, 0.08])
    df["traveltime"] = pick([1, 2, 3, 4], [0.65, 0.27, 0.06, 0.02])
    df["studytime"] = pick([1, 2, 3, 4], [0.26, 0.50, 0.16, 0.08])
    df["failures"] = pick([0, 1, 2, 3], [0.79, 0.13, 0.05, 0.03])

    # --- Binary support / lifestyle flags ---
    df["schoolsup"] = pick(["yes", "no"], [0.13, 0.87])
    df["famsup"] = pick(["yes", "no"], [0.615, 0.385])    # 243 yes / 153 no
    df["paid"] = pick(["yes", "no"], [0.46, 0.54])
    df["activities"] = pick(["yes", "no"], [0.51, 0.49])
    df["nursery"] = pick(["yes", "no"], [0.795, 0.205])
    df["higher"] = pick(["yes", "no"], [0.95, 0.05])
    df["internet"] = pick(["yes", "no"], [0.832, 0.168])
    df["romantic"] = pick(["yes", "no"], [0.33, 0.67])

    # --- Relationships / time / health (1-5 Likert-ish) ---
    df["famrel"] = pick([1, 2, 3, 4, 5], [0.02, 0.05, 0.17, 0.49, 0.27])
    df["freetime"] = pick([1, 2, 3, 4, 5], [0.05, 0.16, 0.39, 0.29, 0.11])
    df["goout"] = pick([1, 2, 3, 4, 5], [0.06, 0.26, 0.33, 0.22, 0.13])
    df["Dalc"] = pick([1, 2, 3, 4, 5], [0.70, 0.18, 0.07, 0.03, 0.02])
    df["Walc"] = pick([1, 2, 3, 4, 5], [0.38, 0.21, 0.20, 0.13, 0.08])
    df["health"] = pick([1, 2, 3, 4, 5], [0.12, 0.11, 0.23, 0.16, 0.38])

    # --- Absences (right-skewed) ---
    df["absences"] = np.clip(RNG.poisson(4, N) + RNG.integers(0, 6, N), 0, 75)

    # --- Build grades from a latent "ability" so correlations are realistic ---
    studytime = df["studytime"].to_numpy()
    failures = df["failures"].to_numpy()
    medu = df["Medu"].to_numpy()
    absences = df["absences"].to_numpy()
    famsup = (df["famsup"] == "yes").astype(int).to_numpy()
    goout = df["goout"].to_numpy()

    # Latent ability drives the grade chain so correlations mirror the real
    # dataset (G2->G3 ~0.89, failures/absences drag down, Medu lifts up).
    ability = (
        9.6
        + 0.85 * studytime
        - 1.9 * failures
        + 0.55 * medu
        + 0.35 * famsup
        - 0.06 * absences
        - 0.25 * goout
        + RNG.normal(0, 1.25, N)
    )

    g1 = np.clip(np.round(ability + RNG.normal(0, 0.8, N)), 0, 20)
    g2 = np.clip(np.round(0.76 * g1 + 0.30 * ability + RNG.normal(0, 0.55, N)), 0, 20)
    g3 = np.clip(np.round(0.92 * g2 + 0.11 * g1 - 0.035 * absences
                          + RNG.normal(0, 0.45, N)), 0, 20)

    # Inject a few dropouts (G3 == 0) among the lowest-ability students,
    # as in the real data, so they remain broadly predictable.
    low = np.argsort(ability)[: int(0.15 * N)]
    drop_idx = RNG.choice(low, size=int(0.035 * N), replace=False)
    g3[drop_idx] = 0

    df["G1"] = g1.astype(int)
    df["G2"] = g2.astype(int)
    df["G3"] = g3.astype(int)

    # Column order identical to the real UCI file
    order = [
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
        "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
        "higher", "internet", "romantic", "famrel", "freetime", "goout",
        "Dalc", "Walc", "health", "absences", "G1", "G2", "G3",
    ]
    return df[order]


if __name__ == "__main__":
    data = build()
    data.to_csv("sample_student_data.csv", index=False)
    print(f"Wrote sample_student_data.csv  ({data.shape[0]} rows x {data.shape[1]} cols)")
    print(data[["sex", "studytime", "failures", "absences", "G1", "G2", "G3"]].head())
