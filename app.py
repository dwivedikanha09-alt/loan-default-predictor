"""
Loan Default Predictor — Streamlit UI
Models: Gaussian Naive Bayes + K-Nearest Neighbours (trained in NaiveBayes_knn.ipynb)

Keep these files in the same folder as app.py:
    naive_bayes_model.pkl, knn_model.pkl, scaler.pkl
(Optional) loan_data.csv / "loan_data (1).csv" -> used only to list the real city / education / job options.

Run:  streamlit run app.py
"""
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
CAT_COLS = ["city", "education", "employment_type"]

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; }
h1, h2, h3, .serif { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }

.hero {
    padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem;
    background:
        radial-gradient(900px 300px at 100% 0%, rgba(201,162,75,.22), transparent 60%),
        linear-gradient(135deg, #12303f 0%, #0c1f2b 100%);
    border: 1px solid rgba(201,162,75,.28);
}
.hero h1 { color: #f5efe0; font-size: 2.5rem; margin: 0 0 .35rem 0; }
.hero p  { color: #a9bcc6; font-size: 1.02rem; margin: 0; max-width: 640px; line-height: 1.55; }

.panel-title { font-family: 'Fraunces', serif; font-size: 1.25rem; color: #e8d9b0; margin: .2rem 0 .6rem 0; }

.stat {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px; padding: .8rem 1rem;
}
.stat .v { font-family: 'Fraunces', serif; font-size: 1.5rem; color: #f5efe0; }
.stat .l { font-size: .8rem; color: #8fa5b1; }

.verdict { border-radius: 18px; padding: 1.1rem 1.3rem; margin-top: .2rem; }
.verdict.ok  { background: rgba(63,182,139,.12); border: 1px solid rgba(63,182,139,.5); }
.verdict.bad { background: rgba(228,99,90,.12);  border: 1px solid rgba(228,99,90,.5); }
.verdict .t  { font-family: 'Fraunces', serif; font-size: 1.5rem; }
.verdict.ok .t  { color: #5fd3a8; }
.verdict.bad .t { color: #ef8a82; }
.verdict .s  { color: #b7c6cf; font-size: .92rem; margin-top: .15rem; }

div[data-testid="stFormSubmitButton"] button {
    width: 100%; border-radius: 12px; padding: .7rem 1rem; font-weight: 700;
    background: linear-gradient(135deg, #c9a24b, #e2c274); color: #1a1405; border: none;
}
div[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.06); color: #1a1405; }
div[data-testid="stForm"] { border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 1.4rem; }
button[data-baseweb="tab"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
PKL_FILES = {
    "nb": "naive_bayes_model.pkl",
    "knn": "knn_model.pkl",
    "scaler": "scaler.pkl",
}
SEARCH_DIRS = [BASE, Path.cwd(), BASE / "models"]


def find_pkl(name):
    for d in SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    out = {}
    for key, name in PKL_FILES.items():
        path = find_pkl(name)
        if path is None:
            raise FileNotFoundError(name)
        with open(path, "rb") as f:
            out[key] = pickle.load(f)
    return out


def get_features(art):
    for obj in (art["scaler"], art["nb"], art["knn"]):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return []


@st.cache_data(show_spinner=False)
def get_categories(features):
    """Options for the dropdowns. Real names come from the CSV if it is present;
    otherwise we rebuild them from the model's one-hot columns (first category was dropped in training)."""
    df = None
    csv = next(iter(sorted(BASE.glob("loan_data*.csv"))), None)
    if csv is not None:
        try:
            df = pd.read_csv(csv).dropna()
        except Exception:
            df = None
    cats = {}
    for col in CAT_COLS:
        if df is not None and col in df.columns:
            cats[col] = sorted(df[col].astype(str).unique())
        else:
            dummies = [f[len(col) + 1:] for f in features if f.startswith(col + "_")]
            cats[col] = ["Other (baseline)"] + dummies
    return cats


def build_row(features, numeric, choices):
    row = {}
    for f in features:
        for col in CAT_COLS:
            if f.startswith(col + "_"):
                row[f] = int(choices[col] == f[len(col) + 1:])
                break
        else:
            row[f] = numeric[f]
    return pd.DataFrame([row])[features]


def approval_probability(art, model_name, X):
    if model_name == "Naive Bayes":
        proba = art["nb"].predict_proba(X)[0]
    else:
        proba = art["knn"].predict_proba(art["scaler"].transform(X))[0]
    return float(proba[0])  # class 0 = Approved, class 1 = Rejected (LabelEncoder order)


def gauge(p_approve, title):
    color = "#3fb68b" if p_approve >= 0.5 else "#e4635a"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(p_approve * 100, 1),
            number={"suffix": "%", "font": {"size": 44, "color": "#f5efe0"}},
            title={"text": title, "font": {"size": 15, "color": "#a9bcc6"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5d7583"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(228,99,90,.14)"},
                    {"range": [50, 100], "color": "rgba(63,182,139,.14)"},
                ],
                "threshold": {"line": {"color": "#c9a24b", "width": 3}, "thickness": 0.8, "value": 50},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"family": "Manrope"})
    return fig


def verdict_card(p_approve, model_name):
    approved = p_approve >= 0.5
    cls, title = ("ok", "Likely approved") if approved else ("bad", "Likely rejected")
    conf = p_approve if approved else 1 - p_approve
    st.markdown(
        f'<div class="verdict {cls}"><div class="t">{title}</div>'
        f'<div class="s">{model_name} is {conf * 100:.1f}% confident in this outcome.</div></div>',
        unsafe_allow_html=True,
    )


def stat(value, label):
    st.markdown(f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- boot
art = None
try:
    art = load_artifacts()
except FileNotFoundError:
    missing = [n for n in PKL_FILES.values() if find_pkl(n) is None]
    st.error(
        f"Missing model file(s): **{', '.join(missing)}**\n\n"
        f"`app.py` is running from: `{BASE}`\n\n"
        "Fix: copy the three `.pkl` files into that folder, or upload them below."
    )
    st.caption("Files found in that folder: " + (", ".join(sorted(p.name for p in BASE.iterdir())) or "none"))
    ups = {k: st.file_uploader(n, type="pkl", key=f"up_{k}") for k, n in PKL_FILES.items()}
    if all(ups.values()):
        try:
            art = {k: pickle.load(u) for k, u in ups.items()}
        except Exception as e:
            st.error(f"Could not read the uploaded files: {e}")
    if art is None:
        st.stop()
except Exception as e:  # version mismatch etc.
    st.error(f"Could not load the models: {e}\n\nInstall the same scikit-learn version you trained with.")
    st.stop()

features = get_features(art)
if not features:
    st.error("The saved models don't carry feature names. Re-train with a pandas DataFrame and save them again.")
    st.stop()
categories = get_categories(tuple(features))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🏦 Loan Default Predictor")
    mode = st.radio("Model", ["Naive Bayes", "K-Nearest Neighbours", "Compare both"], index=2)
    st.caption("Gaussian Naive Bayes runs on raw features. KNN (k = 18) runs on standardised features.")
    st.divider()
    st.caption("Built with scikit-learn and Streamlit. For learning and demos, not for real lending decisions.")

# ---------------------------------------------------------------- hero
st.markdown(
    """
<div class="hero">
  <h1>Will this loan be repaid?</h1>
  <p>Enter an applicant's details and two machine-learning models estimate the chance the loan gets approved.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_perf, tab_about = st.tabs(["Predict", "Model performance", "About"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    with st.form("applicant"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="panel-title">Applicant</div>', unsafe_allow_html=True)
            age = st.number_input("Age", 18, 80, 35)
            city = st.selectbox("City", categories["city"])
            education = st.selectbox("Education", categories["education"])
            employment = st.selectbox("Employment type", categories["employment_type"])
            has_property = st.toggle("Owns property", value=True)
        with right:
            st.markdown('<div class="panel-title">Finances</div>', unsafe_allow_html=True)
            annual_income = st.number_input("Annual income (₹)", 0, 100_000_000, 600_000, step=10_000)
            loan_amount = st.number_input("Loan amount (₹)", 0, 100_000_000, 300_000, step=10_000)
            monthly_emi = st.number_input("Monthly EMI (₹)", 0, 10_000_000, 15_000, step=500)
            credit_score = st.slider("Credit score", 300, 900, 750)
            existing_loans = st.number_input("Existing loans", 0, 20, 1)
        submitted = st.form_submit_button("Check loan outcome")

    if submitted:
        numeric = {
            "age": age,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "monthly_emi": monthly_emi,
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "has_property": int(has_property),
        }
        try:
            X = build_row(features, numeric, {"city": city, "education": education, "employment_type": employment})
        except KeyError as e:
            st.error(f"The model expects a feature this form doesn't collect: {e}")
            st.stop()

        st.markdown("&nbsp;")
        c1, c2, c3 = st.columns(3)
        with c1:
            stat(f"{(monthly_emi * 12 / annual_income * 100) if annual_income else 0:.0f}%", "Yearly EMI as share of income")
        with c2:
            stat(f"{(loan_amount / annual_income) if annual_income else 0:.2f}×", "Loan amount vs annual income")
        with c3:
            stat(str(credit_score), "Credit score")
        st.markdown("&nbsp;")

        names = ["Naive Bayes", "K-Nearest Neighbours"] if mode == "Compare both" else [mode]
        results = {n: approval_probability(art, n, X) for n in names}

        cols = st.columns(len(names), gap="large")
        for col, n in zip(cols, names):
            with col:
                st.plotly_chart(gauge(results[n], f"Approval probability — {n}"), use_container_width=True)
                verdict_card(results[n], n)

        if len(names) == 2:
            a, b = [r >= 0.5 for r in results.values()]
            st.markdown("&nbsp;")
            if a == b:
                st.success("Both models agree on this applicant.")
            else:
                st.warning("The models disagree. Treat this result as uncertain.")

        with st.expander("See the encoded features sent to the model"):
            st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)
    else:
        st.info("Fill in the form and press **Check loan outcome**.")

# ---------------------------------------------------------------- performance tab
with tab_perf:
    st.markdown("Scores on the held-out test set (33% of the data), taken from the notebook.")
    perf = pd.DataFrame(
        {
            "Accuracy": [0.449, 0.685],
            "Precision": [0.230, 0.204],
            "Recall": [0.672, 0.164],
        },
        index=["Naive Bayes", "KNN (k=18)"],
    )
    fig = go.Figure()
    palette = {"Accuracy": "#c9a24b", "Precision": "#4fa3c7", "Recall": "#3fb68b"}
    for metric in perf.columns:
        fig.add_bar(name=metric, x=perf.index, y=perf[metric], marker_color=palette[metric], text=perf[metric].map("{:.2f}".format), textposition="outside")
    fig.update_layout(
        barmode="group", height=380, yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,.08)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope", color="#c9d6dd"),
        margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)
    st.caption(
        "Precision for the Rejected class is low for both models (about 0.2), so many predicted rejections are false alarms. "
        "Naive Bayes catches more rejections (higher recall), KNN is more accurate overall."
    )

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        """
**Pipeline**

Load data → drop missing rows → one-hot encode city, education and employment type → 67/33 train-test split →
SMOTE on the training set to balance classes → train Gaussian Naive Bayes and KNN → evaluate → save as `.pkl`.

**How the app works**

The form values are converted to the same one-hot columns used in training, in the same order.
Naive Bayes receives them as they are; KNN receives them after the saved `StandardScaler`.
The approval probability is the model's probability for class 0 (Approved).

**Files this app needs**

`naive_bayes_model.pkl`, `knn_model.pkl`, `scaler.pkl`
"""
    )
