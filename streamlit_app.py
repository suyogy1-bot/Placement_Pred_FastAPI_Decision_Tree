"""Placement predictor: a Streamlit front end for placement_model.joblib.

Run:  streamlit run streamlit_app.py
Keep placement_model.joblib and columns.joblib in the same folder as this file.
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Placement predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HERE = Path(__file__).parent
BRANCHES = ["CSE", "IT", "ECE", "Mechanical", "Civil", "Commerce"]

# Held-out test results from Decision_Tree.ipynb (80/20 split, random_state=42).
MODEL_FACTS = {"accuracy": "81.5%", "precision": "83.8%", "recall": "77.7%", "rows": "8,139"}

PRESETS = {
    "Average student": dict(study=3.5, sleep=6.8, net=4.5, att=76.0, asg=13, prev=62.0, exam=60.0, extra=False),
    "High achiever": dict(study=5.5, sleep=7.2, net=3.0, att=92.0, asg=19, prev=85.0, exam=78.0, extra=True),
    "At risk": dict(study=1.2, sleep=5.2, net=7.5, att=55.0, asg=5, prev=38.0, exam=42.0, extra=False),
}
DEFAULTS = dict(PRESETS["Average student"], gender="Female", branch="CSE")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(HERE / "placement_model.joblib"), joblib.load(HERE / "columns.joblib")


def encode(profile: dict, columns: list) -> pd.DataFrame:
    """Same encoding the notebook used: LabelEncoder (alphabetical) + one-hot branch."""
    row = {
        "gender": 1 if profile["gender"] == "Male" else 0,  # Female=0, Male=1
        "extracurricular": 1 if profile["extra"] else 0,  # No=0, Yes=1
        "study_hours": profile["study"],
        "attendance": profile["att"],
        "sleep_hours": profile["sleep"],
        "internet_usage": profile["net"],
        "assignments_completed": profile["asg"],
        "previous_score": profile["prev"],
        "exam_score": profile["exam"],
    }
    for b in BRANCHES:
        row[f"branch_{b.upper()}"] = 1 if profile["branch"] == b else 0
    return pd.DataFrame([row]).reindex(columns=columns, fill_value=0)


# ---------------------------------------------------------------- state
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ---------------------------------------------------------------- style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --night:#0D1126; --slate:#151A33; --slate-2:#1C2242; --line:#2B3260;
  --text:#EEF0FF; --muted:#A3ABD0; --iris:#8087FF; --iris-dim:#3B3F8F;
  --paper:#F6F5FF; --ink:#12142B; --ink-soft:#4B4F78;
  --mint:#1F9D6B; --rose:#D6455A; --amber:#B7791F;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; color:var(--text); }
.stApp { background:
   radial-gradient(1200px 500px at 85% -10%, rgba(128,135,255,.16), transparent 60%),
   var(--night); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { max-width:1180px; padding:2.4rem 1.5rem 4rem; }

/* ---------- masthead ---------- */
.mast { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem;
        padding:.5rem 0 2rem; border-bottom:1px solid var(--line); margin-bottom:2rem; position:relative; overflow:hidden; }
.mast h1 { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:clamp(2.2rem,5vw,3.6rem);
           line-height:1.02; letter-spacing:-.035em; margin:0; color:var(--text); max-width:11ch; }
.mast p { color:var(--muted); font-size:1.02rem; line-height:1.6; max-width:38ch; margin:0 0 .35rem; }
.mast .pct { position:absolute; right:-.6rem; top:-2.4rem; font-family:'Space Grotesk',sans-serif; font-weight:700;
             font-size:15rem; line-height:1; color:transparent; -webkit-text-stroke:1.5px var(--iris-dim);
             opacity:.55; pointer-events:none; user-select:none; }
.mast > div { position:relative; z-index:1; }
@media (max-width:820px){ .mast{flex-direction:column; align-items:flex-start;} .mast .pct{font-size:9rem; top:-1rem;} }

/* ---------- form groups ---------- */
.group-title { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.12rem; letter-spacing:-.01em;
               margin:0 0 .15rem; color:var(--text); }
.group-sub { color:var(--muted); font-size:.88rem; margin:0 0 .6rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--slate); border:1px solid var(--line) !important;
               border-radius:18px !important; padding:.35rem .5rem .2rem; }

/* widgets */
[data-testid="stWidgetLabel"] p, label p { color:var(--text) !important; font-size:.9rem !important; font-weight:500 !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color:var(--muted) !important; font-size:.76rem !important; }
[data-testid="stSliderThumbValue"] { color:var(--text) !important; font-weight:600; font-size:.88rem; }
[data-baseweb="select"] > div { background:var(--slate-2) !important; border:1px solid var(--line) !important; border-radius:12px !important; color:var(--text) !important; }
[data-baseweb="select"] svg { fill:var(--muted) !important; }
div[role="radiogroup"] label { background:var(--slate-2); border:1px solid var(--line); border-radius:12px; padding:.45rem .95rem; margin-right:.4rem; }
div[role="radiogroup"] label:has(input:checked) { border-color:var(--iris); background:rgba(128,135,255,.16); }
div[role="radiogroup"] label > div:first-child { display:none; }
[data-baseweb="popover"] li { background:var(--slate-2) !important; color:var(--text) !important; }
:focus-visible { outline:2px solid var(--iris) !important; outline-offset:2px; }

/* preset buttons */
.stButton > button { background:transparent; color:var(--text); border:1px solid var(--line); border-radius:999px;
                     padding:.35rem 1rem; font-size:.86rem; font-weight:500; transition:border-color .15s, background .15s; }
.stButton > button:hover { border-color:var(--iris); background:rgba(128,135,255,.12); color:var(--text); }
.preset-label { color:var(--muted); font-size:.88rem; margin:.2rem 0 .4rem; }

/* ---------- sticky result column ---------- */
div[data-testid="stColumn"]:last-child { position:sticky; top:1.25rem; align-self:flex-start; }

/* ---------- the offer slip ---------- */
.slip { background:var(--paper); color:var(--ink); border-radius:22px; position:relative; overflow:hidden;
        box-shadow:0 30px 60px -28px rgba(0,0,0,.7); }
.slip-top { padding:1.6rem 1.7rem 1.4rem; position:relative; }
.slip-kicker { font-size:.9rem; color:var(--ink-soft); font-weight:500; margin:0; }
.slip-num { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:6.2rem; line-height:.95; letter-spacing:-.05em; margin:.35rem 0 .2rem; }
.slip-num small { font-size:2.4rem; letter-spacing:-.02em; margin-left:.1rem; color:var(--ink-soft); }
.slip-verdict { font-size:1rem; font-weight:600; margin:.2rem 0 0; }
.slip-note { font-size:.84rem; color:var(--ink-soft); margin:.25rem 0 0; }
.meter { position:relative; height:10px; border-radius:99px; background:#DAD8F2; margin:1.2rem 0 .3rem; }
.meter > i { position:absolute; left:0; top:0; bottom:0; border-radius:99px; transition:width .45s cubic-bezier(.2,.8,.2,1); }
.meter > b { position:absolute; left:50%; top:-5px; bottom:-5px; width:2px; background:var(--ink); opacity:.55; }
.meter-scale { display:flex; justify-content:space-between; font-size:.74rem; color:var(--ink-soft); }
.stamp { position:absolute; right:1.3rem; top:1.5rem; transform:rotate(-9deg); border:3px double currentColor; border-radius:14px;
         padding:.35rem .8rem; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; letter-spacing:.02em;
         text-align:center; line-height:1.15; opacity:.92; }
.perf { position:relative; border-top:2px dashed #C3C0E6; margin:0 1.2rem; }
.perf::before, .perf::after { content:""; position:absolute; top:-13px; width:24px; height:24px; border-radius:50%; background:var(--night); }
.perf::before { left:-2.2rem; } .perf::after { right:-2.2rem; }
.slip-bottom { padding:1.2rem 1.7rem 1.5rem; }
.slip-bottom h3 { font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600; margin:0 0 .15rem; letter-spacing:-.01em; }
.slip-bottom .hint { font-size:.82rem; color:var(--ink-soft); margin:0 0 .85rem; }
.sig { display:grid; grid-template-columns:1fr auto; gap:.15rem .8rem; align-items:baseline; margin-bottom:.7rem; }
.sig .name { font-size:.9rem; font-weight:500; }
.sig .val { font-size:.88rem; font-weight:600; font-variant-numeric:tabular-nums; }
.sig .bar { grid-column:1 / -1; height:6px; border-radius:99px; background:#E4E2F7; overflow:hidden; }
.sig .bar > i { display:block; height:100%; background:var(--iris-dim); border-radius:99px; }
.fine { color:var(--muted); font-size:.82rem; line-height:1.6; margin-top:1rem; }
.fine b { color:var(--text); font-weight:600; }

@media (prefers-reduced-motion:reduce){ *{ transition:none !important; animation:none !important; } }
@media (max-width:900px){ div[data-testid="stColumn"]:last-child{ position:static; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- masthead
st.markdown(
    """
<div class="mast">
  <span class="pct" aria-hidden="true">%</span>
  <div><h1>Will this student get placed?</h1></div>
  <div><p>Describe the student on the left. The model reads the profile and updates the estimate as you change it.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    model, columns = load_model()
except Exception as exc:  # missing files, version mismatch...
    st.error(
        "Couldn't load the model. Keep placement_model.joblib and columns.joblib "
        f"in the same folder as this app.\n\nDetails: {exc}"
    )
    st.stop()

# ---------------------------------------------------------------- layout
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<p class="preset-label">Start from an example</p>', unsafe_allow_html=True)
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), use_container_width=True)
    st.write("")

    with st.container(border=True):
        st.markdown('<p class="group-title">About the student</p><p class="group-sub">Who they are and what they do outside class.</p>', unsafe_allow_html=True)
        st.radio("Gender", ["Female", "Male"], key="gender", horizontal=True)
        a, b = st.columns(2)
        a.selectbox("Branch", BRANCHES, key="branch")
        with b:
            st.write("")
            st.toggle("Takes part in extracurriculars", key="extra")

    with st.container(border=True):
        st.markdown('<p class="group-title">Daily habits</p><p class="group-sub">Average hours per day.</p>', unsafe_allow_html=True)
        st.slider("Study hours", 0.0, 12.0, step=0.1, key="study")
        a, b = st.columns(2)
        a.slider("Sleep hours", 3.0, 12.0, step=0.1, key="sleep")
        b.slider("Internet hours", 0.0, 12.0, step=0.1, key="net")

    with st.container(border=True):
        st.markdown('<p class="group-title">Academic record</p><p class="group-sub">Scores and attendance on a 0 to 100 scale.</p>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.slider("Attendance (%)", 0.0, 100.0, step=0.5, key="att")
        b.slider("Assignments completed", 0, 20, key="asg")
        a, b = st.columns(2)
        a.slider("Previous score", 0.0, 100.0, step=0.5, key="prev")
        b.slider("Exam score", 0.0, 100.0, step=0.5, key="exam")

# ---------------------------------------------------------------- predict
ss = st.session_state
profile = {k: ss[k] for k in ["gender", "branch", "extra", "study", "sleep", "net", "att", "asg", "prev", "exam"]}
proba = float(model.predict_proba(encode(profile, columns))[0][1])
placed = proba >= 0.5
pct = round(proba * 100)
close_call = abs(proba - 0.5) < 0.10

color = "var(--mint)" if placed else "var(--rose)"
if close_call:
    color = "var(--amber)"
headline = "Likely to be placed" if placed else "Unlikely to be placed"
stamp = "PLACED" if placed else "NOT YET"
note = (
    "Close call. A small change in the profile could flip this."
    if close_call
    else ("Strong match with students who were placed." if placed else "Closer to students who were not placed.")
)

# What the model looks at: importance, with branch columns merged into one factor.
imp = dict(zip(columns, model.feature_importances_))
branch_total = sum(v for k, v in imp.items() if k.startswith("branch_"))
factors = {
    "Exam score": (imp["exam_score"], f"{ss['exam']:g}"),
    "Extracurriculars": (imp["extracurricular"], "Yes" if ss["extra"] else "No"),
    "Previous score": (imp["previous_score"], f"{ss['prev']:g}"),
    "Assignments completed": (imp["assignments_completed"], f"{ss['asg']}"),
}
shown = sum(v for v, _ in factors.values())
rest = max(0.0, 1 - shown)
top = max(v for v, _ in factors.values())

sig_html = "".join(
    f'<div class="sig"><span class="name">{n}</span><span class="val">{val}</span>'
    f'<div class="bar"><i style="width:{max(v / top * 100, 2):.0f}%"></i></div></div>'
    for n, (v, val) in factors.items()
)
sig_html += (
    f'<div class="sig"><span class="name">Everything else</span><span class="val">{rest * 100:.0f}% of the signal</span>'
    f'<div class="bar"><i style="width:{max(rest / top * 100, 2):.0f}%"></i></div></div>'
)

with right:
    st.markdown(
        f"""
<div class="slip" role="status" aria-live="polite">
  <div class="slip-top">
    <div class="stamp" style="color:{color}">{stamp}</div>
    <p class="slip-kicker">Chance of placement</p>
    <div class="slip-num">{pct}<small>%</small></div>
    <p class="slip-verdict" style="color:{color}">{headline}</p>
    <p class="slip-note">{note}</p>
    <div class="meter"><i style="width:{max(pct, 2)}%; background:{color}"></i><b></b></div>
    <div class="meter-scale"><span>0%</span><span>50% line</span><span>100%</span></div>
  </div>
  <div class="perf"></div>
  <div class="slip-bottom">
    <h3>What the model weighs most</h3>
    <p class="hint">Share of the model's decisions, next to this student's value.</p>
    {sig_html}
  </div>
</div>
<p class="fine">
  <b>About this estimate.</b> A decision tree trained on {MODEL_FACTS['rows']} cleaned student records.
  On held-out data it was right {MODEL_FACTS['accuracy']} of the time (precision {MODEL_FACTS['precision']}, recall {MODEL_FACTS['recall']}).
  Trees respond in steps, so a few points on one score can move the result sharply. Treat it as a guide, not a verdict.
</p>
""",
        unsafe_allow_html=True,
    )
