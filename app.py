import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="NeevCraft | ICU Sepsis Sentinel",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
        color: #1f2d3d;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dce4ed;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce4ed;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 16px #1c355012;
    }
    [data-testid="stMetricLabel"] { color: #5b6b7d; }
    [data-testid="stMetricValue"] { color: #142b43; }
    div[data-testid="stTabs"] button { color: #344b62; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #087f8c;
        border-bottom-color: #087f8c;
    }
    .clinical-note {
        padding: 12px 16px;
        border: 1px solid #c9e3e7;
        border-left: 4px solid #078b98;
        border-radius: 10px;
        background: #eef9fa;
        color: #294659;
        margin: 8px 0 18px;
    }
    .section-caption { color: #607286; font-size: 0.92rem; }
    h1, h2, h3 { color: #16324c; }
    [data-testid="stDataFrame"] { border: 1px solid #dce4ed; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_data():
    return pd.read_csv("train.csv")


df = load_sample_data()
required_columns = {"Patient_ID", "Hour", "SepsisLabel"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
    st.stop()

st.title("🏥 ICU Sepsis Sentinel")
st.caption("Critical-care patient trends and screening indicators")
st.markdown(
    '<div class="clinical-note"><strong>Clinical decision support prototype:</strong> '
    "Screening indicators and reference bands are informational only—not a diagnosis, "
    "validated prediction, or substitute for clinician assessment. Confirm results "
    "against local protocols and the patient's complete record.</div>",
    unsafe_allow_html=True,
)


def number(value, digits=1):
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def latest_flag(value):
    return pd.notna(value) and float(value) == 1.0


def status_for_range(value, low, high):
    if pd.isna(value):
        return "Unavailable"
    return "Within reference" if low <= float(value) <= high else "Outside reference"


patient_ids = sorted(df["Patient_ID"].dropna().unique(), key=float)
if not patient_ids:
    st.error("No patient records are available in train.csv.")
    st.stop()

st.sidebar.header("Patient navigator")
st.sidebar.caption(f"{len(patient_ids):,} unique patients available")
selected_patient = st.sidebar.selectbox(
    "Select patient ID",
    patient_ids,
    format_func=lambda patient_id: f"Patient {int(patient_id)}",
)

patient_data = (
    df.loc[df["Patient_ID"] == selected_patient]
    .sort_values("Hour")
    .reset_index(drop=True)
)
if patient_data.empty:
    st.error(f"No observations found for patient {selected_patient}.")
    st.stop()
latest_record = patient_data.iloc[-1]
latest_hour = latest_record["Hour"]

gender = latest_record.get("Gender")
gender_label = (
    "Female"
    if pd.notna(gender) and float(gender) == 0
    else "Male"
    if pd.notna(gender) and float(gender) == 1
    else "Not recorded"
)
unit1 = latest_record.get("Unit1")
unit2 = latest_record.get("Unit2")
if latest_flag(unit1):
    unit_label = "Unit 1"
elif latest_flag(unit2):
    unit_label = "Unit 2"
else:
    unit_label = "Not recorded"

age = number(latest_record.get("Age"), 0)
iculos = latest_record.get("ICULOS")
iculos_hours = number(iculos, 0)
iculos_days = number(float(iculos) / 24, 1) if pd.notna(iculos) else "—"
admission_offset = latest_record.get("HospAdmTime")
admission_display = f"{number(admission_offset)} h" if pd.notna(admission_offset) else "—"
admission_help = (
    "Hours relative to hospital admission; negative values indicate the ICU record "
    "is after hospital admission."
)
positive_label_hours = patient_data.loc[
    patient_data["SepsisLabel"] == 1, "Hour"
].dropna()
first_positive_hour = positive_label_hours.min() if not positive_label_hours.empty else np.nan
observed_columns = [
    column
    for column in ("HR", "MAP", "O2Sat", "Temp", "Resp", "WBC", "Platelets", "Lactate")
    if column in patient_data.columns
]
latest_coverage = (
    int(latest_record[observed_columns].notna().sum()) if observed_columns else 0
)
coverage_percent = round(100 * latest_coverage / len(observed_columns)) if observed_columns else 0

st.subheader(f"Patient {int(selected_patient)} · latest observation at ICU hour {number(latest_hour, 0)}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age / gender", f"{age} yr · {gender_label}")
col2.metric("ICU unit", unit_label)
col3.metric("ICU length of stay", f"{iculos_hours} h", help=f"Approximately {iculos_days} days")
col4.metric("Hospital admission offset", admission_display, help=admission_help)

timeline1, timeline2, timeline3 = st.columns(3)
timeline1.metric("Recorded observations", f"{len(patient_data):,}", help="Number of hourly records available for this patient")
timeline2.metric(
    "First positive sepsis label",
    f"ICU hour {number(first_positive_hour, 0)}" if pd.notna(first_positive_hour) else "None in records",
    help="Historical dataset label only; not an alert or prediction.",
)
timeline3.metric(
    "Latest data coverage",
    f"{latest_coverage}/{len(observed_columns)} signals · {coverage_percent}%",
    help="Availability of HR, MAP, O₂ saturation, temperature, respiration, WBC, platelets, and lactate at the latest observation.",
)

label_at_latest = latest_record["SepsisLabel"]
if pd.isna(label_at_latest):
    st.info("Latest-hour sepsis label is unavailable in this record.")
elif float(label_at_latest) == 1.0:
    st.error("🔴 SepsisLabel is positive at the latest recorded hour. This is a dataset label, not a real-time prediction.")
else:
    st.success("🟢 SepsisLabel is negative at the latest recorded hour. This does not rule out clinical deterioration.")

st.divider()

tab_vitals, tab_labs, tab_risk = st.tabs(
    ["📈 Vital trends", "🧪 Laboratory panel", "🚦 Screening & criteria"]
)

with tab_vitals:
    st.subheader("Vitals over ICU stay")
    st.markdown(
        '<p class="section-caption">Separate scales keep each signal readable. '
        "Shaded bands are general reference ranges, not patient-specific targets.</p>",
        unsafe_allow_html=True,
    )
    vital_specs = [
        ("HR", "Heart rate", "bpm", 60, 100, "#f06b74"),
        ("MAP", "Mean arterial pressure", "mmHg", 65, 100, "#55c2a5"),
        ("O2Sat", "Oxygen saturation", "%", 92, 100, "#58a9ef"),
        ("Temp", "Temperature", "°C", 36, 38, "#edb85c"),
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[spec[1] for spec in vital_specs],
        vertical_spacing=0.17,
        horizontal_spacing=0.1,
    )
    for index, (column, label, unit, low, high, color) in enumerate(vital_specs):
        row, col = index // 2 + 1, index % 2 + 1
        if column in patient_data.columns:
            series = patient_data[["Hour", column]].dropna()
            if not series.empty:
                figure.add_trace(
                    go.Scatter(
                        x=series["Hour"],
                        y=series[column],
                        mode="lines+markers",
                        name=label,
                        line={"color": color, "width": 2.5},
                        marker={"size": 5},
                        hovertemplate=f"Hour %{{x}}<br>{label}: %{{y:.1f}} {unit}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                figure.add_hrect(
                    y0=low,
                    y1=high,
                    fillcolor="#59c6a1",
                    opacity=0.09,
                    line_width=0,
                    row=row,
                    col=col,
                )
            else:
                figure.add_annotation(
                    text="No measurements available",
                    x=0.5,
                    y=0.5,
                    xref=f"x{index + 1} domain" if index else "x domain",
                    yref=f"y{index + 1} domain" if index else "y domain",
                    showarrow=False,
                    font={"color": "#607286"},
                )
        figure.update_yaxes(
            title_text=unit,
            row=row,
            col=col,
            gridcolor="#e3eaf1",
            zerolinecolor="#d3dde7",
        )
        figure.update_xaxes(
            title_text="ICU hour",
            row=row,
            col=col,
            gridcolor="#e3eaf1",
            zerolinecolor="#d3dde7",
        )
    figure.update_layout(
        height=610,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#344b62"},
        margin={"t": 55, "b": 25, "l": 20, "r": 20},
    )
    st.plotly_chart(figure, width="stretch")

with tab_labs:
    st.subheader("Biomarker history")
    st.markdown(
        '<p class="section-caption">Reference intervals are illustrative and may vary '
        "by laboratory, patient, and treatment context. Missing observations are not imputed.</p>",
        unsafe_allow_html=True,
    )
    lab_specs = {
        "WBC": ("White blood cells", "10³/µL", 4, 11),
        "Platelets": ("Platelets", "10³/µL", 150, 450),
        "Lactate": ("Lactate", "mmol/L", 0.5, 2),
        "Creatinine": ("Creatinine", "mg/dL", 0.6, 1.3),
        "Glucose": ("Glucose", "mg/dL", 70, 140),
    }
    lab_columns = st.columns(len(lab_specs))
    for card, (column, (label, unit, low, high)) in zip(lab_columns, lab_specs.items()):
        value = latest_record.get(column, np.nan)
        state = status_for_range(value, low, high)
        card.metric(label, f"{number(value)} {unit}" if pd.notna(value) else "—", state)

    selected_lab = st.selectbox(
        "Explore biomarker trend",
        list(lab_specs),
        format_func=lambda column: lab_specs[column][0],
    )
    label, unit, low, high = lab_specs[selected_lab]
    lab_history = patient_data[["Hour", selected_lab]].dropna()
    lab_figure = go.Figure()
    if not lab_history.empty:
        lab_figure.add_hrect(
            y0=low,
            y1=high,
            fillcolor="#59c6a1",
            opacity=0.13,
            line_width=0,
            annotation_text="Illustrative reference interval",
            annotation_position="top left",
        )
        lab_figure.add_trace(
            go.Scatter(
                x=lab_history["Hour"],
                y=lab_history[selected_lab],
                mode="lines+markers",
                name=label,
                line={"color": "#61b8ed", "width": 2.5},
                marker={"size": 6},
                hovertemplate=f"Hour %{{x}}<br>{label}: %{{y:.2f}} {unit}<extra></extra>",
            )
        )
        lab_figure.update_layout(
            xaxis_title="ICU hour",
            yaxis_title=unit,
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#344b62"},
            xaxis={"gridcolor": "#e3eaf1"},
            yaxis={"gridcolor": "#e3eaf1"},
            showlegend=False,
        )
        st.plotly_chart(lab_figure, width="stretch")
    else:
        st.info(f"No {label.lower()} measurements are available for this patient.")

    with st.expander("Show latest biomarker values and reference intervals"):
        table_rows = []
        for column, (name, units, lower, upper) in lab_specs.items():
            value = latest_record.get(column, np.nan)
            table_rows.append(
                {
                    "Biomarker": name,
                    "Latest value": number(value),
                    "Unit": units,
                    "Illustrative range": f"{lower}–{upper}",
                    "Status": status_for_range(value, lower, upper),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

with tab_risk:
    st.subheader("Early-warning screening indicators")
    st.markdown(
        '<div class="clinical-note"><strong>Not a validated risk probability.</strong> '
        "The index below is a transparent count of available SIRS-style screening "
        "signals. It does not diagnose sepsis, and missing values are not treated as normal.</div>",
        unsafe_allow_html=True,
    )

    criteria = [
        ("Temperature", latest_record.get("Temp"), lambda value: value < 36 or value > 38, "≤36 or >38 °C"),
        ("Heart rate", latest_record.get("HR"), lambda value: value > 90, ">90 bpm"),
        ("Respiratory rate", latest_record.get("Resp"), lambda value: value > 20, ">20 breaths/min"),
        ("WBC", latest_record.get("WBC"), lambda value: value < 4 or value > 12, "≤4 or ≥12 ×10³/µL"),
    ]
    known_positive = 0
    known_count = 0
    criteria_rows = []
    for name, value, is_positive, threshold in criteria:
        if pd.isna(value):
            result = "Unknown — no measurement"
        else:
            known_count += 1
            positive = is_positive(float(value))
            known_positive += int(positive)
            result = "Signal present" if positive else "No signal"
        criteria_rows.append(
            {
                "SIRS-style component": name,
                "Latest value": number(value),
                "Screening threshold": threshold,
                "Status": result,
            }
        )
    screening_index = known_positive * 25

    gauge_col, detail_col = st.columns([1, 1.6])
    with gauge_col:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=screening_index,
                number={"suffix": "/100", "font": {"color": "#17324b", "size": 36}},
                title={"text": "SIRS-style signal index<br><sup>Not a probability</sup>"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#52667a"},
                    "bar": {"color": "#f06b74" if known_positive >= 2 else "#edb85c"},
                    "bgcolor": "#ffffff",
                    "bordercolor": "#dce4ed",
                    "steps": [
                        {"range": [0, 25], "color": "#e5f4ee"},
                        {"range": [25, 50], "color": "#fff2d8"},
                        {"range": [50, 100], "color": "#fbe5e7"},
                    ],
                    "threshold": {
                        "line": {"color": "#ff6675", "width": 4},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        gauge.update_layout(
            height=280,
            margin={"t": 85, "b": 20, "l": 25, "r": 25},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#344b62"},
        )
        st.plotly_chart(gauge, width="stretch")
        st.caption(
            f"{known_positive} of {known_count} measured components show a signal; "
            f"{len(criteria) - known_count} component(s) unavailable."
        )
    with detail_col:
        if known_positive >= 2:
            st.error("🔴 Multiple SIRS-style signals present — prompt clinical review per local protocol.")
        elif known_positive == 1:
            st.warning("🟠 One SIRS-style signal present — interpret with the full clinical picture.")
        elif known_count:
            st.success("🟢 No measured SIRS-style signals at this observation.")
        else:
            st.info("Screening status unknown: no required component is available.")
        st.dataframe(pd.DataFrame(criteria_rows), hide_index=True, width="stretch")

    st.markdown("#### SOFA-relevant components available in this dataset")
    sofa_components = [
        ("MAP", latest_record.get("MAP"), "mmHg", lambda value: value < 70),
        ("Platelets", latest_record.get("Platelets"), "10³/µL", lambda value: value < 150),
        ("Bilirubin total", latest_record.get("Bilirubin_total"), "mg/dL", lambda value: value > 1.2),
        ("Creatinine", latest_record.get("Creatinine"), "mg/dL", lambda value: value > 1.2),
    ]
    sofa_rows = []
    for name, value, units, abnormal in sofa_components:
        state = (
            "Unavailable"
            if pd.isna(value)
            else "Threshold signal"
            if abnormal(float(value))
            else "No threshold signal"
        )
        sofa_rows.append(
            {"Component": name, "Latest value": number(value), "Unit": units, "Status": state}
        )
    st.dataframe(pd.DataFrame(sofa_rows), hide_index=True, width="stretch")
    st.caption(
        "A full SOFA score cannot be calculated from these fields: oxygenation ratio, "
        "vasopressor use, Glasgow Coma Scale, and urine output are not available. "
        "This component view is not a SOFA score."
    )
