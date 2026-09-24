"""
Household Energy Consumption & Sustainability Analytics Dashboard
=================================================================
Single-file Streamlit app combining:
  - Data loading & cleaning (Steps 1-2)
  - KPI engine (Step 2)
  - Trend analysis (Step 3)
  - Driver analysis (Step 4)
  - Risk & opportunity analysis (Step 5)
  - ML forecast model (Step 6)
  - 3-page dashboard UI (Step 7)

Dataset: UCI Household Electric Power Consumption
Source  : https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
ELECTRICITY_RATE_GBP  = 0.28   # £ per kWh (UK average 2024)
CO2_FACTOR_KG_KWH     = 0.233  # kg CO₂ per kWh (UK grid, 2023)
PEAK_HOURS            = list(range(7, 10)) + list(range(17, 21))  # 07-09, 17-20
SOLAR_PANEL_OUTPUT_KWH_YEAR = 3400  # avg UK 4kWp system annual output
SOLAR_INSTALL_COST_GBP      = 6500  # avg UK cost

COLOUR_PRIMARY   = "#1f77b4"
COLOUR_WARNING   = "#d62728"
COLOUR_SUCCESS   = "#2ca02c"
COLOUR_ACCENT    = "#ff7f0e"
COLOUR_NEUTRAL   = "#7f7f7f"

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring",  4: "Spring",  5: "Spring",
           6: "Summer",  7: "Summer",  8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Household Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── GLOBAL BACKGROUND: deep navy-to-indigo gradient ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #24243e 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* ── SIDEBAR: dark glass panel ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a4e 0%, #0f0c29 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e0 !important;
    font-size: 0.95rem;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── MAIN CONTENT TEXT ── */
.stApp, .stApp p, .stApp li, .stApp span,
.stApp label, .stApp div {
    color: #e2e8f0;
}
h1, h2, h3 { color: #f7fafc !important; }

/* ── METRIC CARDS: glass morphism ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.05) 100%);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.12);
    transition: transform 0.15s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.15);
}
div[data-testid="metric-container"] label {
    color: #90cdf4 !important;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f7fafc !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #68d391 !important;
    font-size: 0.82rem !important;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #90cdf4 !important;
    margin-bottom: 10px;
    padding: 8px 14px;
    background: linear-gradient(90deg, rgba(66,153,225,0.18) 0%, transparent 100%);
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    letter-spacing: 0.02em;
}

/* ── RISK BADGES ── */
.risk-high   { background: linear-gradient(90deg,#fc4747,#e53e3e); color:#fff; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.8rem; letter-spacing:0.05em; box-shadow:0 2px 8px rgba(229,62,62,0.4); }
.risk-medium { background: linear-gradient(90deg,#f6ad55,#ed8936); color:#1a202c; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.8rem; letter-spacing:0.05em; box-shadow:0 2px 8px rgba(237,137,54,0.4); }
.risk-low    { background: linear-gradient(90deg,#48bb78,#38a169); color:#fff; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.8rem; letter-spacing:0.05em; box-shadow:0 2px 8px rgba(56,161,105,0.4); }

/* ── INFO / INSIGHT BOXES ── */
.insight-box {
    background: linear-gradient(90deg, rgba(66,153,225,0.15) 0%, rgba(66,153,225,0.05) 100%);
    border-left: 4px solid #63b3ed;
    padding: 12px 18px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #bee3f8 !important;
    box-shadow: 0 2px 12px rgba(66,153,225,0.12);
}
.warning-box {
    background: linear-gradient(90deg, rgba(245,101,101,0.15) 0%, rgba(245,101,101,0.05) 100%);
    border-left: 4px solid #fc8181;
    padding: 12px 18px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #fed7d7 !important;
    box-shadow: 0 2px 12px rgba(245,101,101,0.12);
}
.success-box {
    background: linear-gradient(90deg, rgba(72,187,120,0.15) 0%, rgba(72,187,120,0.05) 100%);
    border-left: 4px solid #68d391;
    padding: 12px 18px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #c6f6d5 !important;
    box-shadow: 0 2px 12px rgba(72,187,120,0.12);
}

/* ── DATAFRAMES / TABLES ── */
.stDataFrame, iframe { border-radius: 10px !important; }

/* ── EXPANDER ── */
details {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}

/* ── MULTISELECT / SELECT WIDGETS ── */
.stMultiSelect [data-baseweb="tag"] {
    background: linear-gradient(90deg,#4299e1,#667eea) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* ── HORIZONTAL DIVIDERS ── */
hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── PAGE TITLE ── */
.stApp h1 {
    background: linear-gradient(90deg, #63b3ed, #b794f4, #76e4f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

/* ── HIDE BRANDING ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚡ Loading & cleaning energy data…")
def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the UCI Household Power Consumption dataset.
    Handles:
      - '?' missing value placeholders
      - Mixed-type columns
      - Datetime parsing
      - Forward-fill for small gaps
    """
    df = pd.read_csv(
        filepath,
        sep=",",
        low_memory=False,
        na_values=["?", "NA", ""],
    )

    # Combine Date + Time into a single datetime index
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)

    # Cast all measurement columns to float
    numeric_cols = [
        "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill short gaps (≤ 60 min), then drop remaining NaNs
    df.ffill(limit=60, inplace=True)
    df.dropna(subset=["Global_active_power"], inplace=True)

    # ── Derived columns ──────────────────────────────────────────────────────
    # Energy in kWh per minute-interval: P(kW) × (1/60 h)
    df["energy_kwh"] = df["Global_active_power"] / 60.0

    # Calendar features
    df["hour"]      = df.index.hour
    df["weekday"]   = df.index.weekday          # 0=Mon … 6=Sun
    df["week"]      = df.index.isocalendar().week.astype(int)
    df["month"]     = df.index.month
    df["year"]      = df.index.year
    df["date"]      = df.index.date
    df["season"]    = df["month"].map(SEASONS)
    df["is_weekend"]= df["weekday"].isin([5, 6])
    df["is_peak"]   = df["hour"].isin(PEAK_HOURS)

    # Sub-metering energy (already in Wh/min → kWh/min)
    df["sub1_kwh"] = df["Sub_metering_1"] / 1000.0
    df["sub2_kwh"] = df["Sub_metering_2"] / 1000.0
    df["sub3_kwh"] = df["Sub_metering_3"] / 1000.0

    # Estimated cost & CO₂
    df["cost_gbp"] = df["energy_kwh"] * ELECTRICITY_RATE_GBP
    df["co2_kg"]   = df["energy_kwh"] * CO2_FACTOR_KG_KWH

    return df


@st.cache_data(show_spinner=False)
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("D").agg(
        total_kwh    = ("energy_kwh",          "sum"),
        peak_kwh     = ("energy_kwh",          lambda x: x[df.loc[x.index, "is_peak"]].sum()),
        avg_power_kw = ("Global_active_power", "mean"),
        max_power_kw = ("Global_active_power", "max"),
        cost_gbp     = ("cost_gbp",            "sum"),
        co2_kg       = ("co2_kg",              "sum"),
        sub1_kwh     = ("sub1_kwh",            "sum"),
        sub2_kwh     = ("sub2_kwh",            "sum"),
        sub3_kwh     = ("sub3_kwh",            "sum"),
    ).dropna()
    daily["month"]   = daily.index.month
    daily["year"]    = daily.index.year
    daily["weekday"] = daily.index.weekday
    daily["season"]  = daily["month"].map(SEASONS)
    daily["is_weekend"] = daily["weekday"].isin([5, 6])
    daily["week_start"] = daily.index.to_period("W").start_time
    return daily


@st.cache_data(show_spinner=False)
def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.resample("h").agg(
        total_kwh    = ("energy_kwh",          "sum"),
        avg_power_kw = ("Global_active_power", "mean"),
        cost_gbp     = ("cost_gbp",            "sum"),
        co2_kg       = ("co2_kg",              "sum"),
    ).dropna()
    hourly["hour"]    = hourly.index.hour
    hourly["weekday"] = hourly.index.weekday
    hourly["season"]  = hourly.index.month.map(SEASONS)
    return hourly


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — KPI ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    total_kwh  = df["energy_kwh"].sum()
    total_cost = df["cost_gbp"].sum()
    total_co2  = df["co2_kg"].sum()
    peak_kwh   = df.loc[df["is_peak"], "energy_kwh"].sum()
    peak_pct   = (peak_kwh / total_kwh * 100) if total_kwh > 0 else 0
    avg_daily_kwh  = daily["total_kwh"].mean()
    avg_daily_cost = daily["cost_gbp"].mean()
    max_power_kw   = df["Global_active_power"].max()
    n_days         = len(daily)

    # Sub-metering shares (of total recorded sub-metering energy)
    sub_total = df[["sub1_kwh", "sub2_kwh", "sub3_kwh"]].sum().sum()
    sub_shares = {
        "Kitchen (Sub-1)":       df["sub1_kwh"].sum() / sub_total * 100 if sub_total else 0,
        "Laundry (Sub-2)":       df["sub2_kwh"].sum() / sub_total * 100 if sub_total else 0,
        "HVAC/Water (Sub-3)":    df["sub3_kwh"].sum() / sub_total * 100 if sub_total else 0,
        "Other/Unmetered":       max(0, (total_kwh - sub_total) / total_kwh * 100) if total_kwh else 0,
    }

    return {
        "total_kwh":       total_kwh,
        "total_cost":      total_cost,
        "total_co2":       total_co2,
        "peak_kwh":        peak_kwh,
        "peak_pct":        peak_pct,
        "avg_daily_kwh":   avg_daily_kwh,
        "avg_daily_cost":  avg_daily_cost,
        "max_power_kw":    max_power_kw,
        "n_days":          n_days,
        "sub_shares":      sub_shares,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION MODEL
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🤖 Training forecast model…")
def train_forecast_model(daily: pd.DataFrame):
    """
    Random Forest regression on calendar + lag features.
    Target: next-day total_kwh
    """
    df_m = daily[["total_kwh", "cost_gbp", "month", "weekday", "is_weekend"]].copy()
    df_m = df_m.sort_index()

    # Lag features (days 1, 2, 3, 7)
    for lag in [1, 2, 3, 7]:
        df_m[f"lag_{lag}"] = df_m["total_kwh"].shift(lag)

    # Rolling mean features
    df_m["roll_7d"]  = df_m["total_kwh"].shift(1).rolling(7).mean()
    df_m["roll_30d"] = df_m["total_kwh"].shift(1).rolling(30).mean()

    df_m.dropna(inplace=True)

    feature_cols = ["month", "weekday", "is_weekend",
                    "lag_1", "lag_2", "lag_3", "lag_7",
                    "roll_7d", "roll_30d"]
    X = df_m[feature_cols]
    y = df_m["total_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae":   mean_absolute_error(y_test, y_pred),
        "r2":    r2_score(y_test, y_pred),
        "test":  pd.Series(y_test.values,  index=y_test.index),
        "pred":  pd.Series(y_pred,         index=y_test.index),
    }

    # Forecast next 30 days
    last_vals = daily["total_kwh"].values[-30:]
    future_rows = []
    hist = list(daily["total_kwh"].values)

    for i in range(30):
        future_date = daily.index[-1] + timedelta(days=i + 1)
        roll_7  = np.mean(hist[-7:])
        roll_30 = np.mean(hist[-30:])
        row = {
            "month":      future_date.month,
            "weekday":    future_date.weekday(),
            "is_weekend": future_date.weekday() in [5, 6],
            "lag_1":      hist[-1],
            "lag_2":      hist[-2],
            "lag_3":      hist[-3],
            "lag_7":      hist[-7],
            "roll_7d":    roll_7,
            "roll_30d":   roll_30,
        }
        pred_val = model.predict(pd.DataFrame([row]))[0]
        hist.append(pred_val)
        future_rows.append({"date": future_date, "forecast_kwh": pred_val,
                             "forecast_cost": pred_val * ELECTRICITY_RATE_GBP})

    forecast_df = pd.DataFrame(future_rows).set_index("date")

    importances = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    return model, metrics, forecast_df, importances


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — CHART WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────
# Dark-theme chart colours
_CHART_BG   = "rgba(15,12,41,0)"       # fully transparent — shows app gradient
_PAPER_BG   = "rgba(26,26,78,0.55)"    # semi-transparent deep-navy card
_GRID_COLOR = "rgba(255,255,255,0.08)"
_FONT_COLOR = "#e2e8f0"

def styled_fig(fig):
    """Apply consistent dark-theme styling to all Plotly figures."""
    fig.update_layout(
        font_family="Segoe UI, sans-serif",
        font_color=_FONT_COLOR,
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_PAPER_BG,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font_color=_FONT_COLOR, bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR, tickfont_size=11,
                     tickfont_color=_FONT_COLOR, linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR, tickfont_size=11,
                     tickfont_color=_FONT_COLOR, linecolor="rgba(255,255,255,0.12)")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Lightning_bolt.svg/120px-Lightning_bolt.svg.png", width=50)
        st.title("⚡ Energy Analytics")
        st.markdown("**Household Power Consumption**")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["📊 Executive Overview",
             "📈 Consumption & Cost Analysis",
             "⚠️ Risk & Opportunity Analysis"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Dataset Info**")
        st.caption("UCI Household Electric Power Consumption  \nDec 2006 – Nov 2010  \n~2M minute-level readings")
        st.markdown("---")
        st.caption("⚙️ Rates: £0.28/kWh · 0.233 kg CO₂/kWh")

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df = load_and_clean("household_power_consumption.csv")
    except FileNotFoundError:
        st.error("❌ `household_power_consumption.csv` not found. Place it in the same directory as `app.py`.")
        st.stop()

    daily  = aggregate_daily(df)
    hourly = aggregate_hourly(df)
    kpis   = compute_kpis(df, daily)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EXECUTIVE OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    if page == "📊 Executive Overview":
        st.title("📊 Executive Overview")
        st.markdown("**Top-line performance of household energy consumption (2006–2010)**")
        st.markdown("---")

        # ── KPI Row 1 ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⚡ Total Consumption",  f"{kpis['total_kwh']:,.0f} kWh",
                  f"{kpis['avg_daily_kwh']:.1f} kWh/day avg")
        c2.metric("💷 Total Est. Cost",    f"£{kpis['total_cost']:,.0f}",
                  f"£{kpis['avg_daily_cost']:.2f}/day avg")
        c3.metric("🌿 Total CO₂ Emitted",  f"{kpis['total_co2']/1000:,.1f} tonnes",
                  f"{kpis['total_co2']/kpis['n_days']:.2f} kg/day avg")
        c4.metric("🔴 Peak-Hour Share",     f"{kpis['peak_pct']:.1f}%",
                  f"{kpis['peak_kwh']:,.0f} kWh in peak hours")

        # ── KPI Row 2 ─────────────────────────────────────────────────────────
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("📅 Dataset Span",        f"{kpis['n_days']:,} days")
        c6.metric("⚡ Max Instantaneous",   f"{kpis['max_power_kw']:.2f} kW")
        c7.metric("☀️ Solar Offset Potential",
                  f"{SOLAR_PANEL_OUTPUT_KWH_YEAR:,} kWh/yr",
                  f"≈ £{SOLAR_PANEL_OUTPUT_KWH_YEAR * ELECTRICITY_RATE_GBP:.0f} saving/yr")
        c8.metric("💰 Solar Payback",
                  f"{SOLAR_INSTALL_COST_GBP / (SOLAR_PANEL_OUTPUT_KWH_YEAR * ELECTRICITY_RATE_GBP):.1f} yrs",
                  "at current rates")

        st.markdown("---")

        # ── Appliance-wise share (sub-metering) ───────────────────────────────
        col_pie, col_trend = st.columns([1, 2])

        with col_pie:
            st.markdown('<div class="section-header">🔌 Appliance Usage Share</div>', unsafe_allow_html=True)
            share_df = pd.DataFrame.from_dict(
                kpis["sub_shares"], orient="index", columns=["pct"]
            ).reset_index().rename(columns={"index": "Appliance"})
            fig_pie = px.pie(
                share_df, names="Appliance", values="pct",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.45,
            )
            fig_pie.update_traces(textposition="outside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                  paper_bgcolor=_PAPER_BG,
                                  font_color=_FONT_COLOR)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('<div class="insight-box">Sub-1: Kitchen appliances · Sub-2: Laundry · Sub-3: HVAC / Water heater · Remainder: lighting, entertainment, etc.</div>', unsafe_allow_html=True)

        with col_trend:
            st.markdown('<div class="section-header">📅 Monthly Average Daily Consumption (kWh)</div>', unsafe_allow_html=True)
            monthly_avg = daily.groupby(["year", "month"])["total_kwh"].mean().reset_index()
            monthly_avg["period"] = monthly_avg["year"].astype(str) + "-" + monthly_avg["month"].astype(str).str.zfill(2)
            fig_monthly = px.bar(
                monthly_avg.sort_values("period"),
                x="period", y="total_kwh",
                color="total_kwh",
                color_continuous_scale="Blues",
                labels={"total_kwh": "Avg Daily kWh", "period": "Month"},
            )
            fig_monthly.update_layout(coloraxis_showscale=False, paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG)
            fig_monthly.update_xaxes(tickangle=-45, tickfont_size=9)
            st.plotly_chart(fig_monthly, use_container_width=True)

        st.markdown("---")

        # ── Yearly summary table ───────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Year-by-Year Summary</div>', unsafe_allow_html=True)
        yearly = daily.groupby("year").agg(
            Total_kWh    = ("total_kwh",  "sum"),
            Est_Cost_GBP = ("cost_gbp",   "sum"),
            CO2_kg       = ("co2_kg",     "sum"),
            Avg_Daily_kWh= ("total_kwh",  "mean"),
            Days         = ("total_kwh",  "count"),
        ).reset_index()
        yearly["Est_Cost_GBP"] = yearly["Est_Cost_GBP"].map("£{:,.0f}".format)
        yearly["Total_kWh"]    = yearly["Total_kWh"].map("{:,.0f}".format)
        yearly["CO2_kg"]       = yearly["CO2_kg"].map("{:,.0f} kg".format)
        yearly["Avg_Daily_kWh"]= yearly["Avg_Daily_kWh"].map("{:.1f}".format)
        st.dataframe(yearly, use_container_width=True, hide_index=True)

        # ── Forecast teaser ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 30-Day Energy Forecast (ML Preview)</div>', unsafe_allow_html=True)
        _, metrics, forecast_df, _ = train_forecast_model(daily)
        fig_fc = go.Figure()
        last30 = daily["total_kwh"].tail(60)
        fig_fc.add_trace(go.Scatter(x=last30.index, y=last30.values,
                                    name="Historical", line=dict(color=COLOUR_PRIMARY)))
        fig_fc.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast_kwh"],
                                    name="Forecast", line=dict(color=COLOUR_ACCENT, dash="dash")))
        fig_fc.update_layout(yaxis_title="Daily kWh", paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                             margin=dict(l=30, r=30, t=10, b=30))
        st.plotly_chart(styled_fig(fig_fc), use_container_width=True)
        fc_total = forecast_df["forecast_kwh"].sum()
        fc_cost  = forecast_df["forecast_cost"].sum()
        st.markdown(f'<div class="insight-box">🤖 Model R² = <b>{metrics["r2"]:.3f}</b> · MAE = <b>{metrics["mae"]:.2f} kWh/day</b><br>Forecast next 30 days: <b>{fc_total:.1f} kWh</b> · estimated cost: <b>£{fc_cost:.2f}</b></div>', unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — CONSUMPTION & COST ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📈 Consumption & Cost Analysis":
        st.title("📈 Consumption & Cost Analysis")
        st.markdown("**Deep-dive into usage patterns, cost drivers, and sub-metering breakdown**")
        st.markdown("---")

        # ── Filters ───────────────────────────────────────────────────────────
        with st.expander("🔧 Filter Options", expanded=False):
            year_opts = sorted(daily["year"].unique())
            sel_years = st.multiselect("Select Year(s)", year_opts, default=year_opts)
            season_opts = ["Winter", "Spring", "Summer", "Autumn"]
            sel_seasons = st.multiselect("Select Season(s)", season_opts, default=season_opts)

        daily_f = daily[daily["year"].isin(sel_years) & daily["season"].isin(sel_seasons)]
        hourly_f = hourly[hourly.index.year.isin(sel_years)]

        # ── Daily consumption trend ────────────────────────────────────────────
        st.markdown('<div class="section-header">📅 Daily Consumption Trend</div>', unsafe_allow_html=True)
        weekly_agg = daily_f.resample("W")["total_kwh"].mean()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_f.index, y=daily_f["total_kwh"],
            name="Daily kWh", line=dict(color="#cce5ff", width=1), opacity=0.5
        ))
        fig_trend.add_trace(go.Scatter(
            x=weekly_agg.index, y=weekly_agg.values,
            name="7-day rolling avg", line=dict(color=COLOUR_PRIMARY, width=2.5)
        ))
        fig_trend.update_layout(yaxis_title="kWh", paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                margin=dict(l=30, r=30, t=10, b=30))
        st.plotly_chart(styled_fig(fig_trend), use_container_width=True)

        # ── Seasonal & weekday patterns ────────────────────────────────────────
        col_s, col_w = st.columns(2)

        with col_s:
            st.markdown('<div class="section-header">🌦️ Seasonal Consumption</div>', unsafe_allow_html=True)
            seas_order = ["Winter", "Spring", "Summer", "Autumn"]
            seas_df = daily_f.groupby("season")["total_kwh"].mean().reindex(seas_order).reset_index()
            fig_seas = px.bar(seas_df, x="season", y="total_kwh",
                              color="season",
                              color_discrete_map={"Winter":"#4a90d9","Spring":"#5cb85c",
                                                  "Summer":"#f0ad4e","Autumn":"#d9534f"},
                              labels={"total_kwh": "Avg Daily kWh", "season": "Season"},
                              text_auto=".1f")
            fig_seas.update_layout(showlegend=False, paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                   margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_seas, use_container_width=True)
            st.markdown('<div class="insight-box">Winter demand is typically 30–40% higher than Summer — driven by heating and shorter daylight hours.</div>', unsafe_allow_html=True)

        with col_w:
            st.markdown('<div class="section-header">📆 Weekday vs Weekend Usage</div>', unsafe_allow_html=True)
            day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            wday_df = daily_f.groupby("weekday")["total_kwh"].mean().reset_index()
            wday_df["day_name"] = wday_df["weekday"].map(dict(enumerate(day_names)))
            wday_df["type"]     = wday_df["weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
            fig_wday = px.bar(wday_df, x="day_name", y="total_kwh",
                              color="type",
                              color_discrete_map={"Weekday": COLOUR_PRIMARY, "Weekend": COLOUR_ACCENT},
                              labels={"total_kwh": "Avg Daily kWh", "day_name": "Day"},
                              text_auto=".1f")
            fig_wday.update_layout(paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                   margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_wday, use_container_width=True)

        # ── Hour-of-day heatmap ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🕐 Hourly Consumption Heatmap (Hour × Weekday)</div>', unsafe_allow_html=True)
        hourly_f2 = hourly_f.copy()
        hourly_f2["weekday"] = hourly_f2.index.weekday
        hourly_f2["hour"]    = hourly_f2.index.hour
        hmap = hourly_f2.groupby(["weekday", "hour"])["total_kwh"].mean().unstack(level="hour")
        hmap.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig_heat = px.imshow(
            hmap,
            color_continuous_scale="YlOrRd",
            labels=dict(x="Hour of Day", y="Day", color="Avg kWh"),
            aspect="auto",
        )
        fig_heat.update_layout(paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                               font_color=_FONT_COLOR,
                               margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('<div class="insight-box">Darkest cells (highest demand) appear in evening hours (18:00–21:00) and weekend mornings — these are key targets for demand-response programmes.</div>', unsafe_allow_html=True)

        # ── Sub-metering detail ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🔌 Sub-Metering Breakdown Over Time</div>', unsafe_allow_html=True)
        monthly_sub = daily_f.resample("ME").agg(
            Kitchen   = ("sub1_kwh", "sum"),
            Laundry   = ("sub2_kwh", "sum"),
            HVAC_Water= ("sub3_kwh", "sum"),
        ).reset_index()
        monthly_sub.rename(columns={"Datetime": "date"}, inplace=True)
        monthly_sub_m = monthly_sub.melt(
            id_vars=monthly_sub.columns[0],
            value_vars=["Kitchen","Laundry","HVAC_Water"],
            var_name="Appliance", value_name="kWh"
        )
        fig_sub = px.area(
            monthly_sub_m, x=monthly_sub_m.columns[0], y="kWh",
            color="Appliance",
            color_discrete_map={"Kitchen":"#4e79a7","Laundry":"#f28e2b","HVAC_Water":"#e15759"},
            labels={"kWh": "Monthly kWh"},
        )
        fig_sub.update_layout(paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                              margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_sub, use_container_width=True)

        # ── Cost analysis ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">💷 Monthly Cost Trend</div>', unsafe_allow_html=True)
        monthly_cost = daily_f.resample("ME")["cost_gbp"].sum().reset_index()
        monthly_cost.columns = ["Date", "Monthly_Cost"]
        monthly_cost["MA3"]  = monthly_cost["Monthly_Cost"].rolling(3).mean()
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(x=monthly_cost["Date"], y=monthly_cost["Monthly_Cost"],
                                  name="Monthly Cost", marker_color=COLOUR_ACCENT))
        fig_cost.add_trace(go.Scatter(x=monthly_cost["Date"], y=monthly_cost["MA3"],
                                      name="3-month avg", line=dict(color=COLOUR_WARNING, width=2)))
        fig_cost.update_layout(yaxis_title="£ GBP", paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                               margin=dict(l=30, r=30, t=10, b=30))
        st.plotly_chart(styled_fig(fig_cost), use_container_width=True)

        avg_monthly = daily_f.resample("ME")["cost_gbp"].sum().mean()
        st.markdown(f'<div class="insight-box">Average monthly electricity bill: <b>£{avg_monthly:.2f}</b> · Annual equivalent: <b>£{avg_monthly*12:.0f}</b></div>', unsafe_allow_html=True)

        # ── ML Forecast detail ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 30-Day Demand Forecast (Random Forest)</div>', unsafe_allow_html=True)
        _, metrics, forecast_df, importances = train_forecast_model(daily)

        fc1, fc2 = st.columns([2, 1])
        with fc1:
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=metrics["test"].index, y=metrics["test"].values,
                                         name="Actual", line=dict(color=COLOUR_PRIMARY)))
            fig_val.add_trace(go.Scatter(x=metrics["pred"].index, y=metrics["pred"].values,
                                         name="Predicted", line=dict(color=COLOUR_ACCENT, dash="dot")))
            fig_val.update_layout(yaxis_title="Daily kWh", paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                  margin=dict(l=30, r=30, t=10, b=30))
            st.plotly_chart(styled_fig(fig_val), use_container_width=True)

        with fc2:
            st.markdown("**Model Performance**")
            st.metric("R² Score",  f"{metrics['r2']:.3f}")
            st.metric("MAE",       f"{metrics['mae']:.2f} kWh/day")
            st.markdown("**Feature Importance**")
            imp_df = importances.reset_index()
            imp_df.columns = ["Feature", "Importance"]
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Blues")
            fig_imp.update_layout(showlegend=False, paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                  margin=dict(l=5, r=5, t=5, b=5), height=300,
                                  yaxis=dict(autorange="reversed"))
            fig_imp.update_coloraxes(showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Forecast table
        st.markdown("**Next 30-Day Forecast**")
        fct = forecast_df.copy()
        fct.index = fct.index.strftime("%Y-%m-%d")
        fct.columns = ["Forecast kWh", "Forecast Cost (£)"]
        fct["Forecast kWh"]      = fct["Forecast kWh"].map("{:.2f}".format)
        fct["Forecast Cost (£)"] = fct["Forecast Cost (£)"].map("£{:.2f}".format)
        st.dataframe(fct, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — RISK & OPPORTUNITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "⚠️ Risk & Opportunity Analysis":
        st.title("⚠️ Risk & Opportunity Analysis")
        st.markdown("**Identify overload risks, cost escalation, and energy-saving opportunities**")
        st.markdown("---")

        # ── Risk 1: Peak-hour overload ─────────────────────────────────────────
        st.markdown('<div class="section-header">🔴 Risk 1 — Peak-Hour Demand Overload</div>', unsafe_allow_html=True)
        # Days where peak power > 90th percentile
        p90 = daily["max_power_kw"].quantile(0.90)
        p95 = daily["max_power_kw"].quantile(0.95)
        overload_days = daily[daily["max_power_kw"] > p95]
        normal_days   = daily[daily["max_power_kw"] <= p90]

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("P95 Peak Threshold",  f"{p95:.2f} kW")
        r1c2.metric("Overload Days (>P95)", f"{len(overload_days)}", f"{len(overload_days)/len(daily)*100:.1f}% of total")
        r1c3.metric("Worst Single Peak",   f"{daily['max_power_kw'].max():.2f} kW")

        fig_peak = go.Figure()
        fig_peak.add_trace(go.Scatter(x=daily.index, y=daily["max_power_kw"],
                                      name="Peak Power", line=dict(color="#aecde8", width=1)))
        fig_peak.add_hline(y=p90, line_dash="dash", line_color=COLOUR_ACCENT,
                           annotation_text=f"P90 = {p90:.2f} kW")
        fig_peak.add_hline(y=p95, line_dash="dash", line_color=COLOUR_WARNING,
                           annotation_text=f"P95 = {p95:.2f} kW")
        fig_peak.update_layout(yaxis_title="kW", paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                               margin=dict(l=30, r=60, t=10, b=30))
        st.plotly_chart(styled_fig(fig_peak), use_container_width=True)
        st.markdown('<div class="warning-box">⚡ <b>Risk:</b> Recurring peak demand above 5 kW risks tripping circuit breakers and incurring demand charges. '
                    'Concentrate high-load appliances (washing machine, oven, EV charger) in off-peak windows (22:00–06:00).</div>', unsafe_allow_html=True)

        # ── Risk 2: Rising cost trend ──────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">💷 Risk 2 — Annual Cost Escalation</div>', unsafe_allow_html=True)
        annual_cost = daily.groupby("year")["cost_gbp"].sum().reset_index()
        annual_cost.columns = ["Year", "Annual_Cost"]
        yoy_change = annual_cost["Annual_Cost"].pct_change() * 100

        rc1, rc2 = st.columns([2, 1])
        with rc1:
            fig_ac = px.bar(annual_cost, x="Year", y="Annual_Cost",
                            color="Annual_Cost", color_continuous_scale="Reds",
                            text_auto=",.0f", labels={"Annual_Cost": "Estimated Cost (£)"})
            fig_ac.update_layout(coloraxis_showscale=False, paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                 margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_ac, use_container_width=True)
        with rc2:
            st.markdown("**Year-on-Year Change**")
            for i, row in annual_cost.iterrows():
                chg = yoy_change.iloc[i]
                arrow = "↑" if chg > 0 else "↓" if chg < 0 else "→"
                clr   = "🔴" if chg > 5 else "🟡" if chg > 0 else "🟢"
                st.markdown(f"{clr} **{int(row['Year'])}**: £{row['Annual_Cost']:,.0f} {arrow} {chg:+.1f}%")

        st.markdown('<div class="warning-box">📈 <b>Risk:</b> Even at constant unit rates, rising consumption directly increases bills. '
                    'A 10% reduction in daily usage saves ~£{:.0f}/year at current tariffs.</div>'.format(
                    daily["cost_gbp"].sum() / kpis["n_days"] * 365 * 0.10), unsafe_allow_html=True)

        # ── Risk 3: Seasonal spikes ────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">❄️ Risk 3 — Seasonal Consumption Spikes</div>', unsafe_allow_html=True)
        season_order = ["Winter","Spring","Summer","Autumn"]
        seas_box = daily.copy()
        fig_box = px.box(seas_box, x="season", y="total_kwh", color="season",
                         category_orders={"season": season_order},
                         color_discrete_map={"Winter":"#4a90d9","Spring":"#5cb85c",
                                             "Summer":"#f0ad4e","Autumn":"#d9534f"},
                         labels={"total_kwh":"Daily kWh","season":"Season"},
                         points=False)
        fig_box.update_layout(showlegend=False, paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                              margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

        winter_med = seas_box[seas_box["season"]=="Winter"]["total_kwh"].median()
        summer_med = seas_box[seas_box["season"]=="Summer"]["total_kwh"].median()
        st.markdown(f'<div class="warning-box">❄️ Median winter daily usage ({winter_med:.1f} kWh) is <b>{winter_med/summer_med:.1f}× higher</b> than summer ({summer_med:.1f} kWh). '
                    'Loft insulation and smart thermostats could cut winter demand by 15–25%.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Opportunities ──────────────────────────────────────────────────────
        st.markdown("## 💡 Opportunities")

        op1, op2 = st.columns(2)

        with op1:
            st.markdown('<div class="section-header">🔄 Load Shifting Potential</div>', unsafe_allow_html=True)
            # Peak vs off-peak consumption comparison
            hourly_avg = df.groupby("hour")["energy_kwh"].mean().reset_index()
            hourly_avg["Period"] = hourly_avg["hour"].apply(
                lambda h: "Peak" if h in PEAK_HOURS else "Off-Peak"
            )
            fig_shift = px.bar(hourly_avg, x="hour", y="energy_kwh", color="Period",
                               color_discrete_map={"Peak": COLOUR_WARNING, "Off-Peak": COLOUR_SUCCESS},
                               labels={"energy_kwh":"Avg kWh/min","hour":"Hour of Day"})
            fig_shift.update_layout(paper_bgcolor=_PAPER_BG, plot_bgcolor=_CHART_BG,
                                    margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_shift, use_container_width=True)

            peak_avg  = hourly_avg[hourly_avg["Period"]=="Peak"]["energy_kwh"].sum()
            opeak_avg = hourly_avg[hourly_avg["Period"]=="Off-Peak"]["energy_kwh"].sum()
            shift_pct = 20   # conservative assumption: shift 20% of peak to off-peak
            saving_approx = kpis["total_kwh"] * (kpis["peak_pct"] / 100) * (shift_pct / 100) * ELECTRICITY_RATE_GBP
            st.markdown(f'<div class="success-box">✅ Shifting just <b>{shift_pct}% of peak-hour usage</b> to off-peak could save approximately '
                        f'<b>£{saving_approx:,.0f}</b> over the full dataset period '
                        f'(~£{saving_approx/kpis["n_days"]*365:.0f}/year).</div>', unsafe_allow_html=True)

        with op2:
            st.markdown('<div class="section-header">☀️ Solar Panel ROI</div>', unsafe_allow_html=True)
            years_range  = list(range(1, 26))
            solar_savings= [yr * SOLAR_PANEL_OUTPUT_KWH_YEAR * ELECTRICITY_RATE_GBP for yr in years_range]
            net_benefit   = [s - SOLAR_INSTALL_COST_GBP for s in solar_savings]
            fig_solar = go.Figure()
            fig_solar.add_trace(go.Scatter(x=years_range, y=solar_savings,
                                           name="Cumulative Savings", fill="tozeroy",
                                           line=dict(color=COLOUR_SUCCESS)))
            fig_solar.add_trace(go.Scatter(x=years_range, y=[SOLAR_INSTALL_COST_GBP]*25,
                                           name=f"Install Cost (£{SOLAR_INSTALL_COST_GBP:,})",
                                           line=dict(color=COLOUR_WARNING, dash="dash")))
            fig_solar.add_trace(go.Scatter(x=years_range, y=net_benefit,
                                           name="Net Benefit", line=dict(color=COLOUR_ACCENT)))
            fig_solar.update_layout(xaxis_title="Years", yaxis_title="£", paper_bgcolor=_PAPER_BG,
                                    plot_bgcolor=_CHART_BG, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(styled_fig(fig_solar), use_container_width=True)
            payback = SOLAR_INSTALL_COST_GBP / (SOLAR_PANEL_OUTPUT_KWH_YEAR * ELECTRICITY_RATE_GBP)
            st.markdown(f'<div class="success-box">☀️ A 4 kWp solar installation (£{SOLAR_INSTALL_COST_GBP:,}) pays back in '
                        f'<b>{payback:.1f} years</b> and generates <b>£{SOLAR_PANEL_OUTPUT_KWH_YEAR*ELECTRICITY_RATE_GBP*25:,.0f}</b> '
                        f'in savings over 25 years.</div>', unsafe_allow_html=True)

        # ── Recommended Actions ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">✅ Recommended Actions</div>', unsafe_allow_html=True)
        actions = [
            ("🔴 High Priority", "Enrol in a Time-of-Use tariff and shift washing machine / dishwasher / EV charging to 22:00–06:00 off-peak window."),
            ("🔴 High Priority", "Fit a smart thermostat to reduce winter HVAC spend — target 15–20% heating energy reduction."),
            ("🟡 Medium Priority","Install smart plugs on Sub-metering 1 (kitchen) circuits to monitor and limit idle/standby loads."),
            ("🟡 Medium Priority","Commission a 4 kWp solar PV system — payback period is under 6 years at current rates."),
            ("🟢 Low Priority",   "Replace any remaining incandescent lighting with LED (Sub-2 laundry circuit)."),
            ("🟢 Low Priority",   "Set up automated alerts when daily consumption exceeds the historical P95 threshold."),
        ]
        for priority, action in actions:
            badge_class = "risk-high" if "High" in priority else "risk-medium" if "Medium" in priority else "risk-low"
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:10px;margin:8px 0;">'
                f'<span class="{badge_class}">{priority}</span>'
                f'<span style="font-size:0.95rem;color:#1a202c;">{action}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#888;font-size:0.8rem;">'
        '⚡ Household Energy Analytics · Built with Streamlit · Data: UCI ML Repository'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
