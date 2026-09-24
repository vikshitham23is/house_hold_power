# ⚡ Household Energy Consumption & Sustainability Analytics

A business intelligence dashboard for analysing household electricity usage, cost, carbon emissions, and identifying energy-saving opportunities — built with **Streamlit**, **Plotly**, and **scikit-learn**.

---

## 📋 Project Overview

This project uses the **UCI Individual Household Electric Power Consumption** dataset (≈2 million minute-level readings from Dec 2006 to Nov 2010) to build a fully interactive, 3-page analytics dashboard that answers:

- How much energy is consumed, what does it cost, and what is the carbon footprint?
- When is demand highest — by hour, day, season?
- Which appliance circuits are driving usage?
- What are the overload and cost-escalation risks?
- What savings are possible through load-shifting or solar PV?
- What will consumption look like over the next 30 days?

---

## 📂 Project Structure

```
.
├── app.py                            # Single-file Streamlit dashboard (backend + frontend)
├── household_power_consumption.csv   # Dataset (see source below)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🗂️ Dashboard Pages

| Page | Contents |
|---|---|
| **📊 Executive Overview** | KPI summary cards, appliance usage share (donut chart), monthly consumption bar chart, year-by-year table, 30-day ML forecast teaser |
| **📈 Consumption & Cost Analysis** | Daily trend chart, seasonal & weekday breakdowns, hourly heatmap (hour × weekday), sub-metering area chart, monthly cost trend, full ML forecast with feature importance |
| **⚠️ Risk & Opportunity Analysis** | Peak-hour overload detection, annual cost escalation trend, seasonal spike boxplots, load-shifting savings calculator, solar ROI chart, prioritised action plan |

---

## ⚙️ KPIs & Assumptions

| KPI | Formula / Source |
|---|---|
| **Energy (kWh)** | `Global_active_power (kW) × 1/60 h` per minute interval |
| **Estimated Cost** | kWh × £0.28 (UK average unit rate, 2024) |
| **CO₂ Emissions** | kWh × 0.233 kg/kWh (UK grid carbon intensity, DESNZ 2023) |
| **Peak Hours** | 07:00–09:59 and 17:00–20:59 (typical UK Time-of-Use peak window) |
| **Solar System** | 4 kWp, avg UK output 3,400 kWh/yr, install cost £6,500 |

---

## 🤖 Prediction Model

A **Random Forest Regressor** (scikit-learn) is trained on:
- Calendar features: month, weekday, is_weekend
- Lag features: consumption 1, 2, 3, and 7 days prior
- Rolling mean features: 7-day and 30-day rolling averages

The model produces a **30-day ahead daily energy and cost forecast**, displayed on both the Executive Overview and Consumption & Cost pages.

---

## 🚀 Setup & Running

### 1. Prerequisites

- Python 3.10 or higher
- The dataset CSV file (see below)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Download the dataset (see link below) and place `household_power_consumption.csv` in the same directory as `app.py`.

> The original UCI dataset uses semicolon (`;`) separators and `?` as missing value placeholders. This project expects a **comma-separated CSV** version. If using the raw UCI download, run:
> ```python
> import pandas as pd
> df = pd.read_csv("household_power_consumption.txt", sep=";", na_values="?")
> df.to_csv("household_power_consumption.csv", index=False)
> ```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Dashboard UI framework |
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `scikit-learn` | Random Forest forecasting model |

---

## 🔗 Dataset Source

**UCI Machine Learning Repository — Individual Household Electric Power Consumption**  
[https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

> Hébrail, G. & Bérard, A. (2012). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

---

## 📄 Licence

For educational and analytical purposes only. Dataset © UCI ML Repository.
