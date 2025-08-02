import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="EV Forecast", layout="wide")

# Load model
model = joblib.load('forecasting_ev_model.pkl')

# --- Custom Styling ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #f0f0f0;
        }
        h1, h2, h3, h4, h5, h6, p, label, .css-1v3fvcr, .css-1d391kg {
            color: #f0f0f0 !important;
        }
        .css-1cpxqw2 edgvbvh3, .stSelectbox label, .stMultiSelect label {
            color: #f0f0f0 !important;
        }
        .highlight-box {
            background-color: #151f3a;
            padding: 1rem;
            border-radius: 0.5rem;
            color: #f0f0f0;
            font-weight: 500;
            font-size: 16px;
            border-left: 6px solid #8a2be2;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #ffffff; margin-top: 20px;'>
        🔮 EV Adoption Forecaster for a County in United States
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #cccccc;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
""", unsafe_allow_html=True)

# Image
st.image("ev-car-factory.jpg", use_container_width=True)

st.markdown("""
    <div style='font-size: 20px; padding-top: 10px; color: #dddddd;'>
        Select a county and see the forecasted EV adoption trend for the next 3 years.
    </div>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# County dropdown
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df['County'].unique():
    st.markdown("<div class='highlight-box'>County not found in dataset.</div>", unsafe_allow_html=True)
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# Forecast logic
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0] if len(cumulative_ev) >= 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    historical_ev.pop(0)
    cumulative_ev.append(cumulative_ev[-1] + pred)
    cumulative_ev.pop(0)

# Combine data
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# Plot forecast
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color='#ffffff')
ax.set_xlabel("Date", color='#ffffff')
ax.set_ylabel("Cumulative EV Count", color='#ffffff')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='#ffffff')
ax.legend()
st.pyplot(fig)

# Growth info
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.markdown(f"<div class='highlight-box'>📌 EV adoption in <b>{county}</b> is expected to show a <b>{trend}</b> of <b>{forecast_growth_pct:.2f}%</b> over the next 3 years.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='highlight-box'>⚠️ Historical EV total is zero; can't compute percentage change.</div>", unsafe_allow_html=True)

# Compare counties
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")

multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []
    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]
        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            ev_slope = np.polyfit(range(6), cum_ev[-6:], 1)[0] if len(cum_ev) >= 6 else 0

            new_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }

            pred = model.predict(pd.DataFrame([new_row]))[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
            hist_ev.append(pred)
            hist_ev.pop(0)
            cum_ev.append(cum_ev[-1] + pred)
            cum_ev.pop(0)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    comp_df = pd.concat(comparison_data, ignore_index=True)

    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='#ffffff')
    ax.set_xlabel("Date", color='#ffffff')
    ax.set_ylabel("Cumulative EV Count", color='#ffffff')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='#ffffff')
    ax.legend(title="County")
    st.pyplot(fig)

    # Summary text
    growth_summaries = []
    for cty in multi_counties:
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        forecasted_total = cty_df['Cumulative EV'].iloc[-1]
        if historical_total > 0:
            growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A")

    growth_sentence = " | ".join(growth_summaries)
    st.markdown(f"<div class='highlight-box'>📊 Forecasted EV adoption growth over next 3 years — {growth_sentence}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='highlight-box'>✅ Forecast complete. Prepared for the <b>AICTE Internship Cycle 2 by S4F</b>.</div>", unsafe_allow_html=True)
