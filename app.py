import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Train model automatically — no saved file needed
@st.cache_resource
def train_model():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return model, round(r2, 2), round(mae * 100000)

# Header
st.title("🏠 House Price Predictor")
st.markdown("Predicts **California house prices** using a Random Forest ML model trained on 16,000+ samples.")
st.divider()

with st.spinner("⏳ Training model... takes about 10 seconds on first load"):
    model, r2, mae = train_model()

st.success(f"✅ Model ready!  |  R² Score: **{r2}**  |  MAE: **~${mae:,}**")
st.divider()

st.subheader("Adjust house features:")
col1, col2 = st.columns(2)

with col1:
    med_inc    = st.slider("Median Income (×$10K)",          0.5,  15.0,  5.0,  0.1)
    house_age  = st.slider("House Age (years)",               1,    52,    20,   1)
    ave_rooms  = st.slider("Avg Rooms per Household",         1.0,  10.0,  5.0,  0.1)
    ave_bedrms = st.slider("Avg Bedrooms per Household",      0.5,  5.0,   1.0,  0.1)

with col2:
    population = st.slider("Block Population",                10,   3500,  1200, 50)
    ave_occup  = st.slider("Avg Occupancy",                   1.0,  6.0,   3.0,  0.1)
    latitude   = st.slider("Latitude",                        32.5, 42.0,  37.5, 0.1)
    longitude  = st.slider("Longitude",                      -124.0,-114.0,-122.0, 0.1)

st.divider()

if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    features = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    prediction = model.predict(features)[0]
    price_usd  = prediction * 100_000

    st.success(f"### 🏡 Estimated House Price: **${price_usd:,.0f}**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Low Estimate",  f"${price_usd * 0.88:,.0f}")
    col_b.metric("Predicted",     f"${price_usd:,.0f}")
    col_c.metric("High Estimate", f"${price_usd * 1.12:,.0f}")

    st.caption("⚠️ This is an ML estimate based on the California Housing dataset.")

# Sidebar
with st.sidebar:
    st.header("📊 About this project")
    st.markdown("""
**Dataset**: California Housing (sklearn)

**Model**: Random Forest Regressor
- 100 decision trees
- ~82% variance explained (R²)

**Features**:
- Median income
- House age
- Avg rooms / bedrooms
- Block population
- Avg occupancy
- Latitude / Longitude

**Tech stack**:
Python · Scikit-learn · Streamlit
    """)
    st.divider()
    st.markdown("🎓 Beginner ML portfolio project by Ishika")
