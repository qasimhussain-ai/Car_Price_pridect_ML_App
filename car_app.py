import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & columns
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

MAE = 879  # your CV MAE

# Page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:32px !important;
    font-weight:700;
    color:#1f4aff;
}
.card {
    padding:20px;
    border-radius:15px;
    background: #f5f7ff;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.sub {
    color: gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<p class='big-font'>🚗 Car Price Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Estimate your car’s market value using machine learning</p>", unsafe_allow_html=True)
st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    year = st.slider("📅 Manufacturing Year", 1990, 2025, 2018)
    mileage = st.number_input("🛣 Mileage", min_value=0, value=50000, step=1000)
    engineSize = st.number_input("⚙ Engine Size (L)", 0.5, 6.0, 1.6)

with col2:
    mpg = st.number_input("⛽ MPG", 10.0, 100.0, 50.0)
    tax = st.number_input("💷 Tax", min_value=0, value=150)
    transmission = st.selectbox(
    "🔁 Transmission",
    ["Manual", "Semi-Auto", "Automatic"])
# Dropdowns
model_name = st.selectbox(
    "🚘 Car Model",
    [
        "C-MAX","EcoSport","Edge","Escort","Fiesta","Focus","Fusion","Galaxy",
        "Grand C-MAX","Grand Tourneo Connect","KA","Ka+","Kuga","Mondeo",
        "Mustang","Puma","Ranger","S-MAX","Streetka",
        "Tourneo Connect","Tourneo Custom","Transit Tourneo"
    ]
)

fuel = st.selectbox("🔥 Fuel Type", ["Petrol", "Hybrid", "Electric", "Other"])

st.divider()

# Predict
if st.button("💰 Predict Price", use_container_width=True):
    input_data = {
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize,
        "car_age": 2025 - year,
        "transmission": transmission,
        "model": model_name,
        "fuelType": fuel
    }

    df = pd.DataFrame([input_data])

    # Align with model's expected features. If the estimator exposes
    # `feature_names_in_`, prefer that; otherwise fall back to `model_columns`.
    expected_cols = getattr(model, "feature_names_in_", model_columns)

    # If the model expects the raw categorical names, just reindex.
    if all(x in expected_cols for x in ("model", "fuelType", "transmission")):
        # If the saved estimator is a plain sklearn estimator (not a pipeline)
        # it likely expects numeric-encoded category values. Try to derive
        # category encodings from `model_columns` if available.
        if not hasattr(model, 'named_steps'):
            model_cats = []
            fuel_cats = []
            trans_cats = []
            for col in model_columns:
                if col.startswith('model_'):
                    model_cats.append(col.split('model_', 1)[1].strip())
                elif col.startswith('fuelType_'):
                    fuel_cats.append(col.split('fuelType_', 1)[1].strip())
                elif col.startswith('transmission_'):
                    trans_cats.append(col.split('transmission_', 1)[1].strip())

            def cat_index(lst, val):
                try:
                    return lst.index(val)
                except Exception:
                    return -1

            if model_cats:
                df['model'] = cat_index(model_cats, model_name)
            if fuel_cats:
                df['fuelType'] = cat_index(fuel_cats, fuel)
            if trans_cats:
                df['transmission'] = cat_index(trans_cats, transmission)

        df = df.reindex(columns=expected_cols, fill_value=0)
    else:
        # Model expects one-hot columns. Create zeros then set the matching one-hot.
        df = df.reindex(columns=expected_cols, fill_value=0)

        # Helper to set one-hot for a value; tries both `prefix+value` and `prefix+ ' '+value`
        def set_onehot(prefix, value):
            key1 = f"{prefix}{value}"
            key2 = f"{prefix} {value}"
            if key1 in df.columns:
                df.loc[0, key1] = 1
            elif key2 in df.columns:
                df.loc[0, key2] = 1

        set_onehot("model_", model_name)
        set_onehot("transmission_", transmission)
        set_onehot("fuelType_", fuel)

    prediction = model.predict(df)[0]

    st.markdown(
        f"""
        <div class="card">
            <h2>💰 Estimated Price</h2>
            <h1>£ {prediction:,.0f}</h1>
            <p class="sub">Expected range: £ {prediction-MAE:,.0f} – £ {prediction+MAE:,.0f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption("📊 Model Performance: CV R² = 0.925 | RMSE ≈ 1270 | MAE ≈ 879")
