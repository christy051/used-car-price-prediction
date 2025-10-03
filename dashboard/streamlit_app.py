# streamlit_app.py - Stylish Used Car Price Predictor 🚗

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import streamlit as st

# -------------------------
# 1) Load & Clean Dataset
# -------------------------
file_path = r"C:\Users\SAMSUNG\OneDrive\Documents\CarProject\Used_Car_Price_Prediction.csv"
df = pd.read_csv(file_path)

# Clean data
df['ad_created_on'] = pd.to_datetime(df['ad_created_on'], errors='coerce')
df['original_price'] = df['original_price'].fillna(df['original_price'].median())

for col in ['car_availability', 'transmission', 'body_type', 'source',
            'registered_city', 'registered_state', 'car_rating', 'fitness_certificate']:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

df = df[df['sale_price'] > 10000].reset_index(drop=True)

# -------------------------
# 2) Prepare for Modeling
# -------------------------
df_model = df.copy()
df_model = df_model.drop(columns=['ad_created_on'], errors='ignore')

encoders = {}
for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

X = df_model.drop(columns=['sale_price'])
y = df_model['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3) Train XGBoost Model
# -------------------------
best_model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
best_model.fit(X_train, y_train)

# -------------------------
# 4) Prediction Function
# -------------------------
def predict_car_price(car_name, yr_mfr, fuel_type, kms_run, city, body_type, transmission, total_owners):
    input_data = {}
    for col in X.columns:
        if col in encoders:
            default_val = encoders[col].classes_[0]
            input_data[col] = [default_val]
        else:
            input_data[col] = [0]

    input_data.update({
        "car_name": [car_name],
        "yr_mfr": [int(yr_mfr)],
        "fuel_type": [fuel_type],
        "kms_run": [int(kms_run)],
        "city": [city],
        "body_type": [body_type],
        "transmission": [transmission],
        "total_owners": [int(total_owners)]
    })

    input_df = pd.DataFrame(input_data)

    for col in input_df.columns:
        if col in encoders:
            input_df[col] = input_df[col].apply(
                lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
            )
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

    input_df = input_df[X.columns]

    prediction = best_model.predict(input_df)[0]
    return f"💰 Estimated Price: ₹{prediction:,.0f}"

# -------------------------
# 5) Stylish Streamlit UI
# -------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

# Add background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #1e3c72, #2a5298);
        background-attachment: fixed;
        color: white;
    }
    .stApp {
        background: url("https://images.unsplash.com/photo-1503736334956-4c8f8e92946d?auto=format&fit=crop&w=1600&q=80") no-repeat center center fixed;
        background-size: cover;
    }
    .card {
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='card' style='text-align:center'><h1>🚗 Used Car Price Predictor</h1><p>Predict resale prices with AI</p></div>", unsafe_allow_html=True)

# Dropdowns from dataset
car_names_list = sorted(df['car_name'].unique().tolist())
fuel_types_list = sorted(df['fuel_type'].dropna().unique().tolist())
cities_list = sorted(df['city'].dropna().unique().tolist())
body_types_list = sorted(df['body_type'].dropna().unique().tolist())
transmission_list = sorted(df['transmission'].dropna().unique().tolist())

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🔧 Enter Car Details")

car_name = st.selectbox("Car Name", car_names_list)
yr_mfr = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, value=2017)
fuel_type = st.selectbox("Fuel Type", fuel_types_list)
kms_run = st.number_input("Kms Run", min_value=1000, max_value=300000, value=30000)
city = st.selectbox("City", cities_list)
body_type = st.selectbox("Body Type", body_types_list)
transmission = st.selectbox("Transmission", transmission_list)
total_owners = st.number_input("Total Owners", min_value=1, max_value=5, value=1)

if st.button("🔮 Predict Price"):
    price = predict_car_price(car_name, yr_mfr, fuel_type, kms_run, city, body_type, transmission, total_owners)
    st.success(price)

st.markdown("</div>", unsafe_allow_html=True)
