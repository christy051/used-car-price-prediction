# app.py - Used Car Price Prediction Project 🚗

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import gradio as gr

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
    input_data = {
        "car_name": [car_name],
        "yr_mfr": [int(yr_mfr)],
        "fuel_type": [fuel_type],
        "kms_run": [int(kms_run)],
        "city": [city],
        "body_type": [body_type],
        "transmission": [transmission],
        "total_owners": [int(total_owners)]
    }

    input_df = pd.DataFrame(input_data)

    # Encode categorical inputs
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

    prediction = best_model.predict(input_df)[0]
    return f"💰 Estimated Price: ₹{prediction:,.0f}"

# -------------------------
# 5) Gradio Web App
# -------------------------
car_names_list = sorted(df['car_name'].unique().tolist())
fuel_types_list = sorted(df['fuel_type'].dropna().unique().tolist())
cities_list = sorted(df['city'].dropna().unique().tolist())
body_types_list = sorted(df['body_type'].dropna().unique().tolist())
transmission_list = sorted(df['transmission'].dropna().unique().tolist())

with gr.Blocks(css="""
body {
    background: url('https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1600&q=80') no-repeat center center fixed;
    background-size: cover;
    font-family: Arial, sans-serif;
}
.gradio-container {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 20px;
}
""") as demo:

    gr.Markdown(
        """
        <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.6); border-radius: 10px;">
            <h1 style="color: white;">🚗 Used Car Price Predictor</h1>
            <p style="color: #f0f0f0;">Enter your car details below to get the estimated resale price.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            car_name = gr.Dropdown(choices=car_names_list, label="Car Name", allow_custom_value=True)
            yr_mfr = gr.Number(label="Year of Manufacture", value=2017)
            fuel_type = gr.Dropdown(choices=fuel_types_list, label="Fuel Type")
            kms_run = gr.Number(label="Kms Run", value=30000)
            city = gr.Dropdown(choices=cities_list, label="City")
            body_type = gr.Dropdown(choices=body_types_list, label="Body Type")
            transmission = gr.Dropdown(choices=transmission_list, label="Transmission")
            total_owners = gr.Number(label="Total Owners", value=1)

        with gr.Column():
            output = gr.Textbox(label="Predicted Price", lines=2)

    gr.Button("🔮 Predict Price").click(
        fn=predict_car_price,
        inputs=[car_name, yr_mfr, fuel_type, kms_run, city, body_type, transmission, total_owners],
        outputs=output
    )

# Launch App
demo.launch()
