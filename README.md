 
🚗 Used Car Price Prediction

📌 Project Overview
This project predicts the *resale value of used cars* based on multiple features such as:
- Car Name  
- Year of Manufacture  
- Fuel Type  
- Kilometers Driven  
- City  
- Body Type  
- Transmission  
- Number of Owners  

The prediction model is built using XGBoost and deployed as an interactive Streamlit Dashboard.  
Additionally, Exploratory Data Analysis (EDA) was performed to identify market insights and pricing trends.

⚙ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit (for dashboard)
- Power BI (for additional insights and visualization)


📊 Key Insights from EDA
- Car price decreases with higher kilometers driven and older year of manufacture.  
- Diesel cars generally show higher resale value compared to petrol cars.  
- SUVs and Sedans have higher average prices compared to hatchbacks.  
- Top cities like Delhi, Mumbai, and Bangalore dominate the used car market.  


🤖 Machine Learning Model
- Model Used: XGBoost Regressor  
- Features Used: Car name, fuel type, year of manufacture, kilometers run, city, body type, transmission, and total owners.  
- R² Score: ~0.82  
- Evaluation Metrics: RMSE, MAE, R²  
- The model explains approximately 82% of price variation in used cars.  


💻 Streamlit Dashboard
An interactive dashboard was created using Streamlit where users can input:
- Car details such as model, fuel type, and mileage  
- The app predicts the estimated resale price instantly  

🟢 How to Run

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Run the Streamlit app
streamlit run streamlit_app.py

📈 Power BI Dashboard

A separate Power BI dashboard was developed to visualize:

- Average price by car brand, city, and fuel type
- Price trends based on year of manufacture
- Market distribution of car types


🎥 Demo Video

🎬 You can watch the full project demonstration video here:
👉 Used Car Price Prediction Demo


📄 Project Report

You can view or download the full Data Analyst Project Report (PDF) here:
👉 Used Car Price Prediction Report


📂 Project Structure

used-car-price-prediction/

─ Used_Car_Price_Prediction.csv        # Dataset
─ streamlit_app.py                     # Streamlit Dashboard Script
─ app.py                               # Gradio version (optional)
─ Used Car Price Prediction Report.pdf # Final project report
─ README.md                            # Project documentation
─ PowerBI_Dashboard.png                # Dashboard screenshot (optional)