# 🚗 AutoValue AI — Used Car Price Prediction

A professional, production-ready machine learning web application that accurately predicts the resale value of used cars in India (in INR Lakhs) based on vehicle specifications and market trends.

Powered by a trained **Random Forest Regressor Pipeline** and an interactive, modern automotive dashboard built with **Streamlit**, AutoValue AI provides instantaneous and reliable data-driven resale price estimates.

---

## 🚀 Live Demo

[AutoValue AI on Streamlit Community Cloud](https://share.streamlit.io/)

---

## ✨ Key Features

- **AI-Powered Resale Estimation**: Real-time predictions computed directly from the trained Machine Learning pipeline.
- **Brand & Model-Specific Intelligence**: Robust string parsing and brand-to-model dynamic hierarchy.
- **Location-Aware Valuation**: Incorporates regional market pricing variations across major Indian cities (Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Coimbatore, Jaipur, Kochi).
- **Comprehensive Vehicle Attributes**: 12 key features including Year, Kilometers Driven, Fuel Type, Transmission, Owner History, Engine CC, Max Power (bhp), Mileage (km/l), and Seats.
- **Modern Automotive UI**: Polished, responsive, and daylight-themed dashboard featuring an integrated automotive hero background, frosted glass HUD visual, clean input cards, and dynamic valuation cards.

---

## 📊 Model Performance

Two regression algorithms were trained and rigorously evaluated using an 80/20 train/test split:

| Model | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| Linear Regression | 3.0604 Lakhs | 5.0256 Lakhs | 0.7836 |
| **Random Forest Regressor (Selected)** | **1.4109 Lakhs** | **3.4963 Lakhs** | **0.8953** |

> **Selected Model**: The **Random Forest Regressor** achieved an $R^2$ of **0.8953** and a Mean Absolute Error of **1.41 Lakhs**, demonstrating high precision in capturing nonlinear relationships between vehicle wear, engine output, and brand equity.

---

## 🧠 Features Used by the Pipeline

The serialized pipeline (`model/car_price_model_final.pkl`) expects exactly 12 input features:

### Categorical Features (6)
- `Brand`
- `Model`
- `Location`
- `Fuel_Type`
- `Transmission`
- `Owner_Type`

### Numerical Features (6)
- `Year`
- `Kilometers_Driven`
- `Mileage`
- `Engine`
- `Power`
- `Seats`

---

## ⚙️ Machine Learning Workflow

```
Raw Dataset (train.csv)
   │
   ▼
Data Cleaning & Feature Extraction (Engine CC, Power bhp, Mileage km/l, Brand/Model parsed)
   │
   ▼
Train/Test Split (80% Train, 20% Test — No data leakage)
   │
   ▼
Preprocessing Pipeline (ColumnTransformer with OneHotEncoder & SimpleImputer)
   │
   ▼
Model Training & Hyperparameter Evaluation (Random Forest Regressor)
   │
   ▼
Unified Pipeline Serialization (Exported as car_price_model_final.pkl via joblib)
   │
   ▼
Interactive Streamlit Application (Production-ready web interface)
```

---

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Machine Learning**: Scikit-Learn, Joblib
- **Data Manipulation**: Pandas, NumPy
- **Styling**: Vanilla CSS with custom Google Fonts (Space Grotesk & Inter)
- **Deployment Platform**: Streamlit Community Cloud

---

## 📁 Repository Structure

```
AutoValue-AI/
│
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Minimal deployment dependencies
├── README.md                   # Project documentation
│
├── data/
│   └── train.csv               # Historical used-car dataset
│
├── model/
│   ├── car_price_model_final.pkl  # Trained production Random Forest pipeline
│   └── model_features.json        # Feature metadata schema
│
├── assets/
│   └── hero_car.jpg            # Automotive hero background visual asset
│
└── .streamlit/
    └── config.toml             # Streamlit theme configuration (#0788D1)
```

---

## 💻 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/autovalue-ai.git
cd autovalue-ai
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

Follow these steps to deploy AutoValue AI to production on **Streamlit Community Cloud**:

1. **Push to GitHub**:
   Ensure all project files (`app.py`, `requirements.txt`, `model/`, `assets/`, `data/`, `.streamlit/`) are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Prepare AutoValue AI for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Open Streamlit Community Cloud**:
   Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

3. **Create New App**:
   Click **"Create app"** (or **"New app"**).

4. **Configure Repository Details**:
   - **Repository**: Select your `autovalue-ai` repository.
   - **Branch**: `main` (or `master`).
   - **Main file path**: `app.py`.

5. **Deploy**:
   Click **"Deploy!"**. Streamlit Cloud will automatically install packages from `requirements.txt` and serve the application globally.

---

## ⚠️ Disclaimer

*This application provides an estimated resale value based on historical used-car transaction data in India. Actual market prices may vary depending on individual vehicle condition, service history, optional equipment, local market demand, and negotiation.*

---

## 👩‍💻 Author

**Supraja Seemakurthi**
