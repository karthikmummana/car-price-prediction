# 🚗 AutoValue AI — Used Car Price Prediction

An end-to-end Machine Learning web application that predicts the **estimated used-car asking price** in **INR (₹) Lakhs** for pre-owned vehicles across India. 

Powered by a **Tuned Random Forest Regressor Pipeline** trained on a recent Indian used-car dataset and delivered through a responsive, daylight-themed **Streamlit** dashboard, AutoValue AI generates real-time, data-driven asking price estimates.

---

## 🚀 Live Demo

- **Web Application:** [AutoValue AI — Live Demo](https://ai-used-car-valuation.streamlit.app/)
- **GitHub Repository:** [AutoValue AI on GitHub](https://github.com/seemakurthisupraja/car-price-prediction)

---

## 🎯 Problem Statement

Determining a fair resale valuation for used vehicles in India is challenging due to large variance across brands, trim models, vehicle age, cumulative mileage, fuel technology, transmission types, and ownership history. 

AutoValue AI addresses this pricing asymmetry by learning nonlinear depreciation curves and brand equity patterns from recent Indian used-car market listings, providing buyers and sellers with an instant, data-driven approximate valuation benchmark.

---

## ✨ Key Features

- **Dynamic Machine Learning Inferences**: Real-time predictions computed directly on-the-fly via a serialized Scikit-Learn pipeline (`model/car_price_model_final.pkl`) without hardcoded outputs.
- **Brand & Model Hierarchy**: Supports 42 manufacturers and 430 distinct car models with dynamic dropdown filtering.
- **Automotive UI Dashboard**: Custom styling featuring daylight palette (`#EEF4FA`, `#0788D1`, `#102A43`), frosted glass HUD visual card with animated status indicator, side-by-side specification cards, and wide valuation output with vector SVG accents.
- **Robust Outlier & Input Handling**: Built-in median imputation and categorical one-hot encoding with `handle_unknown="ignore"` for safe inference on edge cases.
- **Transparent Valuation Framing**: Transparently labels outputs as estimated used-car asking prices accompanied by contextual market disclaimers.

---

## 📊 Dataset

- **File**: [`data/used_cars_dataset_v2.csv`](data/used_cars_dataset_v2.csv)
- **Total Records (Raw)**: 14,993 rows × 11 columns
- **Total Records (Cleaned)**: 13,987 rows
- **Target Variable**: `AskPrice` (Numerical asking price in INR Lakhs)
- **Dataset Timeline**: Contemporary Indian used-car listings predominantly indexed in late 2024.

### Raw Data Attributes
1. `Brand`: Vehicle manufacturer
2. `model`: Specific model designation
3. `Year`: Manufacturing year (1996 to 2024 in cleaned data)
4. `Age`: Vehicle age derived from listing timeline (2024 - Year)
5. `kmDriven`: Cumulative odometer reading
6. `Transmission`: Gearbox type (`Manual`, `Automatic`)
7. `Owner`: Ownership count (`First`, `Second`)
8. `FuelType`: Engine fuel technology (`Petrol`, `Diesel`, `Hybrid`, `CNG / Hybrid`)
9. `PostedDate`: Listing date
10. `AdditionInfo`: Listing title and trim description
11. `AskPrice`: Target asking price string formatted in ₹ Lakhs / Crores

---

## 🧹 Data Cleaning and Preprocessing

The data preparation pipeline implemented in [`train_model_final.py`](train_model_final.py) executes the following justified steps:

1. **Deduplication**: Removed **997 exact duplicate rows**, reducing the dataset from 14,993 to 13,996 records.
2. **Numerical Parsing**:
   - `AskPrice`: Converted string currency representations (handles ₹ symbols, commas, Lakhs, and Crores) into standardized numeric Lakhs.
   - `kmDriven`: Stripped unit notations (`km`, `,`) and parsed to clean numeric floats.
3. **Categorical Standardization**:
   - Standardized split brand names (e.g., `'Toyota Land'` $\to$ Brand: `'Toyota'`, Model: `'Land Cruiser'`).
   - Cleaned web-scraping suffixes (e.g., `'VentoTest'` $\to$ `'Vento'`).
   - Standardized Fuel Types into 4 canonical categories: `Petrol`, `Diesel`, `Hybrid`, `CNG / Hybrid`.
   - Capitalized Transmission (`Manual`, `Automatic`) and Owner (`First`, `Second`).
4. **Outlier & Anomaly Filtering**:
   - Filtered out 9 invalid records with obvious data entry typo years (Year < 1995) and zero/token down payment listings (AskPrice $\le$ 0.1 Lakhs).
   - Final cleaned dataset size: **13,987 rows**.

---

## 🧠 Features Used

The final model pipeline utilizes **8 features** (3 numerical + 5 categorical):

### Numerical Features (3)
- `Year`: Year of vehicle manufacture
- `Age`: Age of vehicle in years (2024 - Year)
- `kmDriven`: Total distance driven in kilometers

### Categorical Features (5)
- `Brand`: Vehicle manufacturer (42 unique brands)
- `model`: Car model name (430 unique models)
- `Transmission`: `Manual`, `Automatic`
- `Owner`: `First`, `Second`
- `FuelType`: `Petrol`, `Diesel`, `CNG / Hybrid`, `Hybrid`

> **Note on PostedDate**: 99.8% of records in the dataset were posted in late 2024, and vehicle age is mathematically defined as `2024 - Year`. Raw `PostedDate` strings were excluded to avoid redundant cardinality and target leakage.

---

## ⚙️ Machine Learning Workflow

```
Raw Dataset (data/used_cars_dataset_v2.csv — 14,993 rows)
   │
   ▼
Data Cleaning & Deduplication (13,987 cleaned records)
   │
   ▼
Train/Test Split (80% Train [11,189 rows] / 20% Test [2,798 rows], random_state=42)
   │
   ▼
ColumnTransformer Preprocessing Pipeline
   ├── Numerical: SimpleImputer(strategy='median')
   └── Categorical: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
   │
   ▼
Model Benchmarking & Hyperparameter Tuning (6 Regressors evaluated)
   │
   ▼
Final Model Selection: Tuned Random Forest Regressor (R² = 0.7930, MAE = 1.8655 Lakhs)
   │
   ▼
Serialization (Saved to model/car_price_model_final.pkl & model_features.json)
   │
   ▼
Streamlit Web Deployment (Interactive dashboard at app.py)
```

---

## 📈 Models Compared & Performance

All models were evaluated on the **same unseen 20% test partition** (2,798 samples) using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score**:

| Model | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 4.6291 | 10.8891 | 0.4655 |
| **Random Forest Regressor** | 1.8650 | 6.7904 | 0.7922 |
| **Extra Trees Regressor** | 1.9014 | 7.5892 | 0.7404 |
| **HistGradientBoostingRegressor** | 2.8963 | 8.9597 | 0.6382 |
| **Gradient Boosting Regressor** | 2.5679 | 7.1605 | 0.7689 |
| **Tuned Random Forest Regressor (Selected)** | **1.8655** | **6.7767** | **0.7930** |

---

## 🏆 Final Model

The **Tuned Random Forest Regressor** was selected as the final production model due to its superior $R^2$ score (**0.7930**), lowest test RMSE (**6.7767 Lakhs**), and robust generalizability across budget, mid-range, and luxury segments.

### Hyperparameters:
- `n_estimators`: `300`
- `min_samples_split`: `3`
- `max_features`: `0.85`
- `random_state`: `42`
- `n_jobs`: `-1`

The complete end-to-end preprocessing transformer and tuned regressor are serialized together in [`model/car_price_model_final.pkl`](model/car_price_model_final.pkl). Model metadata is tracked in [`model/model_features.json`](model/model_features.json).

---

## 🔍 Hyundai Creta Sanity Validation

To confirm that the trained model naturally captures empirical pricing without bias or hardcoding, predictions were validated against matching historical listings in the dataset for a **2019 Hyundai Creta (Petrol, Manual, 1st Owner, 50,000–70,000 km)**:

| Metric | Dataset Records | Model Prediction |
| :--- | :---: | :---: |
| **Comparable Listings in Dataset** | 3 actual records | — |
| **Actual Price Range** | **₹ 9.50 – ₹ 9.85 Lakhs** | — |
| **Actual Mean Price** | **₹ 9.70 Lakhs** | — |
| **Actual Median Price** | **₹ 9.75 Lakhs** | — |
| **Model Inferred Output** | — | **₹ 9.51 Lakhs** |

> **Result**: The model prediction (₹ 9.51 Lakhs) sits naturally within the empirical asking price range (₹ 9.50L – ₹ 9.85L) without manual intervention.

---

## 🚘 Sample Predictions

| Vehicle Specification | Configuration | Predicted Asking Price |
| :--- | :--- | :---: |
| **2019 Hyundai Creta** | Petrol, Manual, 1st Owner, 60,000 km | **₹ 9.51 Lakhs** |
| **2021 Maruti Suzuki Swift** | Petrol, Manual, 1st Owner, 60,000 km | **₹ 6.20 Lakhs** |
| **2017 Honda City** | Petrol, Manual, 1st Owner, 60,000 km | **₹ 7.00 Lakhs** |
| **2018 Toyota Fortuner** | Petrol, Manual, 1st Owner, 60,000 km | **₹ 25.28 Lakhs** |
| **2020 BMW 3 Series** | Petrol, Manual, 1st Owner, 60,000 km | **₹ 22.84 Lakhs** |

---

## 🏗️ Project Architecture & Workflow

1. **User Interaction**: The user selects vehicle attributes across two intuitive cards (*Basic Information* and *Technical Details*).
2. **Client-Side Validation & Mapping**: Dynamic dropdown menus query the manufacturer's corresponding model catalog.
3. **Pipeline Ingestion**: Form inputs are structured into a single-row Pandas DataFrame matching the training schema.
4. **Feature Transformation**: `ColumnTransformer` executes median imputation on numerical values and one-hot encoding on categorical values.
5. **Inference Execution**: The loaded Random Forest Regressor predicts the expected asking price in Lakhs.
6. **HUD Presentation**: The formatted price is rendered in the green valuation card along with dynamic vehicle subtext and contextual disclaimers.

---

## 🛠️ Technology Stack

- **Web Framework**: [Streamlit](https://streamlit.io/) (v1.30+)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (v1.3+)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Styling**: Vanilla CSS, Google Fonts (*Space Grotesk*, *Inter*), SVG Graphics
- **Deployment Platform**: [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 📁 Repository Structure

```
car-price-prediction/
├── app.py                      # Production Streamlit web application
├── train_model_final.py        # Complete data cleaning, benchmarking & training pipeline
├── requirements.txt            # Minimal deployment dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
├── .gitattributes              # Git LFS / line ending rules
├── .streamlit/
│   └── config.toml             # Custom daylight theme configuration
├── assets/
│   └── hero_car.jpg            # Automotive hero banner image asset
├── data/
│   └── used_cars_dataset_v2.csv # Cleaned Indian used-car dataset
└── model/
    ├── car_price_model_final.pkl # Serialized Scikit-Learn ML pipeline
    └── model_features.json      # Model feature schema and test metrics
```

---

## 💻 Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/seemakurthisupraja/car-price-prediction.git
cd car-price-prediction
```

### 2. Create and Activate Virtual Environment
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

---

## 🏃 Running the Application

### Option A: Launch the Web App
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### Option B: Retrain the Machine Learning Model
```bash
python train_model_final.py
```
This script cleans the dataset, runs comparative benchmarks, tunes the Random Forest regressor, verifies the Hyundai Creta sanity check, and updates `model/car_price_model_final.pkl` and `model/model_features.json`.

---

## ☁️ Deployment

The project is configured for continuous deployment on **Streamlit Community Cloud**:

1. Fork or push the repository to GitHub: `https://github.com/seemakurthisupraja/car-price-prediction`
2. Connect your GitHub account at [share.streamlit.io](https://share.streamlit.io/).
3. Select the repository `seemakurthisupraja/car-price-prediction`, branch `master` (or `main`), and main file `app.py`.
4. Click **Deploy**. Streamlit Cloud provisions the environment and deploys the app at:
   👉 **[https://ai-used-car-valuation.streamlit.app/](https://ai-used-car-valuation.streamlit.app/)**

---

## ⚠️ Limitations & Disclaimer

- **Asking Price vs. Transaction Price**: The model predicts the **estimated used-car asking price** based on public listing data. Actual realized transaction values may vary.
- **Unmodeled Factors**: Physical vehicle condition, chassis/engine health, accident history, comprehensive insurance validity, individual service records, city-specific regional road taxes, and buyer-seller negotiations can influence the final market price.
- **Disclaimer**: *Valuations generated by AutoValue AI represent statistical estimates derived from historical and contemporary listing data and should serve as an informational benchmark rather than a legally binding appraisal.*

---

## 🔮 Future Improvements

- [ ] Integrate deep-learning regression architectures (e.g., TabNet, CatBoost, LightGBM) for comparison.
- [ ] Add regional city/RTO-based tier adjustments when geographic location data becomes available.
- [ ] Implement automated confidence interval estimation (e.g., Quantile Regression Forests) to display an expected price range alongside the point estimate.
- [ ] Build a REST API wrapper using FastAPI for programmatic vehicle valuation queries.

---

## 👩‍💻 Author

**Supraja Seemakurthi**  
- **GitHub**: [@seemakurthisupraja](https://github.com/seemakurthisupraja)  
- **Project**: [AutoValue AI — Used Car Price Prediction](https://github.com/seemakurthisupraja/car-price-prediction)
