import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure output directory exists
os.makedirs("model", exist_ok=True)

# ==========================================
# STEP 1 — DATASET LOADING & PARSING
# ==========================================
print("=== LOADING AND PARSING DATASET ===")
dataset_path = "data/train.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please make sure it is copied correctly.")

df = pd.read_csv(dataset_path)
print(f"Original Shape: {df.shape}")

# Drop index and New_Price (excessive missing values)
cols_to_drop = [c for c in ['Unnamed: 0', 'New_Price'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

# Parse Brand and Model from Name
def parse_name(name):
    if not isinstance(name, str):
        return 'Unknown', 'Unknown'
    words = name.split()
    if len(words) == 0:
        return 'Unknown', 'Unknown'
    
    # 1. Extract Brand (handle multi-word 'Land Rover')
    if len(words) > 1 and words[0].lower() == 'land' and words[1].lower() == 'rover':
        brand = 'Land Rover'
        remaining = words[2:]
    else:
        brand = words[0]
        remaining = words[1:]
        
    if brand.lower() == 'isuzu':
        brand = 'Isuzu'
        
    # 2. Handle 'New' prefix in remaining words
    if len(remaining) > 0 and remaining[0].lower() == 'new':
        remaining = remaining[1:]
        
    # 3. Extract Model from remaining words (handle known multi-word models)
    if len(remaining) == 0:
        return brand, 'Unknown'
        
    if len(remaining) > 1 and remaining[0].lower() == 'grand' and remaining[1].lower() in ['i10', 'vitara']:
        model = 'Grand ' + remaining[1]
    elif len(remaining) > 1 and remaining[0].lower() == 'wagon' and remaining[1].lower() == 'r':
        model = 'Wagon R'
    elif len(remaining) > 1 and remaining[0] in ['1', '3', '5', '6', '7', '8'] and remaining[1].lower() == 'series':
        model = remaining[0] + ' Series'
    elif len(remaining) > 1 and remaining[0].lower() == 'range' and remaining[1].lower() == 'rover':
        model = 'Range Rover'
    elif len(remaining) > 1 and remaining[1].lower() in ['class', 'class,']:
        model = remaining[0] + ' Class'
    elif len(remaining) > 1 and remaining[0].lower() == 's' and remaining[1].lower() == 'cross':
        model = 'S Cross'
    elif len(remaining) > 1 and remaining[0].lower() == 'alto' and remaining[1].lower() in ['k10', '800']:
        model = 'Alto ' + remaining[1]
    elif len(remaining) > 1 and remaining[0].lower() == 'pajero' and remaining[1].lower() == 'sport':
        model = 'Pajero Sport'
    elif len(remaining) > 1 and remaining[0].lower() == 'innova' and remaining[1].lower() == 'crysta':
        model = 'Innova Crysta'
    elif len(remaining) > 1 and remaining[0].lower() == 'corolla' and remaining[1].lower() == 'altis':
        model = 'Corolla Altis'
    elif len(remaining) > 1 and remaining[0].lower() == 'swift' and remaining[1].lower() == 'dzire':
        model = 'Swift Dzire'
    else:
        model = remaining[0]
        
    return brand, model

df['Brand'] = df['Name'].apply(lambda x: parse_name(x)[0])
df['Model'] = df['Name'].apply(lambda x: parse_name(x)[1])

# Keep Name out of training features
df = df.drop(columns=['Name'])

# ==========================================
# STEP 2 — DATA CLEANING & CONVERSION
# ==========================================
def clean_numeric_units(val):
    if pd.isna(val) or str(val).strip().lower() == 'null':
        return np.nan
    val_str = str(val).split()[0]
    try:
        num = float(val_str)
        # Treat 0 in Mileage as missing (NaN) since 0.0 is unrealistic
        if num == 0.0:
            return np.nan
        return num
    except ValueError:
        return np.nan

df['Mileage'] = df['Mileage'].apply(clean_numeric_units)
df['Engine'] = df['Engine'].apply(clean_numeric_units)
df['Power'] = df['Power'].apply(clean_numeric_units)
df['Seats'] = pd.to_numeric(df['Seats'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# ==========================================
# STEP 3 — FEATURE SELECTION & SPLIT
# ==========================================
categorical_cols = ['Brand', 'Model', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
numerical_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
target_col = 'Price'

X = df[categorical_cols + numerical_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4 — PIPELINE SETUP (PREVENT LEAKAGE)
# ==========================================
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ==========================================
# STEP 5 — MODEL TRAINING & COMPARISON
# ==========================================
print("\n=== MODEL TRAINING AND COMPARISON ===")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)
}

trained_pipelines = {}
metrics_results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Preprocessing is fitted ONLY on X_train
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline
    
    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Evaluate
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    
    metrics_results[name] = {
        "Train_R2": r2_train,
        "Test_R2": r2_test,
        "Test_MAE": mae_test,
        "Test_RMSE": rmse_test
    }
    
    print(f"\n{name} metrics:")
    print(f"  Training R² : {r2_train:.4f}")
    print(f"  Test R²     : {r2_test:.4f}")
    print(f"  Test MAE    : {mae_test:.4f} Lakhs")
    print(f"  Test RMSE   : {rmse_test:.4f} Lakhs")

# ==========================================
# STEP 6 — SAVE MODEL & METADATA
# ==========================================
best_model_name = "Random Forest Regressor"
best_pipeline = trained_pipelines[best_model_name]

# Save pipeline as model/car_price_model_final.pkl
new_model_path = "model/car_price_model_final.pkl"
joblib.dump(best_pipeline, new_model_path)
print(f"\nSaved best model pipeline to: {new_model_path}")

# Save features and metadata to model/model_features.json
features_info = {
    "numerical_features": numerical_cols,
    "categorical_features": categorical_cols,
    "target_column": target_col,
    "dataset_name": "train(3).csv",
    "model_type": best_model_name
}
features_json_path = "model/model_features.json"
with open(features_json_path, 'w') as f:
    json.dump(features_info, f, indent=4)
print(f"Saved model features information to: {features_json_path}")
