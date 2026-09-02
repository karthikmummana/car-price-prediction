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
    raise FileNotFoundError(f"Dataset not found at {dataset_path}.")

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
    
    # 1. Extract Brand
    if len(words) > 1 and words[0].lower() == 'land' and words[1].lower() == 'rover':
        brand = 'Land Rover'
        remaining = words[2:]
    else:
        brand = words[0]
        remaining = words[1:]
        
    if brand.lower() == 'isuzu':
        brand = 'Isuzu'
        
    # 2. Handle 'New' prefix
    if len(remaining) > 0 and remaining[0].lower() == 'new':
        remaining = remaining[1:]
        
    # 3. Extract Model
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
# STEP 4 — PIPELINE SETUP
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
# STEP 5 — MODEL TRAINING
# ==========================================
print("\n=== MODEL TRAINING ===")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', rf_model)
])

pipeline.fit(X_train, y_train)

# Save pipeline as model/car_price_model.pkl (do not overwrite model_final.pkl)
model_path = "model/car_price_model.pkl"
joblib.dump(pipeline, model_path)
print(f"Saved pipeline to: {model_path}")
