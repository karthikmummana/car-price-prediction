from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Define deployment-safe cross-platform base paths using pathlib
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "car_price_model_final.pkl"
HERO_IMG_PATH = BASE_DIR / "assets" / "hero_car.jpg"
DATA_PATH = BASE_DIR / "data" / "train.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "train.csv"

# Set page config for a premium and professional appearance
st.set_page_config(
    page_title="AutoValue AI — Used Car Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to load and encode local image to base64
def get_base64_image(image_path):
    path = Path(image_path)
    if path.exists():
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return ""

hero_bg_b64 = get_base64_image(HERO_IMG_PATH)

# Custom CSS matching the reference design exactly
st.markdown(f"""
<style>
/* Hide standard Streamlit header and footer elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Custom Fonts & Global Styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

.stApp {{
    background-color: #EEF4FA !important;
    background-image: radial-gradient(at 90% 10%, rgba(24, 183, 242, 0.06) 0px, transparent 45%),
                      radial-gradient(at 10% 90%, rgba(7, 136, 209, 0.05) 0px, transparent 45%) !important;
    color: #102A43 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* Centered Content Container */
.main .block-container {{
    max-width: 1220px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 0.75rem !important;
    padding-bottom: 2.5rem !important;
}}

div[data-testid="stVerticalBlock"] {{
    gap: 0.85rem !important;
}}

/* Top Header */
.main-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #D6E2EE;
    padding-bottom: 0.75rem;
    margin-bottom: 1.25rem;
}}

.main-header-left {{
    display: flex;
    flex-direction: column;
}}

.main-header-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    color: #102A43;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    line-height: 1.2;
    letter-spacing: -0.02em;
}}

.main-header-subtitle {{
    font-size: 0.88rem;
    color: #627D98;
    font-weight: 500;
    margin-top: 0.15rem;
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    background-color: #E0F2FE;
    color: #0788D1;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    border: 1px solid #BAE6FD;
    letter-spacing: 0.06em;
}}

.status-badge-dot {{
    width: 7px;
    height: 7px;
    background-color: #0AA66A;
    border-radius: 50%;
    margin-right: 7px;
    display: inline-block;
    box-shadow: 0 0 6px rgba(10, 166, 106, 0.6);
}}

/* Automotive Hero Section */
.hero-wrapper {{
    position: relative;
    border-radius: 18px;
    background-image: 
        linear-gradient(to right, rgba(238, 244, 250, 0.96) 0%, rgba(238, 244, 250, 0.88) 42%, rgba(238, 244, 250, 0.15) 65%, rgba(238, 244, 250, 0.0) 100%),
        url('data:image/jpeg;base64,{hero_bg_b64}');
    background-size: cover;
    background-position: center right;
    background-repeat: no-repeat;
    padding: 3.2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 32px rgba(16, 42, 67, 0.08), 0 2px 6px rgba(16, 42, 67, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.95);
    overflow: hidden;
}}

.hero-grid {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 2rem;
}}

.hero-left {{
    flex: 1.35;
    max-width: 600px;
}}

.hero-badge {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #0788D1;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.85rem;
    color: #102A43 !important;
    letter-spacing: -0.02em;
}}

.hero-desc {{
    font-size: 1.02rem;
    color: #486581 !important;
    line-height: 1.55;
    margin-bottom: 1.35rem;
    font-weight: 400;
}}

.hero-features {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}}

.hero-feature-item {{
    font-size: 0.85rem;
    font-weight: 600;
    color: #0788D1;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    border: 1px solid #D6E2EE;
    box-shadow: 0 2px 6px rgba(16, 42, 67, 0.04);
}}

.hero-right {{
    flex: 0.75;
    display: flex;
    justify-content: flex-end;
}}

/* Translucent HUD Valuation Visual Card */
.hud-visual-card {{
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 215px;
    box-shadow: 0 14px 35px rgba(16, 42, 67, 0.16), 0 2px 8px rgba(16, 42, 67, 0.06);
    text-align: center;
}}

.hud-ring-wrapper {{
    position: relative;
    width: 76px;
    height: 76px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.85rem;
}}

.hud-ring {{
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    border: 2px dashed rgba(7, 136, 209, 0.45);
    animation: spin 30s linear infinite;
}}

@keyframes spin {{
    100% {{ transform: rotate(360deg); }}
}}

.hud-ring-inner {{
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
    border: 2px solid #0788D1;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(7, 136, 209, 0.25);
}}

.hud-icon {{
    font-size: 1.6rem;
}}

.hud-currency-tag {{
    position: absolute;
    bottom: -2px;
    right: -2px;
    background: #0788D1;
    color: #FFFFFF;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 800;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid #FFFFFF;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
}}

.hud-label-text {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    color: #627D98;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}}

.hud-value-text {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    color: #0788D1;
    font-weight: 800;
    margin-top: 0.2rem;
    letter-spacing: 0.05em;
}}

/* Section Titles */
.section-header-block {{
    margin-top: 0.5rem;
    margin-bottom: 0.85rem;
}}

.section-main-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #102A43;
    margin-bottom: 0.15rem;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.section-subtitle {{
    color: #627D98;
    font-size: 0.92rem;
    margin: 0;
}}

/* Two Vehicle Information Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF !important;
    border: 1px solid #D6E2EE !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 18px rgba(16, 42, 67, 0.04), 0 1px 3px rgba(16, 42, 67, 0.02) !important;
}}

.card-inner-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0788D1;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid #EEF4FA;
    padding-bottom: 0.65rem;
}}

/* Inputs Styling */
div[data-testid="stWidgetLabel"] p {{
    color: #102A43 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.25rem !important;
}}

.stSelectbox div[data-baseweb="select"], .stNumberInput input {{
    background: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    height: 42px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    font-size: 0.92rem !important;
    color: #102A43 !important;
}}

.stSelectbox div[data-baseweb="select"]:hover, .stNumberInput input:hover {{
    border-color: #0788D1 !important;
    background: #FFFFFF !important;
}}

.stSelectbox div[data-baseweb="select"]:focus-within, .stNumberInput input:focus {{
    border-color: #0788D1 !important;
    box-shadow: 0 0 0 2px rgba(7, 136, 209, 0.2) !important;
    background: #FFFFFF !important;
}}

/* Sliders Blue Automotive Accent */
div[data-baseweb="slider"] div[style*="background"] {{
    background: #0788D1 !important;
    background-color: #0788D1 !important;
}}

div[data-baseweb="slider"] div[role="slider"] {{
    background-color: #0788D1 !important;
    border-color: #0788D1 !important;
    box-shadow: 0 0 0 3px rgba(7, 136, 209, 0.25) !important;
}}

div[data-testid="stSliderTickBar"] {{
    display: none !important;
}}

div[data-testid="stThumbValue"] {{
    color: #0788D1 !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}

/* Dropdown popover styling */
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
    background-color: #FFFFFF !important;
    color: #102A43 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px rgba(16, 42, 67, 0.12) !important;
}}

li[role="option"] {{
    color: #102A43 !important;
    background-color: #FFFFFF !important;
    font-size: 0.9rem !important;
    padding: 8px 14px !important;
}}

li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
    background-color: #E0F2FE !important;
    color: #0788D1 !important;
    font-weight: 600 !important;
}}

/* Centered Estimate CTA Button */
div.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, #0788D1 0%, #18B7F2 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.18rem !important;
    border-radius: 12px !important;
    padding: 0.9rem 2.25rem !important;
    border: none !important;
    box-shadow: 0 6px 22px rgba(7, 136, 209, 0.38) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
}}

div.stButton > button:hover {{
    background: linear-gradient(135deg, #0672b0 0%, #0788D1 100%) !important;
    box-shadow: 0 8px 26px rgba(7, 136, 209, 0.5) !important;
    transform: translateY(-2px) !important;
}}

div.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* Wide Green Prediction Result Card */
.result-valuation-card-wide {{
    background: #EAF8F1 !important;
    border: 2px solid #A7F3D0 !important;
    border-radius: 18px;
    padding: 2.25rem 2.5rem;
    box-shadow: 0 10px 30px rgba(10, 166, 106, 0.1), 0 2px 8px rgba(10, 166, 106, 0.04);
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}}

.result-card-left-art {{
    flex: 0 0 110px;
    display: flex;
    justify-content: center;
    align-items: center;
}}

.result-card-center-content {{
    flex: 1;
    text-align: center;
    padding: 0 1.5rem;
}}

.result-card-right-art {{
    flex: 0 0 110px;
    display: flex;
    justify-content: center;
    align-items: center;
}}

.result-badge-success {{
    display: inline-flex;
    align-items: center;
    background-color: #D1FAE5;
    color: #065F46;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    border: 1px solid #6EE7B7;
    margin-bottom: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

.result-badge-dot {{
    width: 6px;
    height: 6px;
    background-color: #0AA66A;
    border-radius: 50%;
    margin-right: 6px;
    display: inline-block;
}}

.result-valuation-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    color: #065F46;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}}

.result-valuation-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    color: #0AA66A;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
}}

.result-valuation-footer {{
    font-size: 0.92rem;
    color: #047857;
    opacity: 0.9;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.45;
}}

/* Subtle Disclaimer Card */
.disclaimer-card {{
    background: #FFFFFF;
    border: 1px solid #D6E2EE;
    border-radius: 10px;
    padding: 0.95rem 1.5rem;
    margin-top: 1.5rem;
    font-size: 0.84rem;
    color: #627D98;
    line-height: 1.5;
    text-align: center;
    box-shadow: 0 1px 4px rgba(16, 42, 67, 0.03);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}}

/* Responsive Breakpoints */
@media (max-width: 850px) {{
    .hero-grid {{
        flex-direction: column;
        align-items: flex-start;
    }}
    .hero-right {{
        width: 100%;
        justify-content: center;
        margin-top: 1rem;
    }}
    .hero-title {{
        font-size: 2.1rem;
    }}
    .result-valuation-value {{
        font-size: 2.6rem;
    }}
    .result-card-left-art, .result-card-right-art {{
        display: none;
    }}
}}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-header-left">
        <div class="main-header-title">🚗 AutoValue AI</div>
        <div class="main-header-subtitle">AI-Powered Used Car Valuation</div>
    </div>
    <div class="status-badge">
        <span class="status-badge-dot"></span>AI VALUATION
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown(f"""
<div class="hero-wrapper">
    <div class="hero-grid">
        <div class="hero-left">
            <div class="hero-badge">AI-POWERED USED CAR VALUATION</div>
            <h1 class="hero-title">What's your car worth?</h1>
            <p class="hero-desc">
                Get an instant estimate of your used car's resale value using machine learning trained on historical used-car data.
            </p>
            <div class="hero-features">
                <span class="hero-feature-item">✓ Data-driven estimate</span>
                <span class="hero-feature-item">✓ Instant prediction</span>
                <span class="hero-feature-item">✓ Easy to use</span>
            </div>
        </div>
        <div class="hero-right">
            <div class="hud-visual-card">
                <div class="hud-ring-wrapper">
                    <div class="hud-ring"></div>
                    <div class="hud-ring-inner">
                        <span class="hud-icon">🚗</span>
                    </div>
                    <div class="hud-currency-tag">₹</div>
                </div>
                <div class="hud-label-text">AUTO VALUATION</div>
                <div class="hud-value-text">AI-POWERED</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Vehicle Details Title
st.markdown("""
<div class="section-header-block">
    <div class="section-main-title">🚗 Vehicle Details</div>
    <p class="section-subtitle">Enter your car specifications</p>
</div>
""", unsafe_allow_html=True)

# Check if model exists
if not MODEL_PATH.exists():
    st.error(f"⚠️ Model file not found at: {MODEL_PATH}")
    st.stop()

# Cache resource loader
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model_pipeline = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load model pipeline. Error: {str(e)}")
    st.stop()

# Robust Brand and Model Parser
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

@st.cache_data
def get_brand_model_mapping():
    if not DATA_PATH.exists():
        return {}
    df = pd.read_csv(DATA_PATH)
    df['Brand'] = df['Name'].apply(lambda x: parse_name(x)[0])
    df['Model'] = df['Name'].apply(lambda x: parse_name(x)[1])
    
    mapping = {}
    for brand in sorted(df['Brand'].unique()):
        models = sorted(df[df['Brand'] == brand]['Model'].unique())
        mapping[brand] = models
    return mapping

brand_model_mapping = get_brand_model_mapping()

# Input Options (matching dataset schema)
locations = [
    'Ahmedabad', 'Bangalore', 'Chennai', 'Coimbatore', 'Delhi', 'Hyderabad', 
    'Jaipur', 'Kochi', 'Kolkata', 'Mumbai', 'Pune'
]
fuels = ['Diesel', 'Petrol', 'Electric']
transmissions = ['Manual', 'Automatic']
owners = ['First', 'Second', 'Third', 'Fourth & Above']

# Initialize session states for storing predictions across updates
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Two Large Cards Side-by-Side
input_col1, input_col2 = st.columns(2)

with input_col1:
    # LEFT CARD — 🚘 Basic Information
    with st.container(border=True):
        st.markdown('<div class="card-inner-title">🚘 Basic Information</div>', unsafe_allow_html=True)
        
        # Row 1: Car Brand & Location
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            brand_list = list(brand_model_mapping.keys()) if brand_model_mapping else ['Audi', 'BMW', 'Hyundai', 'Maruti', 'Mercedes-Benz', 'Toyota']
            default_brand_idx = brand_list.index("Audi") if "Audi" in brand_list else 0
            brand = st.selectbox("Car Brand", options=brand_list, index=default_brand_idx)
        with r1_c2:
            location = st.selectbox("Location", options=sorted(locations), index=0)
            
        # Row 2: Car Model & Fuel Type
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            available_models = brand_model_mapping.get(brand, [])
            default_model_idx = 0
            if "A3" in available_models:
                default_model_idx = available_models.index("A3")
            elif "A4" in available_models:
                default_model_idx = available_models.index("A4")
            model = st.selectbox("Car Model", options=available_models if available_models else ['A3', 'A4', 'A6', 'Q3', 'Q5', 'Q7'], index=default_model_idx)
        with r2_c2:
            fuel = st.selectbox("Fuel Type", options=fuels, index=0)
            
        # Row 3: Kilometers Driven & Model Year
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=500000, value=50000, step=1000)
        with r3_c2:
            year = st.slider("Model Year", min_value=1998, max_value=2019, value=2017, step=1)

with input_col2:
    # RIGHT CARD — ⚙ Technical Details
    with st.container(border=True):
        st.markdown('<div class="card-inner-title">⚙ Technical Details</div>', unsafe_allow_html=True)
        
        # Row 1: Transmission & Owner Type
        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            transmission = st.selectbox("Transmission", options=transmissions, index=0)
        with r4_c2:
            owner = st.selectbox("Owner Type", options=owners, index=0)
            
        # Row 2: Engine, Max Power, Mileage
        r5_c1, r5_c2, r5_c3 = st.columns(3)
        with r5_c1:
            engine = st.number_input("Engine (CC)", min_value=600, max_value=6000, value=1200, step=50)
        with r5_c2:
            max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=85.0, step=1.0)
        with r5_c3:
            mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=35.0, value=18.0, step=0.1)
            
        # Row 3: Number of Seats
        seats = st.slider("Number of Seats", min_value=2, max_value=10, value=5, step=1)

# Helper function to format prediction output in Lakhs
def format_lakhs(amount):
    if amount < 0:
        amount = 0.0
    return f"₹ {amount:.2f} Lakhs"

# Estimate Button Column Setup (Centered Prominent CTA Button)
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 1.6, 1])
with btn_col2:
    estimate_clicked = st.button("✨ Estimate Resale Value", use_container_width=True)

if estimate_clicked:
    # Build dataframe for inference matching trained model feature schema
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model],
        'Location': [location],
        'Fuel_Type': [fuel],
        'Transmission': [transmission],
        'Owner_Type': [owner],
        'Year': [year],
        'Kilometers_Driven': [km_driven],
        'Mileage': [mileage],
        'Engine': [engine],
        'Power': [max_power],
        'Seats': [seats]
    })
    
    try:
        # Run through model pipeline
        prediction = model_pipeline.predict(input_data)
        predicted_price = prediction[0]
        formatted_price = format_lakhs(predicted_price)
        
        # Persist result to session state
        st.session_state.prediction = formatted_price
    except Exception as e:
        st.error(f"Prediction failed. Error: {str(e)}")

# Display results if present in session state (Wide Green Valuation Card)
if st.session_state.prediction is not None:
    st.markdown(f"""
    <div class="result-valuation-card-wide">
        <div class="result-card-left-art">
            <svg width="100" height="75" viewBox="0 0 100 75" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity: 0.65;">
                <path d="M18 45 C18 45 22 32 32 28 C42 24 54 24 64 28 C74 32 78 45 78 45" stroke="#0AA66A" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 45 L84 45 C87 45 90 48 90 52 L90 58 C90 61 87 64 84 64 L12 64 C9 64 6 61 6 58 L6 52 C6 48 9 45 12 45 Z" stroke="#0AA66A" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M30 32 L33 43 L46 43 L46 28 C39 28 34 29 30 32 Z" stroke="#0AA66A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M50 28 L50 43 L63 43 L66 32 C62 29 57 28 50 28 Z" stroke="#0AA66A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="16" cy="53" r="3" fill="#0AA66A"/>
                <circle cx="80" cy="53" r="3" fill="#0AA66A"/>
                <path d="M32 58 L64 58" stroke="#0AA66A" stroke-width="2" stroke-linecap="round"/>
                <circle cx="76" cy="50" r="15" fill="#EAF8F1" stroke="#0AA66A" stroke-width="2.5"/>
                <text x="76" y="56" font-family="'Space Grotesk', sans-serif" font-size="15" font-weight="bold" fill="#0AA66A" text-anchor="middle">₹</text>
            </svg>
        </div>
        <div class="result-card-center-content">
            <div class="result-badge-success">
                <span class="result-badge-dot"></span>AI PREDICTION
            </div>
            <div class="result-valuation-title">YOUR ESTIMATED MARKET VALUE</div>
            <div class="result-valuation-value">{st.session_state.prediction}</div>
            <div class="result-valuation-footer">Estimated resale price based on the vehicle specifications provided.</div>
        </div>
        <div class="result-card-right-art">
            <svg width="100" height="75" viewBox="0 0 100 75" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity: 0.65;">
                <path d="M12 56 L36 40 L60 46 L88 16" stroke="#0AA66A" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M72 16 L88 16 L88 32" stroke="#0AA66A" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                <rect x="14" y="46" width="11" height="18" rx="2" stroke="#0AA66A" stroke-width="2.2" fill="none"/>
                <rect x="34" y="38" width="11" height="26" rx="2" stroke="#0AA66A" stroke-width="2.2" fill="none"/>
                <rect x="54" y="30" width="11" height="34" rx="2" stroke="#0AA66A" stroke-width="2.2" fill="none"/>
                <rect x="74" y="22" width="11" height="42" rx="2" stroke="#0AA66A" stroke-width="2.2" fill="none"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Subtle Disclaimer Card
st.markdown("""
<div class="disclaimer-card">
    <span style="font-size: 1rem; color: #0788D1;">ⓘ</span> Estimated from historical used car data. Actual resale value may vary depending on vehicle condition, service history, market demand and factors.
</div>
""", unsafe_allow_html=True)
