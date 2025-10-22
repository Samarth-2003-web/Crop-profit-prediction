# app.py - Complete Agricultural Profit Predictor
import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ============================================================================
# CONFIGURATION
# ============================================================================
CROP_MAP = {'Paddy': 'paddy', 'Wheat': 'wheat', 'Maize': 'Corn', 'Bajra': 'Millet',
            'Jowar': 'Jowar', 'Ragi': 'ragi', 'Soybean': 'soyabean',
            'Groundnut': 'Groundnut', 'Cotton': 'Cotton', 'Sugarcane': 'Sugarcane'}

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================
def generate_price_data_inline():
    """Generate synthetic price data if not available"""
    np.random.seed(42)
    base_prices = {'Paddy': 1800, 'Wheat': 2000, 'Maize': 1600, 'Bajra': 1700,
                   'Jowar': 2500, 'Ragi': 2800, 'Soybean': 3500, 'Groundnut': 5000,
                   'Cotton': 5500, 'Sugarcane': 300}
    
    data = []
    for crop, base in base_prices.items():
        for year in range(2015, 2026):
            for month in range(1, 13):
                price = base * (1 + (year - 2015) * 0.06) * (1 + 0.1 * np.sin(2 * np.pi * month / 12)) * np.random.normal(1.0, 0.08)
                data.append({'Crop': crop, 'Year': year, 'Month': month, 'Price_Tonne': round(price * 10, 2)})
    
    df = pd.DataFrame(data)
    df['crop_type'] = df['Crop'].map(CROP_MAP)
    return df

def train_models():
    """Train yield and price prediction models"""
    print("🔧 Training models...")
    
    # Load yield data
    df_yield = None
    for path in ["Crop_yield_enriched.csv", "Crop_yield.csv"]:
        try:
            df_yield = pd.read_csv(path)
            print(f"✓ Loaded yield data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df_yield is None:
        print("❌ Error: Crop yield data not found.")
        return None
    
    # Encode categorical features
    le_season, le_crop, le_dist = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df_yield['season_enc'] = le_season.fit_transform(df_yield['season'])
    df_yield['crop_enc'] = le_crop.fit_transform(df_yield['crop_type'])
    df_yield['dist_enc'] = le_dist.fit_transform(df_yield['district'])
    
    # Check if enriched features exist
    enriched_features = ['soil_ph', 'soil_nitrogen', 'irrigation_type', 'rainfall_mm']
    is_enriched = all(col in df_yield.columns for col in enriched_features)
    
    # Select features
    if is_enriched:
        print("🌱 Using ENRICHED dataset")
        df_yield['soil_fertility'] = (df_yield['soil_nitrogen'] * df_yield['soil_phosphorus'] * df_yield['soil_potassium']) ** (1/3)
        df_yield['water_temp_interaction'] = df_yield['rainfall_mm'] * df_yield['avg_temperature']
        
        X = df_yield[['season_enc', 'year', 'crop_enc', 'dist_enc', 'area', 
                      'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'organic_matter',
                      'irrigation_type', 'rainfall_mm', 'avg_temperature',
                      'soil_fertility', 'water_temp_interaction']]
    else:
        print("⚠️  Using BASIC dataset")
        X = df_yield[['season_enc', 'year', 'crop_enc', 'dist_enc', 'area']]
    
    y = df_yield['crop_yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train yield model
    models_to_try = {}
    
    gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=8,
                                         min_samples_split=4, min_samples_leaf=2, subsample=0.8, random_state=42)
    gb_model.fit(X_train, y_train)
    models_to_try['GB'] = (gb_model, r2_score(y_test, gb_model.predict(X_test)))
    
    rf_model = RandomForestRegressor(n_estimators=400, max_depth=25, min_samples_split=3,
                                     min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models_to_try['RF'] = (rf_model, r2_score(y_test, rf_model.predict(X_test)))
    
    if HAS_XGBOOST:
        xgb_model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8,
                                min_child_weight=2, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        models_to_try['XGB'] = (xgb_model, r2_score(y_test, xgb_model.predict(X_test)))
    
    # Select best model
    best_model_name = max(models_to_try, key=lambda x: models_to_try[x][1])
    yield_model, yield_score = models_to_try[best_model_name]
    print(f"✓ {best_model_name} selected - R² Score: {yield_score:.4f}")
    
    # Load/generate price data
    df_price = None
    for path in ["crop_price_trends_2015_2025.csv"]:
        try:
            df_price = pd.read_csv(path)
            print(f"✓ Loaded price data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df_price is None:
        df_price = generate_price_data_inline()
        df_price.to_csv('crop_price_trends_2015_2025.csv', index=False)
        print("✓ Generated price data")
    
    if 'Month' not in df_price.columns:
        df_price['Month'] = df_price.groupby(['Crop', 'Year']).cumcount() + 1
    
    # Train price model
    df_price['crop_type'] = df_price['Crop'].map(CROP_MAP).fillna(df_price['Crop'])
    df_price = df_price[df_price['crop_type'].isin(le_crop.classes_)]
    df_price['crop_enc'] = le_crop.transform(df_price['crop_type'])
    
    Xp = df_price[['crop_enc', 'Year', 'Month']]
    yp = df_price['Price_Tonne']
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)
    
    price_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    price_model.fit(Xp_train, yp_train)
    price_score = r2_score(yp_test, price_model.predict(Xp_test))
    print(f"✓ Price model trained - R² Score: {price_score:.4f}")
    
    # Save models
    joblib.dump(yield_model, 'yield_model.joblib')
    joblib.dump(price_model, 'price_model.joblib')
    joblib.dump(le_season, 'le_season.joblib')
    joblib.dump(le_crop, 'le_crop.joblib')
    joblib.dump(le_dist, 'le_dist.joblib')
    
    # Save form options
    options = {
        'seasons': le_season.classes_.tolist(),
        'crops': le_crop.classes_.tolist(),
        'districts': le_dist.classes_.tolist()
    }
    with open('form_options.json', 'w') as f:
        json.dump(options, f)
    
    print("✓ Models and options saved")
    
    return {
        'yield_model': yield_model,
        'price_model': price_model,
        'le_season': le_season,
        'le_crop': le_crop,
        'le_dist': le_dist
    }

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_profit(season, crop_type, district, year, area, prev_production, prev_area, prev_cost, models_data):
    """Predict agricultural profit"""
    yield_model = models_data['yield_model']
    price_model = models_data['price_model']
    le_season = models_data['le_season']
    le_crop = models_data['le_crop']
    le_dist = models_data['le_dist']
    
    # Calculate cost per acre
    cost_per_area = prev_cost / prev_area if prev_area > 0 else 0
    
    # Calculate month from season
    season_months = {'Kharif': 7, 'Rabi': 1, 'Summer': 4, 'Whole Year': 7}
    month = season_months.get(season, 7)
    
    # Convert units and encode
    area_ha = area * 0.404686
    
    try:
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop_type])[0]
        dist_enc = le_dist.transform([district])[0]
    except ValueError as e:
        return {'error': str(e)}
    
    # Default agricultural parameters
    soil_ph = 6.8
    soil_nitrogen = 250.0
    soil_phosphorus = 30.0
    soil_potassium = 200.0
    organic_matter = 2.0
    irrigation_type = 1
    rainfall_mm = 800.0
    avg_temperature = 25.0
    
    # Calculate interaction features
    soil_fertility = (soil_nitrogen * soil_phosphorus * soil_potassium) ** (1/3)
    water_temp_interaction = rainfall_mm * avg_temperature
    
    # Create prediction DataFrame
    yield_features = pd.DataFrame([{
        'season_enc': season_enc,
        'year': year,
        'crop_enc': crop_enc,
        'dist_enc': dist_enc,
        'area': area_ha,
        'soil_ph': soil_ph,
        'soil_nitrogen': soil_nitrogen,
        'soil_phosphorus': soil_phosphorus,
        'soil_potassium': soil_potassium,
        'organic_matter': organic_matter,
        'irrigation_type': irrigation_type,
        'rainfall_mm': rainfall_mm,
        'avg_temperature': avg_temperature,
        'soil_fertility': soil_fertility,
        'water_temp_interaction': water_temp_interaction
    }])
    
    price_features = pd.DataFrame([{
        'crop_enc': crop_enc,
        'Year': year,
        'Month': month
    }])
    
    # Predict
    yield_tonnes_per_ha = yield_model.predict(yield_features)[0]
    price_per_tonne = price_model.predict(price_features)[0]
    
    # Convert to user units and calculate profit
    # Model predicts in tonnes/hectare, convert to quintals/acre
    # 1 tonne = 10 quintals
    # 1 hectare = 2.47105 acres
    # Therefore: tonnes/hectare × (10 quintals/tonne) ÷ (2.47105 acres/hectare) = quintals/acre
    yield_per_acre = yield_tonnes_per_ha * 4.047  # tonnes/ha → quintals/acre
    price_per_quintal = price_per_tonne / 10  # ₹/tonne → ₹/quintal
    
    total_yield = yield_per_acre * area
    revenue = total_yield * price_per_quintal
    cost = cost_per_area * area
    profit = revenue - cost
    
    return {
        'predicted_yield_per_acre': float(yield_per_acre),
        'predicted_price_per_quintal': float(price_per_quintal),
        'total_yield_quintals': float(total_yield),
        'total_revenue': float(revenue),
        'total_cost': float(cost),
        'profit': float(profit),
        'roi': float((profit / cost) * 100 if cost > 0 else 0),
        'profit_per_acre': float(profit / area)
    }

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)

# Load or train models on startup
models_data = None
form_options = None

try:
    models_data = {
        'yield_model': joblib.load('yield_model.joblib'),
        'price_model': joblib.load('price_model.joblib'),
        'le_season': joblib.load('le_season.joblib'),
        'le_crop': joblib.load('le_crop.joblib'),
        'le_dist': joblib.load('le_dist.joblib')
    }
    with open('form_options.json', 'r') as f:
        form_options = json.load(f)
    print("✓ Models loaded successfully")
except FileNotFoundError:
    print("⚠️  Model files not found. Training new models...")
    models_data = train_models()
    with open('form_options.json', 'r') as f:
        form_options = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/options')
def get_options():
    if form_options:
        return jsonify(form_options)
    return jsonify({'error': 'Options not loaded'}), 500

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if not models_data:
        return jsonify({'error': 'Models not loaded'}), 500
    
    data = request.get_json()
    
    try:
        season = data['season']
        crop_type = data['crop']
        district = data['district']
        year = int(data['year'])
        area = float(data['area'])
        prev_production = float(data['prev_production'])
        prev_area = float(data['prev_area'])
        prev_cost = float(data['prev_cost'])
        
        result = predict_profit(season, crop_type, district, year, area,
                              prev_production, prev_area, prev_cost, models_data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
