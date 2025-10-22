import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

CROP_MAP = {'Paddy': 'paddy', 'Wheat': 'wheat', 'Maize': 'Corn', 'Bajra': 'Millet',
            'Jowar': 'Jowar', 'Ragi': 'ragi', 'Soybean': 'soyabean',
            'Groundnut': 'Groundnut', 'Cotton': 'Cotton', 'Sugarcane': 'Sugarcane'}

def generate_price_data_inline():
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

def load_and_train_models():
    # Load enriched yield data (try enriched first, fallback to original)
    df_yield = None
    paths = [
        "Crop_yield_enriched.csv",
        r"c:\Users\samar\OneDrive\ドキュメント\Computer\Saciam\Crop_yield_enriched.csv",
        "Crop_yield.csv",
        r"c:\Users\samar\OneDrive\ドキュメント\Computer\Saciam\Crop_yield.csv"
    ]
    
    for path in paths:
        try:
            df_yield = pd.read_csv(path)
            print(f"✓ Loaded yield data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df_yield is None:
        print("❌ Error: Crop_yield.csv not found.")
        print("💡 Tip: Run 'python enrich_data.py' to create enriched dataset")
        return None
    
    # Check if enriched features exist
    enriched_features = ['soil_ph', 'soil_nitrogen', 'irrigation_type', 'rainfall_mm']
    is_enriched = all(col in df_yield.columns for col in enriched_features)
    
    # Encode categorical features
    le_season, le_crop, le_dist = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df_yield['season_enc'] = le_season.fit_transform(df_yield['season'])
    df_yield['crop_enc'] = le_crop.fit_transform(df_yield['crop_type'])
    df_yield['dist_enc'] = le_dist.fit_transform(df_yield['district'])
    
    # Select features based on dataset
    if is_enriched:
        print("🌱 Using ENRICHED dataset with agricultural features")
        X = df_yield[['season_enc', 'year', 'crop_enc', 'dist_enc', 'area', 
                      'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'organic_matter',
                      'irrigation_type', 'rainfall_mm', 'avg_temperature',
                      'fertilizer_npk', 'organic_fertilizer', 'pesticide_usage',
                      'seed_quality', 'mechanization', 'farmer_experience']]
    else:
        print("⚠️  Using BASIC dataset (limited features)")
        print("💡 Run 'python enrich_data.py' for better accuracy")
        X = df_yield[['season_enc', 'year', 'crop_enc', 'dist_enc', 'area']]
    
    y = df_yield['crop_yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    yield_model.fit(X_train, y_train)
    yield_score = r2_score(y_test, yield_model.predict(X_test))
    print(f"✓ Yield model trained - R² Score: {yield_score:.4f}")
    
    # Load price data
    df_price = None
    for path in [r"c:\Users\samar\OneDrive\ドキュメント\Computer\Saciam\crop_price_trends_2015_2025.csv", "crop_price_trends_2015_2025.csv"]:
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
    
    price_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    price_model.fit(Xp_train, yp_train)
    price_score = r2_score(yp_test, price_model.predict(Xp_test))
    print(f"✓ Price model trained - R² Score: {price_score:.4f}")
    
    return {'yield_model': yield_model, 'price_model': price_model, 'le_season': le_season,
            'le_crop': le_crop, 'le_dist': le_dist, 'yield_score': yield_score, 
            'price_score': price_score, 'is_enriched': is_enriched}

def predict_profit(season, crop_type, district, year, month, area, cost_per_area, models_data, 
                   soil_ph=6.5, soil_n=250, soil_p=25, soil_k=200, organic_matter=2.0,
                   irrigation=1, rainfall=600, temperature=25, fertilizer=150, 
                   organic_fert=50, pesticide=2, seed_quality=1, mechanization=1, experience=15):
    """
    Predict profit with optional agricultural features
    Uses enriched model if available, otherwise basic model
    """
    yield_model, price_model = models_data['yield_model'], models_data['price_model']
    le_season, le_crop, le_dist = models_data['le_season'], models_data['le_crop'], models_data['le_dist']
    is_enriched = models_data.get('is_enriched', False)
    
    # Convert and encode
    area_ha = area * 0.404686
    season_enc = le_season.transform([season])[0]
    crop_enc = le_crop.transform([crop_type])[0]
    dist_enc = le_dist.transform([district])[0]
    
    # Predict yield
    if is_enriched:
        yield_per_ha = yield_model.predict([[season_enc, year, crop_enc, dist_enc, area_ha,
                                             soil_ph, soil_n, soil_p, soil_k, organic_matter,
                                             irrigation, rainfall, temperature,
                                             fertilizer, organic_fert, pesticide,
                                             seed_quality, mechanization, experience]])[0]
    else:
        yield_per_ha = yield_model.predict([[season_enc, year, crop_enc, dist_enc, area_ha]])[0]
    
    # Predict price
    price_per_t = price_model.predict([[crop_enc, year, month]])[0]
    
    # Convert to user units and calculate profit
    yield_per_acre = yield_per_ha * 4.04686
    price_per_quintal = price_per_t / 10
    total_yield = yield_per_acre * area
    revenue = total_yield * price_per_quintal
    cost = cost_per_area * area
    profit = revenue - cost
    
    return {'predicted_yield_per_acre': yield_per_acre, 'predicted_price_per_quintal': price_per_quintal,
            'total_yield_quintals': total_yield, 'total_revenue': revenue, 'total_cost': cost,
            'profit': profit, 'roi': (profit / cost) * 100 if cost > 0 else 0,
            'profit_per_acre': profit / area}

def main():
    print("=" * 70)
    print("🌾 Agricultural Profit Predictor - ML Module")
    print("=" * 70)
    
    models_data = load_and_train_models()
    
    if models_data:
        # Example prediction
        result = predict_profit('Kharif', 'paddy', 'Ballari', 2024, 10, 250.0, 8000, models_data,
                               soil_ph=6.8, soil_n=280, soil_p=30, soil_k=220, organic_matter=2.5,
                               irrigation=2, rainfall=800, temperature=28, fertilizer=180,
                               organic_fert=60, pesticide=2, seed_quality=1, mechanization=1, experience=20)
        
        print(f"\n{'Example Prediction:':-^70}")
        print(f"📊 Yield: {result['predicted_yield_per_acre']:.2f} quintals/acre")
        print(f"💰 Price: ₹{result['predicted_price_per_quintal']:,.0f}/quintal")
        print(f"🌾 Total Yield: {result['total_yield_quintals']:,.2f} quintals")
        print(f"💵 Revenue: ₹{result['total_revenue']:,.0f} | Cost: ₹{result['total_cost']:,.0f}")
        print(f"{'✅ Profit' if result['profit'] > 0 else '❌ Loss'}: ₹{abs(result['profit']):,.0f} (ROI: {result['roi']:.1f}%)")
        print(f"\nOptions: {sorted(models_data['le_season'].classes_.tolist())} | {len(models_data['le_crop'].classes_)} crops | {len(models_data['le_dist'].classes_)} districts")
        print("=" * 70)

if __name__ == "__main__":
    main()
