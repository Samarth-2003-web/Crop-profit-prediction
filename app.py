# app.py
import joblib
import json
from flask import Flask, request, jsonify, render_template

# --- Helper Function (Copied from your script) ---
# We need this function available for the server
def predict_profit(season, crop_type, district, year, area, prev_production_quintals, prev_area, prev_cost, models_data):
    # Extract models and encoders
    yield_model, price_model = models_data['yield_model'], models_data['price_model']
    le_season, le_crop, le_dist = models_data['le_season'], models_data['le_crop'], models_data['le_dist']
    
    # Calculate cost per acre from previous year data
    cost_per_area = prev_cost / prev_area if prev_area > 0 else 0
    
    # Calculate month from season (average month for each season)
    season_months = {'Kharif': 7, 'Rabi': 1, 'Summer': 4, 'Whole Year': 7}
    month = season_months.get(season, 7)
    
    # Convert units and encode
    area_ha = area * 0.404686
    
    # Handle unknown labels gracefully
    try:
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop_type])[0]
        dist_enc = le_dist.transform([district])[0]
    except ValueError as e:
        print(f"Encoding error: {e}")
        # Return a dictionary indicating failure
        return {'error': str(e)}

    # Default agricultural parameters for enriched model (using average values)
    soil_ph = 6.8
    soil_nitrogen = 250.0
    soil_phosphorus = 30.0
    soil_potassium = 200.0
    organic_matter = 2.0
    irrigation_type = 1
    rainfall_mm = 800.0
    avg_temperature = 25.0
    fertilizer_npk = 150.0
    organic_fertilizer = 50.0
    pesticide_usage = 1
    seed_quality = 1
    mechanization = 1
    farmer_experience = 10

    # Calculate interaction features (same as in ml_model.py)
    soil_fertility = (soil_nitrogen * soil_phosphorus * soil_potassium) ** (1/3)
    npk_balance = fertilizer_npk / (soil_nitrogen + soil_phosphorus + soil_potassium + 1)
    water_temp_interaction = rainfall_mm * avg_temperature
    quality_score = (seed_quality + mechanization + irrigation_type) / 3
    total_fertilizer = fertilizer_npk + organic_fertilizer
    experience_quality = farmer_experience * seed_quality

    # Predict yield with all 25 features
    yield_per_ha = yield_model.predict([[
        season_enc, year, crop_enc, dist_enc, area_ha,
        soil_ph, soil_nitrogen, soil_phosphorus, soil_potassium, organic_matter,
        irrigation_type, rainfall_mm, avg_temperature, fertilizer_npk, organic_fertilizer,
        pesticide_usage, seed_quality, mechanization, farmer_experience,
        soil_fertility, npk_balance, water_temp_interaction, quality_score, total_fertilizer, experience_quality
    ]])[0] 
    price_per_t = price_model.predict([[crop_enc, year, month]])[0]
    
    # Convert to user units and calculate profit
    yield_per_acre = yield_per_ha * 4.04686 # 1 t/ha ≈ 4.047 quintals/acre
    price_per_quintal = price_per_t / 10
    
    total_yield = yield_per_acre * area
    revenue = total_yield * price_per_quintal
    cost = cost_per_area * area
    profit = revenue - cost
    
    return {'predicted_yield_per_acre': yield_per_acre, 'predicted_price_per_quintal': price_per_quintal,
            'total_yield_quintals': total_yield, 'total_revenue': revenue, 'total_cost': cost,
            'profit': profit, 'roi': (profit / cost) * 100 if cost > 0 else 0,
            'profit_per_acre': profit / area}
# --- End of Helper Function ---


# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load models and options (globally, on startup)
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
    print("✓ Models and form options loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model files not found. Run 'python train.py' first.")
    models_data = None
    form_options = None

# 3. Define Routes

@app.route('/')
def home():
    """Serves the main HTML page."""
    # We will create 'index.html' next
    return render_template('index.html') 

@app.route('/options')
def get_options():
    """Provides the options for the form dropdowns."""
    if form_options:
        return jsonify(form_options)
    return jsonify({'error': 'Options not loaded'}), 500

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles the prediction request."""
    if not models_data:
        return jsonify({'error': 'Models not loaded'}), 500

    # Get data from the HTML form (sent as JSON)
    data = request.get_json()

    try:
        # Extract and convert types
        season = data['season']
        crop_type = data['crop']
        district = data['district']
        year = int(data['year'])
        area = float(data['area'])
        prev_production = float(data['prev_production'])
        prev_area = float(data['prev_area'])
        prev_cost = float(data['prev_cost'])

        # Call the prediction function
        result = predict_profit(season, crop_type, district, year, area, 
                              prev_production, prev_area, prev_cost, models_data)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

# 4. Run the App
if __name__ == '__main__':
    # Make sure to create a 'templates' folder and put 'index.html' inside it
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
