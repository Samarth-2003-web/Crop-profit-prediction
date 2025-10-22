# 🌾 Agricultural Profit Predictor


A machine learning-powered web application that helps farmers predict crop yields, prices, and profitability based on historical data and agricultural parameters.A pure Python Machine Learning module that predicts crop yield, price, and calculates expected profit for farmers.



## 📊 Features



- **Yield Prediction**: ML model trained on enriched agricultural data (53% R² accuracy)- **Crop Yield Prediction**: Random Forest model predicts yield per acre based on historical data

- **Price Forecasting**: Predict crop prices based on historical trends (97% R² accuracy)- **Price Forecasting**: Predicts crop prices with year-based trends (6% annual increase)

- **Profit Analysis**: Calculate expected revenue, costs, ROI, and profit per acre- **Profit Calculator**: Calculates net profit using the formula:

- **User-Friendly Interface**: Modern, responsive web design with animations  ```

- **Farmer-Centric Inputs**: Simple inputs based on previous year's data  Profit = (Yield × Price × Area) - (Cost × Area)

  ```

## Technologies Used- **Farmer-Friendly Units**: Uses acres and quintals (standard Indian units)

- **10+ Crops Supported**: Paddy, Wheat, Maize, Cotton, Sugarcane, and more

- **Backend**: Flask (Python web framework)- **Year-Based Predictions**: Accounts for inflation and market trends

- **Machine Learning**: scikit-learn (RandomForestRegressor)

- **Data Processing**: pandas, numpy## 📈 Model Performance

- **Model Persistence**: joblib

- **Frontend**: HTML5, CSS3, JavaScript- **Yield Model Accuracy**: 78.4% R² Score

- **Price Model Accuracy**: 94.5% R² Score

## Installation- **Algorithm**: Random Forest Regressor



1. Clone the repository:## 🚀 Quick Start

```bash

git clone https://github.com/YOUR_USERNAME/crop-profit-prediction.git### Installation

cd crop-profit-prediction

```1. **Clone the repository**

   ```bash

2. Install required packages:   git clone <your-repo-url>

```bash   cd Saciam

pip install -r requirements.txt   ```

```

2. **Install dependencies**

3. Train the models (first time only):   ```bash

```bash   pip install -r requirements.txt

python train.py   ```

```

3. **Run the module**

4. Run the Flask application:   ```bash

```bash   python Prediction/profit_app.py

python app.py   ```

```

## 📁 Project Structure

5. Open your browser and navigate to:

``````

http://127.0.0.1:5000Saciam/

```├── Data/

│   └── Crop_yield.csv                    # Historical yield data (19,173 records)

## Project Structure├── Prediction/

│   ├── profit_app.py                     # Main ML module

```│   └── crop_price_trends_2015_2025.csv   # Price data (1,320 records)

Saciam/├── requirements.txt                       # Python dependencies

├── app.py                              # Flask web server└── README.md                              # This file

├── ml_model.py                         # ML training functions```

├── train.py                            # Model training script

├── Crop_yield_enriched.csv            # Training dataset (19,171 rows, 21 features)## 💡 How to Use

├── crop_price_trends_2015_2025.csv    # Price data

├── templates/### As a Python Module

│   └── index.html                      # Web interface

├── *.joblib                            # Trained models (5 files)```python

├── form_options.json                   # Dropdown optionsfrom profit_app import load_and_train_models, predict_profit

└── requirements.txt                    # Python dependencies

```# Load and train models

models_data = load_and_train_models()

## Features Included

# Make prediction

### Input Parametersresult = predict_profit(

- Season (Kharif, Rabi, Summer, Whole Year)    season='Kharif',

- Crop Type (10 major crops)    crop_type='paddy',

- District (Karnataka districts)    district='Ballari',

- Year    year=2024,

- Area (in acres)    area=250.0,              # acres

- Previous Year Production (quintals)    production=5000.0,        # quintals

- Previous Year Area (acres)    cost_per_area=8000,      # ₹ per acre

- Previous Year Total Cost (₹)    models_data=models_data

)

### ML Model Features (19 total)

- Basic: season, year, crop, district, area# View results

- Agricultural: soil properties (pH, N, P, K, organic matter)print(f"Predicted Yield: {result['predicted_yield_per_acre']:.2f} quintals/acre")

- Irrigation & Weather: irrigation type, rainfall, temperatureprint(f"Predicted Price: ₹{result['predicted_price_per_quintal']:.0f}/quintal")

- Management: fertilizers, pesticides, seed quality, mechanization, farmer experienceprint(f"Net Profit: ₹{result['profit']:,.0f}")

print(f"ROI: {result['roi']:.1f}%")

### Output Predictions```

- Predicted Yield (quintals/acre)

- Predicted Price (₹/quintal)## 🌾 Supported Crops

- Total Revenue (₹)

- Total Cost (₹)- Paddy (Rice)

- Profit/Loss (₹)- Wheat

- Return on Investment (ROI %)- Corn (Maize)

- Profit per Acre (₹)- Cotton

- Groundnut

## Model Performance- Millet (Bajra)

- Jowar

- **Yield Model**: 53.05% R² Score- Ragi

- **Price Model**: 97.03% R² Score- Soybean

- Sugarcane

## Usage

## 💰 Calculation Formula

1. Select your crop season and type

2. Choose your district```python

3. Enter the year and area to be cultivatedPredicted Yield (quintals/acre) = ML Model(Season, Crop, District, Year, Area, Production)

4. Provide previous year's production, area, and total costPredicted Price (₹/quintal) = ML Model(Crop, Year) × Year Adjustment Factor

5. Click "Predict Profit" to get instant predictionsTotal Revenue = Yield × Price × Area

Total Cost = Cost per Acre × Area

## ContributingNet Profit = Total Revenue - Total Cost

ROI = (Profit / Total Cost) × 100

Contributions are welcome! Please feel free to submit a Pull Request.```



## License## 🎯 Year Impact



This project is open source and available under the MIT License.The application includes inflation and market growth factors:

- Base Year: 2020

## Author- Annual Price Increase: 6%

- Example: Year 2024 prices are 24% higher than 2020

Created for helping farmers make data-driven decisions about crop selection and resource allocation.

## 🔧 Technology Stack

- **ML Models**: scikit-learn (Random Forest Regressor)
- **Data Processing**: pandas, numpy
- **Language**: Python 3.8+

## � Data Sources

- Historical crop yield data (19,173+ records)
- Price trends data (1,320+ records, 2015-2025)

## 📝 License

This project is open source and available for educational purposes.

## �‍💻 Author

Created for helping farmers make data-driven decisions about crop profitability.

---

**Made with ❤️ for Indian Farmers** 🌾

