# train.py
import joblib
import json
from ml_model import load_and_train_models # Import from your original file

print("Starting model training...")

# 1. Train models
models_data = load_and_train_models()

if models_data:
    # 2. Save models and encoders
    joblib.dump(models_data['yield_model'], 'yield_model.joblib')
    joblib.dump(models_data['price_model'], 'price_model.joblib')
    joblib.dump(models_data['le_season'], 'le_season.joblib')
    joblib.dump(models_data['le_crop'], 'le_crop.joblib')
    joblib.dump(models_data['le_dist'], 'le_dist.joblib')
    
    print("✓ Models and encoders saved to disk.")

    # 3. Save dropdown options
    options = {
        'seasons': models_data['le_season'].classes_.tolist(),
        'crops': models_data['le_crop'].classes_.tolist(),
        'districts': models_data['le_dist'].classes_.tolist()
    }
    
    with open('form_options.json', 'w') as f:
        json.dump(options, f)
        
    print(f"✓ Form options saved to 'form_options.json'.")
    print("Training complete. You can now run 'flask run' or 'python app.py'.")

else:
    print("❌ Model training failed. Check 'ml_model.py' and data paths.")

if __name__ == "__main__":
    pass # This file is meant to be run directly as a script