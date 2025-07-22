from flask import Flask, render_template, request
import numpy as np
import os

# Try different import methods for joblib
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler using relative paths
model_path = os.path.join(BASE_DIR, "salary_predictor (3).pkl")
scaler_path = os.path.join(BASE_DIR, "scaler (3).pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model, scaler = None, None

# Define all possible categories for one-hot encoding
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov']
education_options = ['Bachelors', 'HS-grad', 'Some-college']
marital_status_options = ['Married-civ-spouse', 'Never-married']
occupation_options = ['Exec-managerial', 'Craft-repair']
relationship_options = ['Husband', 'Not-in-family']
race_options = ['White', 'Black']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'India']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }
    return status

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None or scaler is None:
        return render_template("index.html", prediction="❌ Model not loaded. Please check server logs.")
    
    try:
        # Get numeric input
        age = int(request.form['age'])
        fnlwgt = int(request.form['fnlwgt'])
        education_num = int(request.form['education_num'])
        capital_gain = int(request.form['capital_gain'])
        capital_loss = int(request.form['capital_loss'])
        hours_per_week = int(request.form['hours_per_week'])

        # Get categorical input
        workclass = request.form['workclass']
        education = request.form['education']
        marital_status = request.form['marital_status']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']
        gender = request.form['gender']
        native_country = request.form['native_country']

        # Start feature vector
        features = [
            age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
        ]

        # One-hot encoding for each category
        def one_hot_encode(value, options):
            return [1 if value == option else 0 for option in options]

        features += one_hot_encode(workclass, workclass_options)
        features += one_hot_encode(education, education_options)
        features += one_hot_encode(marital_status, marital_status_options)
        features += one_hot_encode(occupation, occupation_options)
        features += one_hot_encode(relationship, relationship_options)
        features += one_hot_encode(race, race_options)
        features += one_hot_encode(gender, gender_options)
        features += one_hot_encode(native_country, native_country_options)

        # Pad remaining features (if total required = 100)
        while len(features) < 100:
            features.append(0)

        # Convert to NumPy and reshape
        X = np.array(features).reshape(1, -1)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]
        result = "✨ Salary >50K ✨" if prediction == 1 else "💼 Salary <=50K 💼"

        print("Predicted result:", result)
        return render_template("index.html", prediction=result)

    except ValueError as ve:
        print(f"Input validation error: {ve}")
        return render_template("index.html", prediction="❌ Please enter valid numeric values.")
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("index.html", prediction="❌ An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    # Get port from environment variable for deployment platforms like Heroku
    port = int(os.environ.get('PORT', 5000))
    # Set debug based on environment
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
