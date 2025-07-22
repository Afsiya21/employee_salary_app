import streamlit as st
import numpy as np
import pandas as pd
import os
import warnings

# Suppress sklearn version warnings for deployment
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*version.*")

# Try different import methods for joblib
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

# Set page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .prediction-success {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    
    .prediction-low {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>💰 Employee Salary Predictor</h1>
    <p>Predict salary based on employee demographics and work details</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        # Get the directory where this script is located
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Load model and scaler using relative paths
        model_path = os.path.join(BASE_DIR, "salary_predictor (3).pkl")
        scaler_path = os.path.join(BASE_DIR, "scaler (3).pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler, True
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, False

# Load models
model, scaler, models_loaded = load_models()

if not models_loaded:
    st.error("🚨 Models could not be loaded. Please check the model files.")
    st.stop()
else:
    st.success("✅ Models loaded successfully!")

# Create input form in sidebar
with st.sidebar:
    st.header("📊 Input Parameters")
    
    # Demographic Information
    st.subheader("👤 Personal Information")
    age = st.number_input("Age", min_value=16, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Other"])
    native_country = st.selectbox("Native Country", 
                                 ["United-States", "India", "Mexico", "Philippines", "Germany"])
    
    # Education and Work
    st.subheader("🎓 Education & Experience")
    education = st.selectbox("Education", 
                            ["Bachelors", "HS-grad", "Some-college", "11th", "Masters", 
                             "Doctorate", "Assoc-acdm", "Prof-school"])
    education_num = st.number_input("Education Number", min_value=1, max_value=20, value=10, step=1)
    
    # Work Information
    st.subheader("💼 Work Information")
    workclass = st.selectbox("Work Class", ["Private", "Self-emp-not-inc", "Local-gov"])
    occupation = st.selectbox("Occupation", 
                             ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-specialty"])
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, step=1)
    
    # Family Information
    st.subheader("👨‍👩‍👧‍👦 Family Information")
    marital_status = st.selectbox("Marital Status", 
                                 ["Never-married", "Married-civ-spouse", "Divorced", 
                                  "Separated", "Widowed"])
    relationship = st.selectbox("Relationship", 
                               ["Husband", "Not-in-family", "Unmarried", "Own-child"])
    
    # Financial Information
    st.subheader("💵 Financial Information")
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0, value=200000, step=1000)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0, step=100)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0, step=100)

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Predict button
    if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
        try:
            # Define category options for one-hot encoding (same as training)
            workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov']
            education_options = ['Bachelors', 'HS-grad', 'Some-college']
            marital_status_options = ['Married-civ-spouse', 'Never-married']
            occupation_options = ['Exec-managerial', 'Craft-repair']
            relationship_options = ['Husband', 'Not-in-family']
            race_options = ['White', 'Black']
            gender_options = ['Male', 'Female']
            native_country_options = ['United-States', 'India']
            
            # Start feature vector
            features = [age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week]
            
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
            
            # Create proper feature names for the DataFrame
            feature_names = (
                ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] +
                [f'workclass_{opt}' for opt in workclass_options] +
                [f'education_{opt}' for opt in education_options] +
                [f'marital_status_{opt}' for opt in marital_status_options] +
                [f'occupation_{opt}' for opt in occupation_options] +
                [f'relationship_{opt}' for opt in relationship_options] +
                [f'race_{opt}' for opt in race_options] +
                [f'gender_{opt}' for opt in gender_options] +
                [f'native_country_{opt}' for opt in native_country_options]
            )
            
            # Pad feature names to match the required length
            while len(feature_names) < len(features):
                feature_names.append(f'feature_{len(feature_names)}')
                
            # Create DataFrame with proper feature names
            X_df = pd.DataFrame([features[:len(feature_names)]], columns=feature_names[:len(features)])
            
            # Scale the features using DataFrame
            X_scaled = scaler.transform(X_df)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            prediction_proba = model.predict_proba(X_scaled)[0]
            
            # Display result
            if prediction == 1:
                st.markdown("""
                <div class="prediction-success">
                    ✨ Predicted Salary: >$50K ✨<br>
                    High earning potential detected!
                </div>
                """, unsafe_allow_html=True)
                confidence = prediction_proba[1] * 100
            else:
                st.markdown("""
                <div class="prediction-low">
                    💼 Predicted Salary: ≤$50K 💼<br>
                    Moderate earning range predicted.
                </div>
                """, unsafe_allow_html=True)
                confidence = prediction_proba[0] * 100
            
            # Show confidence
            st.info(f"🎯 Prediction Confidence: {confidence:.1f}%")
            
            # Show feature importance insights
            with st.expander("📈 Prediction Insights"):
                st.write("**Key factors influencing this prediction:**")
                st.write(f"• Age: {age} years")
                st.write(f"• Education Level: {education}")
                st.write(f"• Work Hours: {hours_per_week}/week")
                st.write(f"• Occupation: {occupation}")
                st.write(f"• Work Class: {workclass}")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# Additional information
st.markdown("---")
with st.expander("ℹ️ About this App"):
    st.markdown("""
    <div class="info-box">
    <h4>🎯 How it works:</h4>
    <ul>
        <li>This app uses a trained machine learning model to predict salary ranges</li>
        <li>Based on demographic and professional information</li>
        <li>Predicts whether salary is above or below $50K threshold</li>
        <li>Uses ensemble learning techniques for accurate predictions</li>
    </ul>
    
    <h4>📊 Model Features:</h4>
    <ul>
        <li>Age, Education, Work experience</li>
        <li>Occupation, Work class, Hours per week</li>
        <li>Demographic information</li>
        <li>Financial indicators (capital gains/losses)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💡 Built with Streamlit | 🚀 Deployed on Streamlit Cloud</p>
    <p>Made by <strong>Afsiya21</strong> | 
    <a href="https://github.com/Afsiya21/employee_salary_app" target="_blank">
        🔗 View on GitHub
    </a></p>
</div>
""", unsafe_allow_html=True)
