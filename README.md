# Employee Salary Prediction App

A machine learning-powered web application that predicts employee salaries based on years of experience using Flask and scikit-learn.

## 🚀 Features

- **Salary Prediction**: Predict employee salaries based on years of experience
- **Interactive Web Interface**: Clean and user-friendly web interface
- **Machine Learning Model**: Uses trained scikit-learn model for accurate predictions
- **Real-time Results**: Get instant salary predictions with formatted output

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn
- **Frontend**: HTML5, CSS3
- **Data Processing**: NumPy, Pandas
- **Model Persistence**: Pickle

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Afsiya21/employee_salary_app.git
   cd employee_salary_app
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### For Streamlit App (Recommended):
1. **Run the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **The app will automatically open** in your browser at:
   ```
   http://localhost:8501
   ```

3. **Fill in the sidebar form** with employee details and click "Predict Salary".

### For Flask App (Alternative):
1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
employee_salary_app/
│
├── streamlit_app.py            # Main Streamlit application (recommended)
├── app.py                      # Flask application (alternative)
├── requirements.txt            # Python dependencies
├── Procfile                    # For Heroku deployment
├── salary_predictor (3).pkl    # Trained ML model
├── scaler (3).pkl             # Data scaler for preprocessing
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
├── templates/
│   └── index.html             # HTML template for Flask app
│
└── static/
    └── styles.css             # CSS styles for Flask app
```

## 🤖 Model Information

The application uses a pre-trained machine learning model that:
- Predicts salary based on years of experience
- Uses feature scaling for better prediction accuracy
- Trained on employee salary dataset
- Provides predictions in USD format

## 🔧 Dependencies

- Flask
- scikit-learn
- NumPy
- Pandas (if used for data processing)
- Pickle (built-in Python module)

## 🚀 Streamlit Cloud Deployment

This app is deployed on Streamlit Cloud! You can access it at: [Employee Salary Predictor](https://your-app-url.streamlit.app)

### Deploy to Streamlit Cloud:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path to: `streamlit_app.py`
6. Click "Deploy!"

## 📊 API Endpoints (Flask Version)

- `GET /` - Home page with the prediction form
- `POST /predict` - Endpoint for salary prediction
- `GET /health` - Health check endpoint

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Afsiya21**
- GitHub: [@Afsiya21](https://github.com/Afsiya21)
- Email: afsiyanaaz212@gmail.com

## 🙏 Acknowledgments

- Thanks to the scikit-learn community for the machine learning tools
- Flask community for the excellent web framework
- All contributors and testers

## 📈 Future Enhancements

- [ ] Add more features for salary prediction (location, education, skills)
- [ ] Implement data visualization charts
- [ ] Add user authentication
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
- [ ] Add API documentation
- [ ] Implement model retraining functionality

---

⭐ If you found this project helpful, please give it a star!
