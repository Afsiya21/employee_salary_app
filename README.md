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

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Enter years of experience** in the input field and click "Predict Salary" to get the predicted salary.

## 📁 Project Structure

```
employee_salary_app/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── salary_predictor (3).pkl    # Trained ML model
├── scaler (3).pkl             # Data scaler for preprocessing
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
│
├── templates/
│   └── index.html             # HTML template for the web interface
│
└── static/
    └── styles.css             # CSS styles for the web interface
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

## 📊 API Endpoints

- `GET /` - Home page with the prediction form
- `POST /predict` - Endpoint for salary prediction

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
