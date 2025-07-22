💼 SmartSalary+: Employee Salary Prediction App
SmartSalary+ is an interactive Streamlit web application that predicts whether an employee earns more than $50K or less than or equal to $50K annually based on key demographic and professional features. It uses a machine learning model and explains predictions using SHAP values. The app also generates a downloadable PDF report and supports batch predictions via CSV.

🔍 Features:
Predict salary class (>50K or ≤50K) using a trained ML model.
Explain the result using SHAP (SHapley Additive exPlanations).
Generate a downloadable PDF report of the prediction.
Upload CSV files for batch prediction.

Explore the dataset with interactive graphs and visualizations.
📁 Project Structure:
project/
├── employee.py                  # Main Streamlit application
├── data/
│   └── adult 3.csv              # Cleaned dataset used for training
├── utils/
│   ├── train_model.py           # Contains model training and SHAP explainer logic
│   ├── preprocess.py            # Handles preprocessing and label encoding
│   └── predict.py               # Single and batch prediction functions
└── README.md                    # Project documentation

📊 Dataset:
Name: Adult Census Income Dataset
Source: UCI Machine Learning Repository
Key Features: Age, Education, Occupation, Hours-per-week, Native Country, etc.
Derived Feature: experience = age - 18

✅ Future Scope:
Add model comparison and tuning features.
Save user prediction history.
Deploy on Streamlit Cloud or HuggingFace.
Add login/authentication for secured use.
