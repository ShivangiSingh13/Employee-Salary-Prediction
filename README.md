# 💼 SmartSalary+ — Employee Salary Prediction App

An interactive **Streamlit** web application that predicts whether an employee earns **more than $50K** or **≤ $50K** annually, based on demographic and professional features — with every prediction explained using **SHAP (SHapley Additive exPlanations)**.

![Tech](https://img.shields.io/badge/Tech-Python%20%7C%20Streamlit-blue)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20SHAP-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Future Scope](#-future-scope)
- [License](#-license)

---

## 🚀 Overview

SmartSalary+ isn't just a binary salary classifier — it's built to be **explainable**. Rather than returning a prediction as a black box, it uses SHAP values to show exactly which features (age, education, occupation, hours worked, etc.) pushed the prediction toward "> $50K" or "≤ $50K," making the model's reasoning transparent and auditable. It also supports both single predictions through the UI and batch predictions via CSV upload.

---

## ✨ Features

- 🔮 **Predict salary class** (`>50K` or `≤50K`) using a trained machine learning model
- 🧠 **Explainable predictions** via SHAP, showing feature-level contribution to each result
- 📄 **Downloadable PDF report** summarizing the prediction and its explanation
- 📂 **Batch prediction** — upload a CSV of multiple employees and get predictions for all of them at once
- 📊 **Interactive data exploration** — graphs and visualizations of the underlying dataset

---

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Interactive web app UI |
| Scikit-learn | Model training and prediction |
| SHAP | Model explainability |
| Pandas | Data processing |
| Matplotlib / Seaborn (likely, for visualizations) | Dataset exploration graphs |
| FPDF or similar (for PDF export) | Downloadable prediction report |

---

## 📊 Dataset

- **Name**: Adult Census Income Dataset
- **Source**: UCI Machine Learning Repository
- **Key features**: Age, Education, Occupation, Hours-per-week, Native Country, and more
- **Derived feature**: `experience = age - 18`

---

## 📂 Project Structure

```
Employee-Salary-Prediction/
│
├── employee.py              # Main Streamlit application entry point
├── data/
│   └── adult 3.csv          # Adult Census Income dataset
├── utils/
│   ├── train_model.py       # Model training script
│   ├── preprocess.py        # Data cleaning and feature engineering
│   └── predict.py           # Prediction logic
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.x

### Installation

```bash
git clone https://github.com/ShivangiSingh13/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction
pip install streamlit scikit-learn shap pandas matplotlib fpdf
```

> Add a `requirements.txt` to the repo (if not already present) so installs are reproducible with a single `pip install -r requirements.txt`.

### Train the model (if not already trained/saved)

```bash
python utils/train_model.py
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run employee.py
```

Then in the browser window:
1. Enter employee details (age, education, occupation, hours-per-week, etc.) for a single prediction, **or** upload a CSV for batch predictions
2. View the predicted salary class
3. Review the SHAP explanation showing which features influenced the result
4. Download a PDF report of the prediction if needed
5. Explore the dataset visualizations to understand feature distributions and relationships

---

## ✅ Future Scope

- Add model comparison and hyperparameter tuning features
- Save user prediction history
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Add login/authentication for secured, multi-user access

---

## 📄 License

This project is developed for learning and portfolio demonstration purposes.
