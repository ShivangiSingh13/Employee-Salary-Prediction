# app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from utils.train_model import load_model_and_explainer
from utils.predict import predict_single, predict_batch
from utils.preprocess import load_encoders, feature_order, categorical_columns


st.set_page_config(page_title="SmartSalary+", page_icon="💼", layout="wide")
st.title("💼 SmartSalary+: Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features. 📊🔍")

model, explainer = load_model_and_explainer()
load_encoders()

# ---------- GENERATE PDF ----------
def generate_pdf(input_df, prediction):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    textobject = c.beginText(40, 750)
    textobject.setFont("Helvetica", 12)
    textobject.textLine("Prediction Report")
    textobject.textLine("----------------------------")
    for col, val in input_df.iloc[0].items():
        textobject.textLine(f"{col}: {val}")
    textobject.textLine(f"Prediction: {prediction}")
    c.drawText(textobject)
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------- SIDEBAR FORM ----------
st.sidebar.header("📋 Input Employee Details")
sample_df = pd.read_csv("data/adult 3.csv").dropna()
sample_df['experience'] = sample_df['age'] - 18
sample_df['experience'] = sample_df['experience'].clip(lower=0)
sample_row = sample_df.iloc[0]

input_data = {}

for col in feature_order:
    if col in categorical_columns:
        from utils.preprocess import encoders
        options = list(encoders[col].classes_)
        default_val = sample_row[col] if sample_row[col] in options else options[0]
        input_data[col] = st.sidebar.selectbox(col, options, index=options.index(default_val))
    else:
        val = sample_row[col] if col in sample_row else 0
        if pd.api.types.is_integer_dtype(sample_df[col]):
            val = int(val)
        else:
            val = float(val)

        if col == "age":
            input_data[col] = st.sidebar.slider("Age", 18, 90, int(val))
        elif col == "hours-per-week":
            input_data[col] = st.sidebar.slider("Hours per week", 1, 100, int(val))
        elif col == "experience":
            input_data[col] = st.sidebar.slider("Years of Experience", 0, 72, int(val))
        else:
            input_data[col] = st.sidebar.number_input(col, value=val)

input_df = pd.DataFrame([input_data])
st.subheader("🔍 Preview Input")
st.dataframe(input_df)

# ---------- PREDICTION ----------
if st.button("📈 Predict Salary Class"):
    try:
        prediction, processed_input = predict_single(input_df, model)
        pred_label = ">50K" if prediction == 1 else "≤50K"
        st.success(f"✅ Prediction: {pred_label}")

        st.subheader("🔍 Why this prediction?")
        shap_values = explainer(processed_input)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

        pdf_bytes = generate_pdf(input_df, pred_label)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📅 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")

# ---------- BATCH PREDICTION ----------
st.markdown("---")
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)
        batch_output = predict_batch(batch_data, model)
        st.dataframe(batch_output.head())

        csv = batch_output.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "predicted_batch.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction error: {e}")

# ---------- DATA EXPLORATION ----------
st.markdown("---")
st.header("📊 Data Exploration")

if os.path.exists("data/adult 3.csv"):
    df = pd.read_csv("data/adult 3.csv")
    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Descriptive Stats")
    st.dataframe(df.describe(include='all'))

    st.subheader("Income Distribution")
    fig_income = px.histogram(df, x='income', color='income', title='Income Class Distribution')
    st.plotly_chart(fig_income)

    if 'education' in df.columns and 'income' in df.columns:
        edu_income = df.groupby(['education', 'income']).size().unstack(fill_value=0)
        edu_pct = edu_income.apply(lambda x: x / x.sum() * 100, axis=1)
        fig_edu = px.bar(edu_pct, x=edu_pct.index, y=['<=50K', '>50K'], barmode='stack')
        st.subheader("Education vs Income")
        st.plotly_chart(fig_edu)

    if 'occupation' in df.columns and 'income' in df.columns:
        occ_income = df.groupby(['occupation', 'income']).size().unstack(fill_value=0)
        occ_pct = occ_income.apply(lambda x: x / x.sum() * 100, axis=1)
        fig_occ = px.bar(occ_pct, x=occ_pct.index, y=['<=50K', '>50K'], barmode='stack')
        st.subheader("Occupation vs Income")
        st.plotly_chart(fig_occ)
else:
    st.error("Dataset 'adult 3.csv' not found.")
