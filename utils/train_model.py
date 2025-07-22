import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Globals to share
encoders = {}
categorical_columns = []
feature_order = []

def load_model_and_explainer():
    df = pd.read_csv("data/adult 3.csv").dropna()

    if 'experience' not in df.columns:
        df['experience'] = df['age'] - 18
        df['experience'] = df['experience'].clip(lower=0)

    if 'income' not in df.columns:
        raise ValueError("Missing 'income' column in dataset.")

    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    X = df.drop(columns=['income'])

    global categorical_columns, feature_order
    categorical_columns = X.select_dtypes(include='object').columns.tolist()
    feature_order = X.columns.tolist()

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model.predict, X)
    return model, explainer

