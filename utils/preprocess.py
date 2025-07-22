import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoders = {}
categorical_columns = []
feature_order = []

def load_encoders():
    global encoders, categorical_columns, feature_order

    df = pd.read_csv("data/adult 3.csv").dropna()

    if 'experience' not in df.columns:
        df['experience'] = df['age'] - 18
        df['experience'] = df['experience'].clip(lower=0)

    df = df.drop(columns=['income'], errors='ignore')

    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    feature_order = df.columns.tolist()

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le

def preprocess_input(df_input):
    global encoders, feature_order, categorical_columns
    df = df_input.copy()

    if 'experience' not in df.columns and 'age' in df.columns:
        df['experience'] = df['age'] - 18
        df['experience'] = df['experience'].clip(lower=0)

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    return df[feature_order]
load_encoders()