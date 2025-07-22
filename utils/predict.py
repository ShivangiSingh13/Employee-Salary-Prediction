import pandas as pd
from utils.preprocess import preprocess_input

def predict_single(input_df, model):
    processed_input = preprocess_input(input_df)
    prediction = model.predict(processed_input)[0]
    return prediction, processed_input

def predict_batch(batch_df, model):
    processed = preprocess_input(batch_df)
    predictions = model.predict(processed)
    batch_df['PredictedClass'] = [">50K" if p == 1 else "≤50K" for p in predictions]
    return batch_df