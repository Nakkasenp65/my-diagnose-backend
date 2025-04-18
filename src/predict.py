import pandas as pd
import joblib
from pathlib import Path

# PATH 
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "naive_bayes_model.pkl"
DATA_PATH = BASE_DIR / "data" / "transformed_dataset_cleaned.csv"

model = joblib.load("models/naive_bayes_model.pkl")

df = pd.read_csv(DATA_PATH)
symptom_columns = df.columns.drop('Disease')

def predict_disease(symptom_list):
    input_dict = {symptom: False for symptom in symptom_columns}

    for symptom in symptom_list:
        if symptom in input_dict:
            input_dict[symptom] = True
        else:
            print(f"⚠️ Warning: '{symptom}' is not a known symptom column.")

    
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    return prediction, probability

# TEST IN MAIN
if __name__ == "__main__":
    symptoms = ['shivering', 'chills', 'watering_from_eyes','continuous_sneezing']  # You can change these
    disease, prob = predict_disease(symptoms)
    print(f"🩺 Predicted Disease: {disease}")
    print(f"📊 Confidence: {prob:.2%}")
