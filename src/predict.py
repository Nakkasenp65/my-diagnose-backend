import pandas as pd
import joblib
from pathlib import Path

# PATH 
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "naive_bayes_model.pkl"
DATA_PATH = BASE_DIR / "data" / "transformed_dataset_cleaned.csv"

# load model
model = joblib.load("models/naive_bayes_model.pkl")

df = pd.read_csv(DATA_PATH)
symptom_columns = df.columns.drop('Disease')

def predict_disease(symptom_list):
    unknown_list = []
    input_dict = {symptom: False for symptom in symptom_columns}

    for symptom in symptom_list:
        if symptom in input_dict:
            input_dict[symptom] = True
        else:
            unknown_list.append(symptom)
            print(f"⚠️ Warning: '{symptom}' is not a known symptom column.")

    
    input_df = pd.DataFrame([input_dict])

    # prediction result
    prediction = model.predict(input_df)[0]
    
    # % of prediction
    probability = model.predict_proba(input_df).max()

    return prediction, probability, unknown_list

# TEST IN MAIN
# if __name__ == "__main__":
#     symptoms = ['shivering', 'chills', 'watering_from_eyes','continuous_sneezingssss']  # You can change these
#     disease, prob, ret_unknown_list = predict_disease(symptoms)
#     print(f"🩺 Predicted Disease: {disease}")
#     print(f"📊 Confidence: {prob:.2%}")
#     print(f" Unknown Symptom: {ret_unknown_list}")
