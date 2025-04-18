import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from pathlib import Path
import joblib
import os

#PATH
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "naive_bayes_model.pkl"

df = pd.read_csv(os.path.join('data/transformed_dataset_cleaned.csv'), low_memory=False)
# df.columns = df.columns.str.strip()
# df.to_csv("data/transformed_dataset_cleaned.csv", index=False)
# print("Cleaned dataset done successfully")

# df.dropna(subset=['Disease'], inplace=True)

X = df.drop('Disease', axis=1)
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")
