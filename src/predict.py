
import joblib
import pandas as pd

def predict(sample_dict):

    model = joblib.load("models/best_diabetes_model.pkl")

    sample_df = pd.DataFrame([sample_dict])

    prediction = model.predict(sample_df)

    return int(prediction[0])


if __name__ == "__main__":

    example = {
        "gender": "Female",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 28.5,
        "HbA1c_level": 6.8,
        "blood_glucose_level": 150
    }

    result = predict(example)
    print("Prediction:", result)
