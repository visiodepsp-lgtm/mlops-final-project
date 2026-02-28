
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train():

    # Asegurar que exista carpeta models
    os.makedirs("models", exist_ok=True)

    # Cargar dataset
    df = pd.read_csv("/content/diabetes_prediction_dataset.csv")

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    categorical_cols = ["gender", "smoking_history"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model trained successfully!")
    print(f"Validation Accuracy: {acc:.4f}")

    # Guardar modelo
    joblib.dump(model, "models/best_diabetes_model.pkl")
    print("Model saved at models/best_diabetes_model.pkl")

if __name__ == "__main__":
    train()

