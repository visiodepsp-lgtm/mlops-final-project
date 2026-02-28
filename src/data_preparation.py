import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_and_prepare_data(path):

    df = pd.read_csv(path)

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

    return X, y, preprocessor


