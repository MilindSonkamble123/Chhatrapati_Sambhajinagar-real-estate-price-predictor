from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "real_data1.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "real_estate_model.pkl"
TARGET = "Price_INR"
FEATURES = [
    "Locality",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Builtup_Area_sqft",
    "Floor",
    "Property_Age_Years",
    "Furnishing",
    "Sale_Status",
    "Amenities_Level",
    "Construction_Type",
    "Parking",
    "Gated_Community",
    "School_Distance_km",
    "Hospital_Distance_km",
    "Airport_Distance_km",
    "Railway_Distance_km",
    "Bus_Stand_Distance_km",
    "Main_Road_Distance_km",
    "Pollution_AQI_Approx",
    "Total_Rooms_Approx",
    "Area_per_BHK_sqft",
]


def train_and_save_model(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH):
    df = pd.read_csv(dataset_path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset")

    for col in ["Locality", "Property_Type", "Furnishing", "Sale_Status", "Amenities_Level", "Construction_Type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    categorical_features = [
        "Locality",
        "Property_Type",
        "Furnishing",
        "Sale_Status",
        "Amenities_Level",
        "Construction_Type",
    ]
    numeric_features = [feature for feature in FEATURES if feature not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        max_features="sqrt",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"Model trained and saved at: {model_path}")
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    return pipeline


if __name__ == "__main__":
    train_and_save_model()
