"""
Smart City Disaster - Data Preprocessing Module
================================================
Provides:
  * SmartCityPreprocessor : sklearn-compatible transformer
        - handles missing values
        - scales numerical features (StandardScaler)
        - encodes target labels
        - dumps/loads to pickle for reuse
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = [
    'temperature', 'humidity', 'rainfall_mm', 'wind_speed_kmh',
    'air_quality_index', 'seismic_activity', 'water_level_m',
    'building_density', 'road_quality_score', 'power_grid_load_pct',
    'drainage_capacity_pct', 'telecom_strength', 'green_cover_pct',
    'population_density', 'age_vulnerability_index', 'hospital_capacity',
    'evacuation_centers', 'emergency_vehicles',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'past_disaster_count', 'avg_response_time_min', 'recovery_score',
]
TARGET_COL = 'disaster_risk'


class SmartCityPreprocessor:
    """End-to-end preprocessing pipeline for the Smart City disaster dataset."""

    def __init__(self, numeric_features=None, target_col=TARGET_COL):
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.target_col = target_col
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame):
        X = df[self.numeric_features].copy()
        y = df[self.target_col].astype(str)

        self.imputer.fit(X)
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.numeric_features)
        self.scaler.fit(X_imp)
        self.label_encoder.fit(y)
        self._fitted = True
        return self

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame):
        assert self._fitted, "Call fit() before transform()."
        X = df[self.numeric_features].copy()
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.numeric_features)
        X_scaled = pd.DataFrame(self.scaler.transform(X_imp), columns=self.numeric_features)
        y = None
        if self.target_col in df.columns:
            y = self.label_encoder.transform(df[self.target_col].astype(str))
        return X_scaled, y

    # ------------------------------------------------------- fit_transform API
    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    # -------------------------------------------------------- persist / load
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'numeric_features': self.numeric_features,
                'target_col': self.target_col,
                'imputer': self.imputer,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
            }, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        obj = cls(numeric_features=bundle['numeric_features'],
                  target_col=bundle['target_col'])
        obj.imputer = bundle['imputer']
        obj.scaler = bundle['scaler']
        obj.label_encoder = bundle['label_encoder']
        obj._fitted = True
        return obj

    @property
    def classes_(self):
        return list(self.label_encoder.classes_)


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Train / val / test split (stratified)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[TARGET_COL]
    )
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=random_state,
        stratify=train_df[TARGET_COL]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/raw/smart_city_disaster.csv")
    print("Raw shape:", df.shape)
    pp = SmartCityPreprocessor().fit(df)
    X, y = pp.transform(df)
    print("Processed X:", X.shape, " y:", y.shape)
    print("Classes:", pp.classes_)
    pp.save("data/processed/preprocessor.pkl")
    print("[OK] Preprocessor saved.")
