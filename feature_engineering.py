"""
Smart City Disaster - Feature Engineering Module
================================================
Adds domain-driven interaction features, log transforms for skewed features,
and cyclical encodings for temporal features.
"""
import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-relevant engineered features.

    New features
    ------------
    log_rainfall          : log1p of rainfall (handles skew)
    log_pop_density       : log1p of population density
    wind_humidity_index   : composite storm index = wind * humidity
    infrastructure_index  : (road_quality + drainage + telecom)/3
    vulnerability_score   : age_vuln * pop_density_norm * (1 - hospital_capacity)
    green_vs_density      : green_cover / (1 + building_density)
    response_efficiency   : recovery_score / (1 + response_time)
    is_high_rain          : rainfall > 50mm
    is_high_wind          : wind > 60km/h
    is_night              : hour in [20, 23] U [0, 5]
    season                : meteorological season from month
    hour_sin / hour_cos   : cyclical hour encoding
    month_sin / month_cos : cyclical month encoding
    """
    df = df.copy()

    # ---- skewed transforms
    df['log_rainfall'] = np.log1p(df['rainfall_mm'])
    df['log_pop_density'] = np.log1p(df['population_density'])

    # ---- interaction / domain features
    df['wind_humidity_index'] = df['wind_speed_kmh'] * (df['humidity'].fillna(df['humidity'].median()) / 100.0)
    df['infrastructure_index'] = (
        df['road_quality_score']
        + df['drainage_capacity_pct'].fillna(df['drainage_capacity_pct'].median())
        + df['telecom_strength']
    ) / 3.0
    df['vulnerability_score'] = (
        df['age_vulnerability_index']
        * (df['population_density'] / df['population_density'].max())
        * (1 - df['hospital_capacity'])
    )
    df['green_vs_density'] = df['green_cover_pct'] / (1 + df['building_density'])
    df['response_efficiency'] = df['recovery_score'] / (1 + df['avg_response_time_min'])

    # ---- binary flags
    df['is_high_rain'] = (df['rainfall_mm'] > 50).astype(int)
    df['is_high_wind'] = (df['wind_speed_kmh'] > 60).astype(int)
    df['is_night']     = df['hour'].apply(lambda h: 1 if h >= 20 or h <= 5 else 0)

    # ---- season
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
    df['season'] = df['month'].map(season_map)

    # ---- cyclical encodings
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


def get_engineered_feature_columns():
    return [
        'log_rainfall', 'log_pop_density', 'wind_humidity_index',
        'infrastructure_index', 'vulnerability_score', 'green_vs_density',
        'response_efficiency', 'is_high_rain', 'is_high_wind', 'is_night',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        # season one-hot handled separately
    ]


if __name__ == "__main__":
    df = pd.read_csv("data/raw/smart_city_disaster.csv")
    df_eng = add_engineered_features(df)
    print("Engineered shape:", df_eng.shape)
    print("New columns:", [c for c in df_eng.columns if c not in df.columns])
    df_eng.to_csv("data/processed/smart_city_disaster_engineered.csv", index=False)
    print("[OK] Engineered dataset saved.")
