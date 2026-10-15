"""
Smart City Disaster Preparedness & Response - Synthetic Dataset Generator
Generates a realistic dataset for disaster risk prediction in urban environments
and saves it to data/raw/smart_city_disaster.csv
"""
import numpy as np
import pandas as pd
import os
import sys

np.random.seed(42)

def generate_smart_city_disaster_data(n_samples=15000,
                                      output_path="data/raw/smart_city_disaster.csv"):
    """
    Generate synthetic smart city disaster dataset.

    Feature groups:
      Environmental  : temperature, humidity, rainfall, wind_speed, air_quality, seismic_activity, water_level
      Infrastructure : building_density, road_quality, power_grid_load, drainage_capacity, telecom_strength, green_cover
      Demographic    : population_density, age_vulnerability, hospital_capacity, evacuation_centers, emergency_vehicles
      Temporal       : hour, day_of_week, month, is_weekend
      Historical     : past_disaster_count, avg_response_time, recovery_score

    Target:
      disaster_risk  in {Low, Medium, High, Critical}
    """
    data = {}

    # ===== Environmental =====
    data['temperature']         = np.random.normal(28, 7, n_samples).clip(-5, 50)
    data['humidity']            = np.random.normal(65, 20, n_samples).clip(5, 100)
    data['rainfall_mm']         = np.random.exponential(15, n_samples).clip(0, 300)
    data['wind_speed_kmh']      = np.random.exponential(18, n_samples).clip(0, 180)
    data['air_quality_index']   = np.random.normal(120, 50, n_samples).clip(0, 500)
    data['seismic_activity']    = np.random.exponential(0.5, n_samples).clip(0, 8)
    data['water_level_m']       = np.random.normal(2.5, 1.2, n_samples).clip(0, 8)

    # ===== Infrastructure =====
    data['building_density']     = np.random.uniform(0, 1, n_samples)
    data['road_quality_score']   = np.random.normal(0.7, 0.15, n_samples).clip(0, 1)
    data['power_grid_load_pct']  = np.random.normal(60, 20, n_samples).clip(0, 100)
    data['drainage_capacity_pct']= np.random.normal(0.65, 0.2, n_samples).clip(0, 1)
    data['telecom_strength']     = np.random.normal(0.8, 0.15, n_samples).clip(0, 1)
    data['green_cover_pct']      = np.random.normal(25, 10, n_samples).clip(0, 80)

    # ===== Demographic =====
    data['population_density']     = np.random.exponential(5000, n_samples).clip(50, 50000)
    data['age_vulnerability_index']= np.random.beta(2, 5, n_samples)
    data['hospital_capacity']      = np.random.uniform(0, 1, n_samples)
    data['evacuation_centers']     = np.random.poisson(8, n_samples).clip(0, 30)
    data['emergency_vehicles']     = np.random.poisson(25, n_samples).clip(0, 100)

    # ===== Temporal =====
    data['hour']        = np.random.randint(0, 24, n_samples)
    data['day_of_week'] = np.random.randint(0, 7, n_samples)
    data['month']       = np.random.randint(1, 13, n_samples)
    data['is_weekend']  = (data['day_of_week'] >= 5).astype(int)

    # ===== Historical =====
    data['past_disaster_count']   = np.random.poisson(2, n_samples).clip(0, 20)
    data['avg_response_time_min'] = np.random.normal(12, 5, n_samples).clip(1, 60)
    data['recovery_score']        = np.random.beta(5, 2, n_samples)

    df = pd.DataFrame(data)

    # ===== Non-linear risk score =====
    risk_score = (
        0.05 * (df['temperature'] - 25).abs()
        + 0.04 * df['humidity'] / 100
        + 0.06 * df['rainfall_mm'] / 100
        + 0.03 * df['wind_speed_kmh'] / 50
        + 0.02 * df['air_quality_index'] / 200
        + 0.5  * df['seismic_activity']
        + 0.4  * df['water_level_m'] / 5
        + 0.5  * df['building_density']
        - 0.4  * df['road_quality_score']
        + 0.03 * df['power_grid_load_pct'] / 100
        + 0.4  * (1 - df['drainage_capacity_pct'])
        - 0.2  * df['telecom_strength']
        + 0.01 * df['population_density'] / 10000
        + 0.6  * df['age_vulnerability_index']
        - 0.5  * df['hospital_capacity']
        - 0.05 * df['evacuation_centers']
        - 0.02 * df['emergency_vehicles']
        + 0.2  * df['past_disaster_count']
        - 0.05 * df['recovery_score']
        + 0.4  * df['is_weekend']
    )
    risk_score += np.random.normal(0, 0.5, n_samples)

    # Bucketize into 4 risk classes
    q = np.percentile(risk_score, [25, 50, 75])
    conditions = [
        risk_score <= q[0],
        (risk_score > q[0]) & (risk_score <= q[1]),
        (risk_score > q[1]) & (risk_score <= q[2]),
        risk_score > q[2],
    ]
    df['disaster_risk'] = np.select(conditions, ['Low', 'Medium', 'High', 'Critical'])

    # Inject some missing values (3%) for realism
    for col in ['humidity', 'air_quality_index', 'drainage_capacity_pct']:
        mask = np.random.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset generated -> {output_path}  shape={df.shape}")
    print("Class distribution:")
    print(df['disaster_risk'].value_counts())
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/smart_city_disaster.csv"
    generate_smart_city_disaster_data(output_path=out)
