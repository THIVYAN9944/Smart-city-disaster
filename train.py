"""
Smart City Disaster - End-to-End Training Pipeline
==================================================
1. Generate (or load) raw dataset
2. Apply preprocessing
3. Apply feature engineering
4. Train ML models (with hyperparameter search) and save best
5. Train DL models (MLP, CNN1D, LSTM, Transformer) and save best
6. Produce comparison report and all visualizations
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Make sure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_dataset import generate_smart_city_disaster_data
from data_preprocessing import SmartCityPreprocessor, split_data
from feature_engineering import add_engineered_features
from ml_models import train_and_evaluate_ml
from utils import (ensure_dir, save_json, evaluate_classifier,
                   plot_confusion_matrix, plot_roc_curves,
                   plot_training_curves, classification_report_str,
                   compute_roc_auc)
from dl_models import (build_mlp, build_cnn1d, build_lstm, build_transformer,
                       get_callbacks, reshape_for_seq)


def main(epochs=40, batch_size=128, quick=False):
    # 1. Dataset
    raw_path = 'data/raw/smart_city_disaster.csv'
    if not os.path.exists(raw_path):
        generate_smart_city_disaster_data(output_path=raw_path)
    df = pd.read_csv(raw_path)
    print(f"[DATA] loaded raw: {df.shape}")

    # 2. Feature engineering
    df = add_engineered_features(df)
    print(f"[DATA] engineered: {df.shape}")

    # 3. Preprocess (numeric features only for ML/DL)
    pp = SmartCityPreprocessor()
    X, y = pp.fit_transform(df)
    feature_names = pp.numeric_features
    print(f"[DATA] processed: X={X.shape} y={y.shape}  features={len(feature_names)}")
    pp.save('models/preprocessor.pkl')

    # 4. Train / val / test split
    train_df, val_df, test_df = split_data(df)
    X_train, y_train = pp.transform(train_df)
    X_val, y_val     = pp.transform(val_df)
    X_test, y_test   = pp.transform(test_df)
    class_names = pp.classes_

    # ===== 5. ML TRAINING =====
    print("\n========== MACHINE LEARNING MODELS ==========")
    ml_results, best_ml_name, best_ml_model = train_and_evaluate_ml(
        X_train.values, y_train, X_val.values, y_val, X_test.values, y_test,
        class_names=class_names, cv=3
    )
    print(ml_results[['model', 'accuracy', 'f1_macro', 'roc_auc_ovr_macro']])

    # ===== 6. DL TRAINING =====
    print("\n========== DEEP LEARNING MODELS ==========")
    Xtr_seq = reshape_for_seq(X_train.values)
    Xva_seq = reshape_for_seq(X_val.values)
    Xte_seq = reshape_for_seq(X_test.values)
    n_classes = len(class_names)
    input_dim = X_train.shape[1]

    if quick:
        epochs = 8

    dl_results = []
    dl_models = {
        'MLP':        build_mlp(input_dim, n_classes),
        'CNN1D':      build_cnn1d(input