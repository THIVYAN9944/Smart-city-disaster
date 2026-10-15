"""
Smart City Disaster - Machine Learning Models
==============================================
Implements and tunes:
  - Logistic Regression (baseline)
  - Decision Tree
  - Random Forest
  - SVM (RBF)
  - Gradient Boosting (sklearn)
  - XGBoost (if available)
  - LightGBM (if available)

The `train_and_evaluate_ml` function trains all models, performs
cross-validated hyperparameter tuning with GridSearchCV, and
returns aggregated metrics.
"""
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# -------------------------------------------------------------- Model zoo
def get_model_definitions(num_classes: int):
    """Return (name, estimator, param_grid) tuples."""
    models = [
        ('LogisticRegression', LogisticRegression(max_iter=2000, multi_class='multinomial'),
         {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs']}),
        ('DecisionTree', DecisionTreeClassifier(random_state=42),
         {'max_depth': [8, 12, 20], 'min_samples_split': [2, 10]}),
        ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1),
         {'n_estimators': [200, 400], 'max_depth': [12, 20, None],
          'min_samples_split': [2, 5]}),
        ('SVM_RBF', SVC(probability=True, random_state=42),
         {'C': [0.5, 1.0, 5.0], 'gamma': ['scale', 0.1]}),
        ('GradientBoosting', GradientBoostingClassifier(random_state=42),
         {'n_estimators': [150, 300], 'learning_rate': [0.05, 0.1],
          'max_depth': [3, 5]}),
    ]
    if HAS_XGB:
        models.append((
            'XGBoost',
            XGBClassifier(objective='multi:softprob', num_class=num_classes,
                          use_label_encoder=False, eval_metric='mlogloss',
                          random_state=42, n_jobs=-1, tree_method='hist'),
            {'n_estimators': [200, 400], 'learning_rate': [0.05, 0.1],
             'max_depth': [4, 6]}
        ))
    if HAS_LGBM:
        models.append((
            'LightGBM',
            LGBMClassifier(random_state=42, n_jobs=-1),
            {'n_estimators': [200, 400], 'learning_rate': [0.05, 0.1],
             'num_leaves': [31, 63]}
        ))
    return models


def train_and_evaluate_ml(X_train, y_train, X_val, y_val, X_test, y_test,
                          class_names, cv=3, save_dir='models',
                          reports_dir='reports'):
    """Train all ML models with grid search, return metrics table + best model."""
    from utils import (evaluate_classifier, classification_report_str,
                       plot_confusion_matrix, plot_roc_curves,
                       plot_feature_importance, save_json, ensure_dir)

    ensure_dir(save_dir)
    ensure_dir(reports_dir)
    ensure_dir('visualizations/confusion_matrices')
    ensure_dir('visualizations/roc_curves')
    ensure_dir('visualizations/feature_importance')

    results = []
    fitted = {}
    best_name, best_acc, best_model = None, -1, None
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model, grid in get_model_definitions(num_classes=len(class_names)):
        print(f"\n=== Training {name} ===")
        gs = GridSearchCV(model, grid, cv=skf, scoring='accuracy',
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        # evaluate on test
        y_pred = gs.predict(X_test)
        y_proba = gs.predict_proba(X_test) if hasattr(gs, 'predict_proba') else None
        metrics = evaluate_classifier(y_test, y_pred, y_proba, class_names)
        metrics['model'] = name
        metrics['best_params'] = gs.best_params_
        results.append(metrics)
        fitted[name] = gs

        # save artifacts
        with open(os.path.join(save_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(gs, f)

        # report + plots
        report_path = os.path.join(reports_dir, f'{name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {name}\nBest params: {gs.best_params_}\n\n")
            f.write(classification_report_str(y_test, y_pred, class_names))
            f.write(f"\nAccuracy : {metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1 : {metrics['f1_macro']:.4f}\n")
            if 'roc_auc_ovr_macro' in metrics:
                f.write(f"ROC-AUC  : {metrics['roc_auc_ovr_macro']:.4f}\n")

        plot_confusion_matrix(y_test, y_pred, class_names,
                              f'visualizations/confusion_matrices/{name}.png',
                              title=f'{name} - Confusion Matrix')
        if y_proba is not None:
            plot_roc_curves(y_test, y_proba, class_names,
                            f'visualizations/roc_curves/{name}.png',
                            title=f'{name} - ROC Curves')

        # feature importance (tree-based / linear)
        try:
            est = gs.best_estimator_
            if hasattr(est, 'feature_importances_'):
                importances = est.feature_importances_
                if hasattr(X_train, 'columns'):
                    feats = list(X_train.columns)
                else:
                    feats = [f'f{i}' for i in range(X_train.shape[1])]
                plot_feature_importance(importances, feats,
                                        f'visualizations/feature_importance/{name}.png',
                                        title=f'{name} - Feature Importance')
            elif hasattr(est, 'coef_'):
                coefs = np.mean(np.abs(est.coef_), axis=0)
                if hasattr(X_train, 'columns'):
                    feats = list(X_train.columns)
                else:
                    feats = [f'f{i}' for i in range(X_train.shape[1])]
                plot_feature_importance(coefs, feats,
                                        f'visualizations/feature_importance/{name}.png',
                                        title=f'{name} - |Coefficient|')
        except Exception as e:
            print(f"[warn] feature importance for {name} failed: {e}")

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_name = name
            best_model = gs

    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    results_df.to_csv(os.path.join(reports_dir, 'ml_results.csv'), index=False)
    save_json(results, os.path.join(reports_dir, 'ml_results.json'))

    # persist best model as the canonical ML model
    if best_model is not None:
        with open(os.path.join(save_dir, 'best_ml_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
    print(f"\n[OK] Best ML model: {best_name}  acc={best_acc:.4f}")
    return results_df, best_name, best_model
