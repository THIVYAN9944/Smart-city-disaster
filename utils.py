"""
Smart City Disaster - Utilities
================================
Common helpers for evaluation, plotting, and persistence.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def evaluate_classifier(y_true, y_pred, y_proba=None, class_names=None, prefix=""):
    """Return a dict of evaluation metrics + classification report."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics = {
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
    }
    if y_proba is not None and class_names is not None:
        roc_auc = compute_roc_auc(y_true, y_proba, class_names)
        metrics['roc_auc_ovr_macro'] = float(roc_auc)
    return metrics


def classification_report_str(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = [str(c) for c in np.unique(y_true)]
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    ensure_dir(os.path.dirname(out_path))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(y_true, y_proba, class_names, out_path, title="ROC Curves"):
    ensure_dir(os.path.dirname(out_path))
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(importances, feature_names, out_path, top_n=20, title="Feature Importance"):
    ensure_dir(os.path.dirname(out_path))
    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=[feature_names[i] for i in idx], palette='viridis')
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_curves(history, out_path, title="Training Curves"):
    ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get('loss', []), label='Train Loss')
    axes[0].plot(history.history.get('val_loss', []), label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    axes[1].plot(history.history.get(acc_key, []), label='Train Acc')
    axes[1].plot(history.history.get(val_acc_key, []), label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_roc_auc(y_true, y_proba, class_names):
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    aucs = []
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs))


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
