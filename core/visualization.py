import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, class_names, save_path):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OvR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(log_df, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["val_loss"], label="Val Loss")
    plt.plot(log_df["val_macro_f1"], label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_boxplot_f1_scores(output_dir, model_names, folds, save_path):
    data = []
    for model in model_names:
        for fold in range(folds):
            summary_path = os.path.join(output_dir, model, f"fold_{fold}", "fold_summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path, "r") as f:
                summary = json.load(f)
            data.append({
                "Model": model,
                "Fold": fold,
                "Macro F1": summary.get("best_val_f1", 0)
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Model", y="Macro F1", data=df)
    plt.title("F1 Score Comparison Across Models")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
