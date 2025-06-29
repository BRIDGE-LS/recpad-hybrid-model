import os
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

def compute_metrics(
    y_true,
    y_pred,
    class_names,
    save_dir,
    y_probs=None,
    epoch=None,
    save_summary=True
):
    os.makedirs(save_dir, exist_ok=True)

    # === Métricas básicas ===
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # === Classification Report ===
    #print("y_true classes:", sorted(set(y_true)))
    #print("y_pred classes:", sorted(set(y_pred)))
    #print("Esperado:", list(range(len(class_names))))
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(save_dir, "classification_report.csv"))
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # === Confusion Matrix Normalizada ===
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"))
    plt.close()

    # === ROC Curves ===
    if y_probs is not None:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            plt.figure(figsize=(10, 8))

            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (OvR)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, "roc_curves.png"))
            plt.close()
        except Exception as e:
            logging.warning(f"[ROC WARNING] Falha ao calcular ROC: {e}")

    # === Save Summary JSON ===
    if save_summary:
        summary = {
            "val_accuracy": round(acc, 4),
            "val_macro_f1": round(f1_macro, 4),
            "val_balanced_accuracy": round(bal_acc, 4),
            "val_kappa": round(kappa, 4)
        }
        if epoch is not None:
            summary["epoch"] = epoch

        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

    return f1_macro, acc

