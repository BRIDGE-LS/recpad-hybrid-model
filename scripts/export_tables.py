# scripts/export_tables.py

import os
import argparse
import json
import pandas as pd
from pathlib import Path

from core.config_loader import ConfigLoader

def export_fold_results_to_csv(output_dir, model_names, num_folds, save_path):
    rows = []
    for model in model_names:
        for fold in range(num_folds):
            summary_path = os.path.join(output_dir, model, f"fold_{fold}", "metrics", "summary.json")
            if not os.path.exists(summary_path):
                print(f"⚠️ Métricas não encontradas: {summary_path}")
                continue
            with open(summary_path, "r") as f:
                summary = json.load(f)
                rows.append({
                    "model": model,
                    "fold": fold,
                    "val_f1": round(summary.get("val_f1", 0.0), 4),
                    "val_acc": round(summary.get("val_acc", 0.0), 4),
                    "val_auc": round(summary.get("val_auc", 0.0), 4)
                })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅ Tabela de métricas por fold salva em: {save_path}")

def export_ensemble_results(output_dir, save_path="outputs/ensemble_summary.csv"):
    ensemble_path = os.path.join(output_dir, "ensemble", "metrics", "summary.json")
    if not os.path.exists(ensemble_path):
        print(f"⚠️ Resultado do ensemble não encontrado em: {ensemble_path}")
        return

    with open(ensemble_path, "r") as f:
        summary = json.load(f)

    df = pd.DataFrame([{
        "val_macro_f1": round(summary.get("val_macro_f1", 0.0), 4),
        "val_accuracy": round(summary.get("val_accuracy", 0.0), 4),
        "val_balanced_accuracy": round(summary.get("val_balanced_accuracy", 0.0), 4),
        "val_kappa": round(summary.get("val_kappa", 0.0), 4),
        "epoch": summary.get("epoch", None)

    }])

    df.to_csv(save_path, index=False)
    print(f"✅ Resultado agregado do ensemble salvo em: {save_path}")

def main(config_path):
    cfg = ConfigLoader(config_path)
    output_dir = cfg.get_output_dir()
    model_names = [m["name"] for m in cfg.config["models"]]
    folds = cfg.config["dataset"]["folds"]

    Path("outputs/tabelas").mkdir(parents=True, exist_ok=True)

    export_fold_results_to_csv(
        output_dir,
        model_names,
        folds,
        save_path="outputs/tabelas/fold_metrics.csv"
    )

    export_ensemble_results(
        output_dir,
        save_path="outputs/tabelas/ensemble_summary.csv"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
