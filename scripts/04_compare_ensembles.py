# scripts/compare_ensembles.py

import os
import json
import argparse
import pandas as pd
from tabulate import tabulate
from core.config_loader import ConfigLoader


def load_ensemble_summary(model_name: str, config: ConfigLoader) -> dict:
    """
    Tenta ler o summary.json do ensemble do modelo.
    Se não existir, calcula a média dos arquivos fold_summary.json.
    """
    summary_path = os.path.join(config.get_output_dir(), model_name, "ensemble", "metrics", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)

    print(f"⚠️ summary.json não encontrado para '{model_name}'. Usando média dos folds...")

    # Caminho base dos folds
    folds = []
    n_folds = config.get_dataset_config().get("n_folds", 2)
    for i in range(n_folds):
        fold_path = os.path.join(config.get_output_dir(), model_name, f"fold_{i}", "fold_summary.json")
        if os.path.exists(fold_path):
            with open(fold_path, "r") as f:
                folds.append(json.load(f))
        else:
            print(f"⚠️ fold_summary.json não encontrado: {fold_path}")

    if not folds:
        raise FileNotFoundError(f"❌ Nenhum arquivo summary.json ou fold_summary.json encontrado para '{model_name}'")

    # Calcula média das métricas
    f1 = sum(f.get("val_f1", 0) for f in folds) / len(folds)
    acc = sum(f.get("val_acc", 0) for f in folds) / len(folds)
    auc = sum(f.get("val_auc", 0) for f in folds) / len(folds)

    return {"val_f1": f1, "val_acc": acc, "val_auc": auc}


def compare_ensembles(config: ConfigLoader, model_names: list):
    """
    Compara os resultados de ensemble dos modelos fornecidos.
    """
    rows = []
    for model in model_names:
        summary = load_ensemble_summary(model, config)
        rows.append({
            "model": model,
            "f1": summary.get("val_f1", "N/A"),
            "acc": summary.get("val_acc", "N/A"),
            "auc": summary.get("val_auc", "N/A")
        })

    df = pd.DataFrame(rows)
    print("\n📊 Comparação de Ensembles:")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    # Salvar CSV
    output_dir = os.path.join(config.get_output_dir(), "ensemble_comparison")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
    print(f"\n✅ Resultados salvos em: {output_dir}/comparison_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo .yaml")
    parser.add_argument("--models", nargs="+", required=True, help="Lista de nomes dos modelos para comparar")
    args = parser.parse_args()

    config = ConfigLoader(args.config)
    compare_ensembles(config, args.models)
