# scripts/export_fold_metrics.py

import os
import json
import argparse
import pandas as pd
from glob import glob

def extract_fold_metrics(base_dir: str, models: list):
    """
    Extrai métricas de F1, Acurácia e AUC dos arquivos fold_summary.json
    para cada modelo e fold.
    """
    records = []

    for model in models:
        model_dir = os.path.join(base_dir, model)
        for fold_path in sorted(glob(os.path.join(model_dir, "fold_*"))):
            if not os.path.isdir(fold_path):
                continue  # Ignora arquivos que começam com "fold_"

            fold_name = os.path.basename(fold_path)
            try:
                fold_index = int(fold_name.split("_")[-1])
            except ValueError:
                print(f"[!] Nome inesperado de diretório: {fold_name}")
                continue

            summary_path = os.path.join(fold_path, "fold_summary.json")
            if not os.path.exists(summary_path):
                print(f"[!] Arquivo não encontrado: {summary_path}")
                continue

            with open(summary_path, "r") as f:
                summary = json.load(f)

            records.append({
                "model": model,
                "fold": fold_index,
                "f1": summary.get("val_f1", None),
                "acc": summary.get("val_acc", None),
                "auc": summary.get("val_auc", None)
            })

    return pd.DataFrame(records)



def main():
    parser = argparse.ArgumentParser(description="Exporta métricas por fold para análise estatística.")
    parser.add_argument("--base-dir", type=str, default="outputs_sandbox",
                        help="Diretório base onde estão os resultados por modelo.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Lista de modelos para incluir na exportação.")
    parser.add_argument("--output", type=str, default="outputs_sandbox/fold_metrics.csv",
                        help="Caminho para salvar o CSV consolidado.")

    args = parser.parse_args()
    df = extract_fold_metrics(args.base_dir, args.models)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"✅ CSV exportado para: {args.output}")

if __name__ == "__main__":
    main()
