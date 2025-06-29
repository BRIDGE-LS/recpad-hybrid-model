# scripts/evaluate_ensemble.py

import argparse
import pandas as pd
import os

from core.config_loader import ConfigLoader
from core.ensemble import EnsemblePredictor

def main(config_path, csv_path):
    # === Carregar configuração e dataset ===
    config = ConfigLoader(config_path)
    df = pd.read_csv(csv_path)

    # === Definir modelos e folds ===
    model_names = [m["name"] for m in config.config["models"]]
    n_folds = config.config["dataset"]["folds"]

    # === Executar ensemble ===
    predictor = EnsemblePredictor(config, model_names=model_names, fold_count=n_folds)
    results = predictor.evaluate(df)

    print("✅ Resultado final do ensemble:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Caminho para o config.yaml")
    parser.add_argument("--csv", type=str, required=True, help="Caminho para o CSV com os dados para inferência")
    args = parser.parse_args()

    main(args.config, args.csv)
