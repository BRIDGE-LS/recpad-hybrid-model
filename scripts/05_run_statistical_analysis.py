# scripts/run_statistical_analysis.py

import argparse
import os
import pandas as pd
from core.statistics.statistical_analysis import full_statistical_report

def main():
    parser = argparse.ArgumentParser(
        description="Executa análise estatística entre modelos com comparação pareada por fold."
    )
    parser.add_argument("--csv", type=str, default="outputs/fold_metrics.csv",
                        help="Caminho para o CSV com métricas por fold.")
    parser.add_argument("--baseline", type=str, default="efficientnet_b0",
                        help="Nome do modelo baseline para comparações pareadas.")
    parser.add_argument("--metric", type=str, default="f1",
                        choices=["f1", "acc", "auc"],
                        help="Métrica principal usada na análise.")
    parser.add_argument("--output-dir", type=str, default="outputs/stats",
                        help="Diretório para salvar os resultados da análise.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"❌ CSV não encontrado: {args.csv}")
    if not args.csv.endswith(".csv"):
        raise ValueError("❌ O arquivo fornecido não é um .csv")

    try:
        df = pd.read_csv(args.csv)
        print(f"📁 Modelos encontrados no CSV: {list(df['model'].unique())}")
    except Exception as e:
        raise ValueError(f"Erro ao ler o CSV: {e}")

    print("📊 Iniciando análise estatística...")
    full_statistical_report(
        csv_path=args.csv,
        baseline=args.baseline,
        metric=args.metric,
        output_dir=args.output_dir
    )
    print(f"✅ Relatórios salvos em: {args.output_dir}")

if __name__ == "__main__":
    main()
