# scripts/generate_report.py

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_relatorio(predictions_csv, output_dir, plot_type="pie"):
    os.makedirs(output_dir, exist_ok=True)

    # === Carregar predições ===
    df = pd.read_csv(predictions_csv)

    if "pred_label" not in df.columns:
        raise ValueError("❌ CSV deve conter a coluna 'pred_label' com as predições do ensemble.")

    # === Contagem e porcentagem por classe ===
    counts = df["pred_label"].value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    # === Salvar tabela resumo ===
    summary = pd.DataFrame({
        "classe": counts.index,
        "quantidade": counts.values,
        "percentual": percentages.round(2).values
    })
    summary_csv_path = os.path.join(output_dir, "summary_distribution.csv")
    summary.to_csv(summary_csv_path, index=False)
    print(f"📄 Resumo salvo em: {summary_csv_path}")

    # === Gerar gráfico ===
    plt.figure(figsize=(8, 6))
    if plot_type == "pie":
        plt.pie(percentages, labels=counts.index, autopct="%.1f%%", startangle=140)
        plt.title("Distribuição de Tecidos (Ensemble)")
    elif plot_type == "bar":
        sns.barplot(x=counts.index, y=counts.values)
        plt.xlabel("Classe")
        plt.ylabel("Quantidade")
        plt.title("Distribuição de Tecidos (Ensemble)")
    else:
        raise ValueError("❌ plot_type deve ser 'pie' ou 'bar'.")

    fig_path = os.path.join(output_dir, f"plot_distribution_{plot_type}.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"📊 Gráfico salvo em: {fig_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera relatório visual da distribuição de classes previstas.")
    parser.add_argument("--input", type=str, required=True, help="Caminho para o CSV com as predições ('pred_label').")
    parser.add_argument("--output-dir", type=str, required=True, help="Diretório para salvar os arquivos gerados.")
    parser.add_argument("--plot-type", type=str, choices=["pie", "bar"], default="pie", help="Tipo de gráfico a gerar.")
    
    args = parser.parse_args()

    gerar_relatorio(
        predictions_csv=args.input,
        output_dir=args.output_dir,
        plot_type=args.plot_type
    )
