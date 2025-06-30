# scripts/plot_metrics.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(csv_path, output_dir, save_png=True):
    # Verifica se o CSV existe
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Lê o CSV
    df = pd.read_csv(csv_path)

    # Verifica colunas obrigatórias
    required_cols = {"model", "fold", "val_macro_f1", "val_accuracy", "val_balanced_accuracy"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"O CSV precisa conter as colunas: {required_cols}")
    
    # Derrete o DataFrame para formato longo (long-form)
    df_long = df.melt(id_vars=["model", "fold"], 
                    value_vars=["val_macro_f1", "val_accuracy", "val_balanced_accuracy"],
                    var_name="métrica", value_name="valor")

    # Renomear para nomes mais legíveis no gráfico
    df_long["métrica"] = df_long["métrica"].replace({
        "val_macro_f1": "F1 Macro",
        "val_accuracy": "Acurácia",
        "val_balanced_accuracy": "Acurácia Balanceada"
    })

    # Plota com Seaborn
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_long, x="métrica", y="valor", hue="model")
    ax.set_title("Comparação de Métricas por Modelo")
    ax.set_ylabel("Valor")
    ax.set_xlabel("Métrica")
    plt.legend(title="Modelo", loc="best")
    
    # Salva ou mostra
    output_path = os.path.join(output_dir, "comparacao_metricas.png")
    if save_png:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ Gráfico salvo em: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera gráficos de comparação de métricas por modelo.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Caminho para o arquivo CSV de métricas por fold (ex: outputs_sandbox/fold_metrics.csv).")
    parser.add_argument("--output-dir", type=str, default="outputs_sandbox/plots",
                        help="Diretório de saída para salvar os gráficos.")
    parser.add_argument("--no-save", action="store_true",
                        help="Se definido, exibe o gráfico em vez de salvar.")
    args = parser.parse_args()

    plot_metrics(args.csv, args.output_dir, save_png=not args.no_save)
