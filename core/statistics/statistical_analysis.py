# core/statistics/statistical_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from typing import List, Dict

try:
    from orangecontrib.evaluate import graph_ranks
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False
    print("[!] orange3 e orangecontrib.evaluate não estão instalados. Gráficos de diferença crítica não serão gerados.")


# === Funções principais de análise ===
def load_fold_scores(csv_path: str) -> pd.DataFrame:
    """
    Lê o arquivo fold_metrics.csv exportado pelo export_tables.py
    Retorna um DataFrame com colunas: model, fold, f1, accuracy, auc
    """
    df = pd.read_csv(csv_path)
    return df


def compute_ranks(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """
    Calcula os ranks dos modelos por fold usando a métrica especificada.
    """
    pivot = df.pivot(index="fold", columns="model", values=metric)
    ranks = pivot.rank(axis=1, ascending=False)
    return ranks


def friedman_test(df: pd.DataFrame, metric: str = "f1") -> float:
    """
    Executa o teste de Friedman.
    Retorna o p-valor.
    """
    pivot = df.pivot(index="fold", columns="model", values=metric)
    _, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
    return p_value


def nemenyi_posthoc(ranks: pd.DataFrame, save_path: str):
    """
    Gera o diagrama de diferença crítica (CD) usando os ranks dos modelos.
    Requer orange3.
    """
    if not ORANGE_AVAILABLE:
        print("[!] Orange não está instalado. Ignorando gráfico CD.")
        return

    avg_ranks = ranks.mean().values
    names = ranks.columns.tolist()
    graph_ranks(avg_ranks, names, cd=None, width=6, textspace=1.5)
    plt.title("Diagrama de Diferença Crítica")
    plt.savefig(save_path)
    plt.close()


def wilcoxon_against_baseline(df: pd.DataFrame, baseline: str, metric: str = "f1") -> Dict[str, float]:
    """
    Executa o teste de Wilcoxon pareado entre cada modelo e o baseline.
    Retorna um dicionário: {modelo: p-valor}
    """
    pivot = df.pivot(index="fold", columns="model", values=metric)
    results = {}
    for model in pivot.columns:
        if model == baseline:
            continue
        stat, p = wilcoxon(pivot[baseline], pivot[model])
        results[model] = p
    return results


# === Função principal ===
def full_statistical_report(csv_path: str, baseline: str, metric: str, output_dir: str):
    df = load_fold_scores(csv_path)
    ranks = compute_ranks(df, metric=metric)

    # Friedman
    p_friedman = friedman_test(df, metric)
    print(f"\n📊 p-valor do teste de Friedman ({metric}): {p_friedman:.5f}")

    # CD
    nemenyi_posthoc(ranks, save_path=f"{output_dir}/nemenyi_cd_diagram.png")

    # Wilcoxon
    pvals = wilcoxon_against_baseline(df, baseline, metric)
    print(f"\n📌 Wilcoxon contra baseline '{baseline}' ({metric}):")
    for model, p in pvals.items():
        print(f"  - {model}: p = {p:.5f}")

    # Salvar ranks e p-valor
    ranks.to_csv(f"{output_dir}/fold_ranks_{metric}.csv")
    with open(f"{output_dir}/friedman_result_{metric}.txt", "w") as f:
        f.write(f"p-valor do teste de Friedman: {p_friedman:.5f}\n")
        f.write("\nWilcoxon pareado contra baseline:\n")
        for model, p in pvals.items():
            f.write(f"- {model}: p = {p:.5f}\n")
