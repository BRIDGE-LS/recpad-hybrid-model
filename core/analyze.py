import os
import json
import pandas as pd
from core.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_boxplot_f1_scores
)


def analyze_fold(model_dir: str, fold: int, class_names: list[str]):
    """
    Realiza a análise visual dos resultados de um fold específico,
    gerando apenas o histórico de treinamento.
    As demais imagens (matrizes, ROC) já são geradas via compute_metrics.
    """
    fold_path = os.path.join(model_dir, f"fold_{fold}")
    log_path = os.path.join(fold_path, "log.csv")

    if not os.path.exists(log_path):
        print(f"⚠️ Fold {fold}: log ausente.")
        return

    log_df = pd.read_csv(log_path)

    plot_training_history(
        log_df,
        save_path=os.path.join(fold_path, "metrics", "training_history.png")
    )

def analyze_all_folds(config):
    """
    Gera análise visual para todos os folds e todos os modelos definidos na configuração.
    Inclui boxplot comparativo final entre modelos.
    """
    output_dir = config.get_output_dir()
    model_names = [m["name"] for m in config.config["models"]]
    class_names = config.config["dataset"]["class_names"]
    folds = config.config["dataset"]["folds"]

    for model_name in model_names:
        model_dir = os.path.join(output_dir, model_name)
        for fold in range(folds):
            analyze_fold(model_dir, fold, class_names)

    # Comparativo final
    plot_boxplot_f1_scores(
        output_dir,
        model_names,
        folds,
        save_path=os.path.join(output_dir, "f1_scores_comparison.png")
    )

def aggregate_fold_summaries(output_dir: str, model_name: str, num_folds: int):
    """
    Agrega os arquivos 'fold_summary.json' de cada fold e salva versões .csv e .json
    para análise estatística e visual posterior.

    Args:
        output_dir (str): Caminho base para os resultados (ex: "outputs")
        model_name (str): Nome do modelo (ex: "efficientnet_b0")
        num_folds (int): Número de folds (ex: 5)
    """
    summaries = []

    for fold in range(num_folds):
        summary_path = os.path.join(output_dir, model_name, f"fold_{fold}", "fold_summary.json")
        if not os.path.exists(summary_path):
            print(f"⚠️ Fold {fold}: arquivo 'fold_summary.json' não encontrado.")
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)
            summary["fold"] = fold
            summaries.append(summary)

    if not summaries:
        print(f"❌ Nenhum resumo encontrado para {model_name}.")
        return

    df = pd.DataFrame(summaries)
    csv_path = os.path.join(output_dir, model_name, "summary_global.csv")
    json_path = os.path.join(output_dir, model_name, "summary_global.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=4)

    print(f"✅ Resumo global salvo: {csv_path}")
    print(f"✅ Resumo global salvo: {json_path}")
