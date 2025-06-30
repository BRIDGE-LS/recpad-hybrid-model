# scripts/plot_cd_autorank.py

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib
from autorank import autorank, create_report, plot_stats

def autorank_by_fold(df, metric, output_path):
    # === Validação
    expected_cols = {"fold", "model", metric}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"⚠️ O CSV está faltando as colunas: {', '.join(missing)}")

    if df[[metric]].isnull().values.any():
        raise ValueError(f"⚠️ A métrica '{metric}' contém valores nulos.")

    # === Pivot
    df_pivot = df.pivot(index="fold", columns="model", values=metric)

    print(f"\n📊 Autorank por FOLD para a métrica '{metric}'...")
    result = autorank(df_pivot, alpha=0.05, verbose=True, order='descending')

    # === Relatório
    report_stream = io.StringIO()
    with contextlib.redirect_stdout(report_stream):
        create_report(result, decimal_places=3)
    report_text = report_stream.getvalue()

    txt_path = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"📝 Relatório por fold salvo em: {txt_path}")
    print(report_text)

    # === Gráfico
    plt.figure(figsize=(10, 5))
    plot_stats(result)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Gráfico autorank por fold salvo em: {output_path}")


def autorank_by_instance(prediction_dirs, output_path):
    acc_cols = {}
    for model_dir in prediction_dirs:
        model_name = os.path.basename(model_dir)
        csv_path = os.path.join(model_dir, "ensemble", "ensemble_predictions.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Arquivo não encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "true_label" not in df.columns or not any(c.startswith("pred") for c in df.columns):
            print(f"⚠️ Estrutura inesperada em: {csv_path}")
            continue

        # Identifica automaticamente a coluna de predição (ex: pred_b0)
        pred_col = [c for c in df.columns if c.startswith("pred")][0]
        acc = (df["true_label"] == df[pred_col]).astype(int)
        acc_cols[model_name] = acc.reset_index(drop=True)

    if not acc_cols:
        print("⚠️ Nenhum dado de predições carregado para autorank por instância.")
        return

    df_acc = pd.DataFrame(acc_cols)
    print(f"\n📊 Autorank por INSTÂNCIA com base em acurácia...")
    result = autorank(df_acc, alpha=0.05, verbose=True, order='descending')

    # === Relatório
    report_stream = io.StringIO()
    with contextlib.redirect_stdout(report_stream):
        create_report(result, decimal_places=3)
    report_text = report_stream.getvalue()

    txt_path = os.path.splitext(output_path)[0] + "_instances.txt"
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"📝 Relatório por instância salvo em: {txt_path}")
    print(report_text)

    # === Gráfico
    plt.figure(figsize=(10, 5))
    plot_stats(result)
    plt.tight_layout()
    img_path = os.path.splitext(output_path)[0] + "_instances.png"
    plt.savefig(img_path, dpi=300)
    print(f"✅ Gráfico autorank por instância salvo em: {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera gráficos de comparação estatística com autorank.")
    parser.add_argument("--csv", type=str, default="outputs/tabelas/fold_metrics.csv", help="CSV com métricas por fold")
    parser.add_argument("--metric", type=str, default="val_macro_f1", help="Métrica usada nos folds")
    parser.add_argument("--output", type=str, default="outputs/stats/cd_diagram_autorank.png", help="Base para saída dos gráficos")
    parser.add_argument("--pred_dirs", nargs="+", default=[
        "outputs/efficientnet_b0",
        "outputs/efficientnet_b3",
        "outputs/efficientnet_b4"
    ], help="Caminhos para os diretórios contendo ensemble_predictions.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    autorank_by_fold(df, args.metric, args.output)
    autorank_by_instance(args.pred_dirs, args.output)
