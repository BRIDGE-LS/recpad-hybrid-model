# run_meta_ensemble_weighted.py
import pandas as pd
import numpy as np
import os
import argparse
import json
from core.metrics import compute_metrics
from core.config_loader import ConfigLoader
from sklearn.metrics import roc_auc_score

def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    probs = df.filter(regex='^prob_').values
    labels = df['true_label'].values
    names = df['image_name'].values
    return names, labels, probs, df

def load_f1_scores(json_path, class_names):
    with open(json_path, 'r') as f:
        report = json.load(f)
    f1s = [report[str(cls)]['f1-score'] for cls in class_names]
    return np.array(f1s)

def infer_report_path(csv_path):
    model_dir = os.path.dirname(csv_path)
    return os.path.join(model_dir, 'metrics', 'classification_report.json')

def main(model_csvs, args):
    config = ConfigLoader(args.config)
    
    dataset_config = config.get_dataset_config()
    data_config = config.get_data_config()
    class_names = dataset_config["class_names"]
    n_classes = len(class_names)

    all_probs = []
    all_weights = []
    reference_names, reference_labels = None, None

    for model_name, csv_path in model_csvs.items():
        names, labels, probs, df = load_predictions(csv_path)

        # Validação de consistência
        if reference_names is None:
            reference_names = names
            reference_labels = labels
        else:
            assert all(reference_names == names), f"Inconsistência nos nomes das imagens para {model_name}"

        # Carrega pesos automaticamente
        json_path = infer_report_path(csv_path)
        f1_weights = load_f1_scores(json_path, class_names)
        all_probs.append(probs * f1_weights)
        all_weights.append(f1_weights)

    # Combinação ponderada
    total_weights = np.sum(all_weights, axis=0)
    final_probs = np.sum(all_probs, axis=0) / total_weights
    final_preds = np.argmax(final_probs, axis=1)

    # Monta resultado
    df_result = pd.DataFrame(final_probs, columns=[f'prob_{cls}' for cls in class_names])
    df_result['image_name'] = reference_names
    df_result['true_label'] = reference_labels
    df_result['pred_label'] = final_preds

    save_dir = os.path.join(data_config["output_dir"], 'meta_ensemble_weighted')
    os.makedirs(save_dir, exist_ok=True)
    df_result.to_csv(os.path.join(save_dir, 'ensemble_predictions.csv'), index=False)

    # Avaliação
    y_true = df_result["true_label"].values
    y_pred = df_result["pred_label"].values
    y_probs = df_result.filter(regex='^prob_').values

    # Avaliação principal
    val_f1, val_acc = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
        save_dir=save_dir,
        epoch=None,
        save_summary=True
    )

    # AUC separado
    val_auc = roc_auc_score(
        y_true=pd.get_dummies(y_true).values,
        y_score=y_probs,
        average="macro",
        multi_class="ovr"
    )

    # Salva resumo em JSON
    summary_path = os.path.join(save_dir, "summary.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "val_f1": round(val_f1, 4),
            "val_acc": round(val_acc, 4),
            "val_auc": round(val_auc, 4)
        }, f, indent=4)

    print(f"Meta-ensemble salvo em: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument('--b0_csv', required=True)
    parser.add_argument('--b3_csv', required=True)
    parser.add_argument('--b4_csv', required=True)
    args = parser.parse_args()

    csvs = {
        'efficientnet_b0': args.b0_csv,
        'efficientnet_b3': args.b3_csv,
        'efficientnet_b4': args.b4_csv
    }

    main(csvs, args)
