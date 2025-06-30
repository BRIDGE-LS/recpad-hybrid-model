# scripts/run_ensemble.py

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
from torch.utils.data import DataLoader

from core.config_loader import ConfigLoader
from core.model import ModelFactory
from core.dataset import TumorDataset
from core.augmentations import PerClassAugmentation
from core.metrics import compute_metrics


def load_all_fold_models(config, model_name):
    output_dir = config.get_output_dir()
    model_cfg = config.get_model_config(model_name)
    n_folds = config.get_dataset_config()["n_folds"]
    num_classes = config.get_dataset_config()["num_classes"]
    models = []

    for fold in range(n_folds):
        ckpt_path = os.path.join(output_dir, model_name, f"fold_{fold}", "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Modelo não encontrado: {ckpt_path}")
        
        model_factory = ModelFactory(model_cfg, num_classes)
        model = model_factory.get_model().to("cuda")
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        models.append(model)
    
    return models


def run_ensemble(config, model_name):
    print(f"🚀 Rodando ensemble para modelo: {model_name}")

    # === Carregar dados e configs ===
    data_cfg = config.get_data_config()
    dataset_cfg = config.get_dataset_config()
    df = pd.read_csv(data_cfg["csv_path"])

    class_names = dataset_cfg["class_names"]
    label_col = dataset_cfg["target_col"]
    input_col = dataset_cfg["input_col"]
    image_dir = data_cfg["image_dir"]

    # === Augmentation por classe ===
    aug = PerClassAugmentation(config)

    # === Dataset de inferência ===
    model_cfg = config.get_model_config(model_name)
    input_size = model_cfg.get("input_size", 224)
    dataset = TumorDataset(
        df,
        image_dir,
        list(range(len(df))),
        aug,
        input_col=input_col,
        target_col=label_col,
        input_size=input_size,
        is_inference=False
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # === Carregar modelos de todos os folds ===
    models = load_all_fold_models(config, model_name)

    all_probs = []
    all_targets = []
    all_inputs = []
    all_filenames = df[input_col].tolist()

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            outputs = [torch.softmax(m(x), dim=1) for m in models]
            avg_output = torch.stack(outputs).mean(dim=0)
            all_probs.append(avg_output.cpu())
            all_targets.extend(y.numpy())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = np.argmax(all_probs, axis=1)

    # === Métricas ===
    val_f1, val_acc = compute_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        class_names=class_names,
        save_dir=os.path.join(config.get_output_dir(), model_name, "ensemble", "metrics"),
        epoch=None,
        save_summary=True
    )

    val_auc = roc_auc_score(
        y_true=pd.get_dummies(all_targets).values,
        y_score=all_probs,
        average="macro",
        multi_class="ovr"
    )

    # === Salvar resumo ===
    metrics_dir = os.path.join(config.get_output_dir(), model_name, "ensemble", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    summary = {
        "val_f1": round(val_f1, 4),
        "val_acc": round(val_acc, 4),
        "val_auc": round(val_auc, 4)
    }
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # === Salvar predições individuais ===
    pred_df = pd.DataFrame({
        "image_name": all_filenames,
        "true_label": all_targets,
        "pred_label": all_preds
    })

    # Adicionar as probabilidades de cada classe
    for i, cls in enumerate(class_names):
        pred_df[f"prob_{cls}"] = all_probs[:, i]

    pred_csv_path = os.path.join(config.get_output_dir(), model_name, "ensemble", "ensemble_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"📁 Predições salvas em: {pred_csv_path}")
    print("✅ Ensemble finalizado com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    args = parser.parse_args()

    config = ConfigLoader(args.config)
    run_ensemble(config, args.model)
