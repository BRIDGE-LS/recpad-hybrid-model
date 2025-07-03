# training.py

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from core.config_loader import ConfigLoader
from core.model import ModelFactory
from core.dataset import TumorDataset
from core.augmentations import PerClassAugmentation
from core.metrics import compute_metrics
from core.analyze import analyze_fold
from core.utils.data_utils import generate_labels_from_directory
from core.visualization import plot_training_history

sys.path.append(str(Path(__file__).resolve().parent.parent))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight)
        return loss

def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def train_one_fold(config_path, model_name, fold):
    # === Load config ===
    config = ConfigLoader(config_path)
    model_cfg = config.get_model_config(model_name)
    training_cfg = config.get_training_config()
    dataset_cfg = config.get_dataset_config()
    data_cfg = config.get_data_config()
    output_dir = config.get_output_dir()

    # === Prepare output folder ===
    fold_dir = os.path.join(output_dir, model_name, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    setup_logger(os.path.join(fold_dir, "train.log"))

    # === Verificar e gerar labels.csv automaticamente se necessário ===
    label_csv_path = data_cfg["csv_path"]
    if not os.path.exists(label_csv_path):
        print(f"⚠️ Arquivo de rótulos não encontrado em {label_csv_path}. Gerando automaticamente...")
        image_dir = dataset_cfg["image_dir"]
        generate_labels_from_directory(image_dir, label_csv_path)

    # === Read dataset CSV ===
    df = pd.read_csv(data_cfg["csv_path"])
    labels = df[dataset_cfg["target_col"]].values

    classes = dataset_cfg["class_names"]
    class_ids = np.array([classes.index(label) for label in labels])
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=class_ids)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).cuda()

    # === Stratified K-Fold ===
    skf = StratifiedKFold(n_splits=dataset_cfg["n_folds"], shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(np.zeros(len(labels)), labels))[fold]

    # === Augmentations ===
    aug_strategy = PerClassAugmentation(config)

    # === Datasets & Loaders ===
    input_size = model_cfg.get("input_size", 224)

    train_dataset = TumorDataset(df, data_cfg["image_dir"], train_idx, aug_strategy,
                                 dataset_cfg["input_col"], dataset_cfg["target_col"],
                                 input_size=input_size, is_inference=False)
    val_dataset = TumorDataset(df, data_cfg["image_dir"], val_idx, aug_strategy,
                               dataset_cfg["input_col"], dataset_cfg["target_col"],
                               input_size=input_size, is_inference=False)

    train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg["batch_size"], shuffle=False, num_workers=4)

    # === Model ===    
    model_cfg = config.get_model_config(model_name)

    num_classes = config.get_dataset_config()["num_classes"]
    model = ModelFactory(model_cfg, num_classes).get_model().to("cuda")

    # === Optimizer & Scheduler ===
    learning_rate = model_cfg.get("learning_rate")
    weight_decay = model_cfg.get("weight_decay")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
    
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # porque você quer maximizar val_macro_f1
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # === Loss ===
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)

    # === Training Loop ===
    best_f1 = 0
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    patience = training_cfg["early_stopping"]["patience"]
    early_stop_counter = 0

    for epoch in range(training_cfg["epochs"]):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                loss = criterion(out, y)
                val_losses.append(loss.item())

                probs = torch.softmax(out, dim=1)
                preds = probs.argmax(1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())


        # === Métricas ===
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_f1, val_acc = compute_metrics(
            y_true=all_targets,
            y_pred=all_preds,
            class_names=dataset_cfg["class_names"],
            save_dir=os.path.join(fold_dir, "metrics"),
            epoch=epoch + 1,
            save_summary=False
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)

        logging.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}, Val Acc = {val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(fold_dir, "checkpoint.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logging.info("Early stopping triggered.")
                break

        # scheduler.step()
        scheduler.step(val_f1)


    # === Pós-treino ===
    plot_training_history(history, fold_dir)
    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "log.csv"), index=False)
    
    # === Métricas finais ===

    all_probs = np.array(all_probs)
    
    compute_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        class_names=dataset_cfg["class_names"],
        save_dir=os.path.join(fold_dir, "metrics"),
        epoch=epoch + 1,
        save_summary=True
    )


    # === Visualizações automáticas por fold ===
    analyze_fold(
        model_dir=os.path.join(output_dir, model_name),
        fold=fold,
        class_names=dataset_cfg["class_names"]
    )
    
    # === Registrar desempenho do fold ===
    summary_path = os.path.join(fold_dir, "fold_summary.json")
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 4),
        "final_val_acc": round(val_acc, 4)
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    train_one_fold(args.config, args.model, args.fold)
   
