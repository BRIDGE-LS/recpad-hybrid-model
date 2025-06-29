# core/ensemble.py

from typing import List
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

from core.model import ModelFactory
from core.dataset import TumorDataset
from core.augmentations import PerClassAugmentation
from core.metrics import compute_metrics

class EnsemblePredictor:
    def __init__(self, config, model_names: List[str], fold_count: int, device="cuda"):
        self.config = config
        self.model_names = model_names
        self.fold_count = fold_count
        self.device = device
        self.weights = {
            model["name"]: model.get("weight", 1.0)
            for model in config.config["models"]
        }

    def load_model(self, model_name: str, fold: int):
        model_cfg = self.config.get_model_config(model_name)
        model = ModelFactory(model_cfg).to(self.device)
        model_path = os.path.join(
            self.config.get_output_dir(), model_name, f"fold_{fold}", "checkpoint.pth"
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def predict(self, df: pd.DataFrame):
        dataset_cfg = self.config.get_dataset_config()
        aug = PerClassAugmentation(self.config)
        dataset = TumorDataset(
            dataframe=df,
            image_dir=dataset_cfg["image_dir"],
            indices=list(range(len(df))),
            augmentation_strategy=aug,
            input_col=dataset_cfg["input_col"],
            target_col=dataset_cfg["target_col"]
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        all_logits = []
        for model_name in self.model_names:
            model_logits = np.zeros((len(df), self.config.get_dataset_config()["num_classes"]))
            for fold in range(self.fold_count):
                model = self.load_model(model_name, fold)
                idx = 0
                with torch.no_grad():
                    for x, _ in tqdm(loader, desc=f"{model_name} Fold {fold}"):
                        x = x.to(self.device)
                        out = model(x)
                        probs = F.softmax(out, dim=1).cpu().numpy()
                        batch_size = x.size(0)
                        model_logits[idx:idx+batch_size] += probs
                        idx += batch_size
            model_logits /= self.fold_count  # média por fold
            model_logits *= self.weights[model_name]
            all_logits.append(model_logits)

        # === Agregação por modelo ===
        final_logits = np.sum(all_logits, axis=0)
        final_preds = final_logits.argmax(axis=1)
        return final_preds, final_logits

    def evaluate(self, df: pd.DataFrame):
        y_true = df[self.config.get_dataset_config()["target_col"]].values
        y_pred, logits = self.predict(df)

        save_dir = os.path.join(self.config.get_output_dir(), "ensemble")
        os.makedirs(save_dir, exist_ok=True)

        f1, acc = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.config.get_dataset_config()["class_names"],
            save_dir=save_dir,
            save_summary=True
        )

        result = {
            "ensemble_f1": f1,
            "ensemble_acc": acc
        }

        with open(os.path.join(save_dir, "ensemble_summary.json"), "w") as f:
            json.dump(result, f, indent=4)

        return result
