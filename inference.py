import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

from core.config_loader import ConfigLoader
from core.model import ModelFactory
from core.augmentations import PerClassAugmentation


def load_models_for_ensemble(config):
    models_cfg = config.config["models"]
    dataset_cfg = config.config["dataset"]
    output_dir = config.get_data_paths()["output_dir"]

    all_models = []
    for model_cfg in models_cfg:
        name = model_cfg["name"]
        weight = model_cfg["weight"]
        n_folds = dataset_cfg["folds"]

        fold_models = []
        for fold in range(n_folds):
            model_path = os.path.join(output_dir, name, f"fold_{fold}", "checkpoint.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

            model = ModelFactory(config.config).get_model(name)
            model.load_state_dict(torch.load(model_path, map_location="cuda"))
            model.eval()
            model.cuda()
            fold_models.append(model)

        all_models.append({"name": name, "models": fold_models, "weight": weight})

    return all_models


def prepare_image(image_path, class_name, transform):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Imagem inválida: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).cuda()
    return tensor


def predict_with_ensemble(models, image_tensor):
    """
    Executa a predição por ensemble com média ponderada dos modelos (soft voting).
    Cada modelo pode ter múltiplos folds.

    Args:
        models (list): Lista de dicionários contendo:
            - name: nome do modelo
            - models: lista de modelos treinados (um por fold)
            - weight: peso associado ao modelo no ensemble
        image_tensor (Tensor): Tensor de imagem com shape [1, C, H, W]

    Returns:
        final_prob (np.ndarray): Vetor de probabilidades normalizado (num_classes,)
    """
    # === Detectar número de classes com base na estrutura de classifier
    first_model = models[0]["models"][0]
    if hasattr(first_model.model, "classifier") and isinstance(first_model.model.classifier, torch.nn.Sequential):
        num_classes = first_model.model.classifier[-1].out_features
    else:
        raise AttributeError("Não foi possível determinar o número de classes do modelo.")

    total_prob = torch.zeros(num_classes).cuda()
    total_weight = 0

    for model_group in models:
        weight = model_group["weight"]
        fold_probs = []

        for model in model_group["models"]:
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                prob = torch.softmax(output, dim=1).squeeze(0)  # shape: (num_classes,)
                fold_probs.append(prob)

        avg_prob = torch.stack(fold_probs).mean(dim=0)  # shape: (num_classes,)
        total_prob += weight * avg_prob
        total_weight += weight

    final_prob = total_prob / total_weight
    return final_prob.cpu().numpy()



def main(args):
    config = ConfigLoader(args.config)
    class_names = config.config["dataset"]["class_names"]

    models = load_models_for_ensemble(config)
    aug = PerClassAugmentation(config)

    results = []
    input_path = Path(args.input)

    image_paths = [input_path] if input_path.is_file() else list(input_path.glob("*.png"))

    for img_path in tqdm(image_paths, desc="Inferência"):
        class_name = args.class_name or "TUM"  # se não for especificado, assume genérico
        transform = aug.get_transform(class_name)
        image_tensor = prepare_image(str(img_path), class_name, transform)
        probs = predict_with_ensemble(models, image_tensor)

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        pred_score = float(probs[pred_idx])

        results.append({
            "filename": img_path.name,
            "predicted_label": pred_label,
            "confidence": round(pred_score, 4),
            "distribution": probs.tolist()
        })

    df = pd.DataFrame(results)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"✅ Resultados salvos em {args.output}")
    else:
        print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Imagem ou diretório de imagens")
    parser.add_argument("--class_name", type=str, help="Classe para seleção de estratégia de augmentation")
    parser.add_argument("--output", type=str, help="Caminho para salvar CSV com os resultados")
    args = parser.parse_args()
    main(args)
