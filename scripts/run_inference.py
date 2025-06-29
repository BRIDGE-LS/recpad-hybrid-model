import argparse
import os
import pandas as pd
from pathlib import Path
from PIL import Image

from core.config_loader import ConfigLoader
from inference import predict_image


def run_batch_inference(image_paths, config, save_path=None):
    results = []

    for img_path in image_paths:
        pred_class, score_dict = predict_image(img_path, config)
        result = {"image": img_path, "predicted_class": pred_class}
        result.update(score_dict)
        results.append(result)
        print(f"🧪 {img_path} → {pred_class} | Scores: {score_dict}")

    if save_path:
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        print(f"\n📁 Resultados salvos em: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Caminho para o config.yaml")
    parser.add_argument("--image", type=str, help="Caminho para uma imagem individual")
    parser.add_argument("--folder", type=str, help="Caminho para uma pasta com imagens")
    parser.add_argument("--save", type=str, help="Arquivo CSV para salvar os resultados")
    args = parser.parse_args()

    config = ConfigLoader(args.config)

    if args.image:
        assert os.path.isfile(args.image), "Imagem não encontrada."
        run_batch_inference([args.image], config, args.save)

    elif args.folder:
        img_dir = Path(args.folder)
        assert img_dir.exists(), "Pasta de imagens não encontrada."
        image_paths = [str(p) for p in img_dir.glob("*.png")]
        run_batch_inference(image_paths, config, args.save)

    else:
        print("❌ Informe --image ou --folder para realizar a inferência.")
