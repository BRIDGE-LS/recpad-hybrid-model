import os
import pandas as pd
from pathlib import Path

def generate_labels_from_directory(image_dir: str, output_csv: str):
    """
    Gera um arquivo CSV com colunas filename e label a partir de uma estrutura
    de diretórios onde cada subpasta representa uma classe.
    """
    image_dir = Path(image_dir)
    rows = []
    for class_dir in image_dir.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"]:
                    rows.append({
                        "filename": os.path.join(label, img_path.name),
                        "label": label
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Arquivo de rótulos gerado automaticamente em: {output_csv}")
