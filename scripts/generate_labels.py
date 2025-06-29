import os
import argparse
import pandas as pd
from pathlib import Path

def generate_labels_from_folders(image_root: str, output_csv: str, valid_exts={".png", ".jpg", ".jpeg"}):
    """
    Percorre uma estrutura de diretórios onde cada subpasta representa uma classe
    e gera um CSV com colunas: filename,label
    """
    records = []
    image_root = Path(image_root)

    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() in valid_exts:
                records.append({
                    "filename": f"{class_name}/{image_path.name}",
                    "label": class_name
                })

    if not records:
        print("❌ Nenhuma imagem válida encontrada.")
        return

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV gerado com {len(df)} imagens em: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um arquivo CSV com labels a partir da estrutura de pastas.")
    parser.add_argument("--image-dir", type=str, required=True, help="Diretório com subpastas por classe (ex: ADI, TUM...)")
    parser.add_argument("--output", type=str, required=True, help="Caminho para salvar o arquivo CSV (ex: data/labels.csv)")
    args = parser.parse_args()

    generate_labels_from_folders(args.image_dir, args.output)
