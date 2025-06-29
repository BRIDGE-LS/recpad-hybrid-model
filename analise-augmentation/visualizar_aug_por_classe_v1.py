import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pathlib import Path

# Caminho das imagens (ajuste conforme necessário)
input_dir = Path("analise-augmentation/originais/")
output_dir = Path("analise-augmentation/augmentadas")
output_dir.mkdir(parents=True, exist_ok=True)

# Estratégias específicas de augmentation por classe
augmentations_por_classe = {
    "ADI": A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
    ]),
    "DEB": A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
    ]),
    "LYM": A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
    ]),
    "MUC": A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
        A.GaussNoise(mean=0, var_limit=(5.0, 20.0), p=0.3),
    ]),
    "MUS": A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=15, p=0.7),
        A.VerticalFlip(p=0.5),
    ]),
    "NOR": A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    ]),
    "STR": A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=20, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]),
    "TUM": A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.8),
        A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), translate_percent=0.05, p=0.5),
    ])
}

# Processar cada imagem
for classe, transform in augmentations_por_classe.items():
    img_path = input_dir / f"{classe}_1.png" if (input_dir / f"{classe}_1.png").exists() else input_dir / f"{classe}_9.png"
    if not img_path.exists():
        print(f"Imagem não encontrada para classe {classe}")
        continue

    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img_rgb)["image"]

    # Concatenar original + aumentado
    comparativo = np.hstack([img_rgb, augmented])
    
    # Salvar imagem comparativa
    output_path = output_dir / f"comparacao_{classe}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(comparativo, cv2.COLOR_RGB2BGR))

    print(f"✅ Comparação salva para {classe} → {output_path}")
