import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from core.augmentations import PerClassAugmentation, ConfigLoader
from core.dataset import TumorDataset

# ====== CONFIGURAÇÃO ======
CONFIG_PATH = "config/config.yaml"
IMAGE_DIR = "test_images"
IMAGE_FILE = "ADI_1.png"
LABEL = "ADI"

# ====== MOCK DATAFRAME ======
df = pd.DataFrame({
    "filename": [IMAGE_FILE],
    "label": [LABEL]
})

# ====== AUGMENTATION E DATASET ======
config = ConfigLoader(CONFIG_PATH)
augmentation = PerClassAugmentation(config)
dataset = TumorDataset(
    dataframe=df,
    image_dir=IMAGE_DIR,
    indices=[0],  # um único item
    augmentation_strategy=augmentation,
    input_col="filename",
    target_col="label"
)

# ====== APLICAR E VISUALIZAR ======
image_tensor, label = dataset[0]
image_np = image_tensor.permute(1, 2, 0).numpy()

# Carregar imagem original
image_original = cv2.imread(os.path.join(IMAGE_DIR, IMAGE_FILE))
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

# Plotar comparação
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Reverter normalização para visualizar
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
image_np = image_tensor.permute(1, 2, 0) * std + mean
image_np = image_np.clamp(0, 1).numpy()


ax[0].imshow(image_original)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(image_np)
ax[1].set_title(f"Augmentada ({LABEL})")
ax[1].axis("off")

plt.tight_layout()

plt.show()
