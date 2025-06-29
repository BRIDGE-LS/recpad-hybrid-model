import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from core.augmentations import AugmentationStrategy


class TumorDataset(Dataset):
    """
    Dataset personalizado para classificação de tecidos do microambiente tumoral gástrico.
    Pode ser usado tanto para treino/validação quanto para inferência.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        indices: list,
        augmentation_strategy: AugmentationStrategy,
        input_col: str = "filename",
        target_col: str = "label",
        input_size: int = 300,
        is_inference: bool = False,
    ):
        """
        Parâmetros:
            dataframe (pd.DataFrame): tabela com metadados (nomes de arquivos e rótulos)
            image_dir (str): caminho da pasta contendo as imagens
            indices (list): lista de índices válidos para treino/validação
            augmentation_strategy (AugmentationStrategy): estratégia a ser aplicada
            input_col (str): nome da coluna com os caminhos relativos das imagens
            target_col (str): nome da coluna com os rótulos (como strings ou ints)
            is_inference (bool): se True, ignora rótulos e usa transformação padrão
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.indices = indices
        self.input_col = input_col
        self.target_col = target_col
        self.augmentation = augmentation_strategy
        self.input_size = input_size
        self.is_inference = is_inference

        # Subconjunto de dados do fold atual
        self.data = self.df.iloc[indices].reset_index(drop=True)

        if not self.is_inference:
            self.label2index = {label: idx for idx, label in enumerate(sorted(self.df[target_col].unique()))}
            self.index2label = {v: k for k, v in self.label2index.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retorna:
            image_tensor: imagem transformada (Tensor)
            label_idx: índice do rótulo (ou -1 se for inferência)
        """
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row[self.input_col])

        # Carregar imagem com OpenCV (BGR → RGB)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Erro ao ler imagem: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_inference:
            # Aplicar transformação neutra (usamos "TUM" como default)
            transform = self.augmentation.get_transform(class_name="TUM", input_size=self.input_size)
            image_tensor = transform(image=image)["image"]
            return image_tensor, -1
        else:
            # Treinamento/validação
            label_name = row[self.target_col]
            label_idx = self.label2index[label_name]
            transform = self.augmentation.get_transform(class_name=label_name, input_size=self.input_size)
            image_tensor = transform(image=image)["image"]
            return image_tensor, label_idx

    def get_labels(self):
        """Útil para avaliar o balanceamento do conjunto"""
        if self.is_inference:
            return []
        return self.data[self.target_col].map(self.label2index).tolist()
