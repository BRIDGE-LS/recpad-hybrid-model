import torch
import torch.nn as nn
from torchvision.models import get_model_weights, get_model

class SingleModel(nn.Module):
    def __init__(self, model: nn.Module, name: str, input_size: int, dropout: float):
        super().__init__()
        self.model = model
        self.name = name
        self.input_size = input_size
        self.dropout = dropout

    def forward(self, x):
        return self.model(x)

    @property
    def feature_extractor(self):
        return self.model.features


class ModelFactory:
    def __init__(self, model_cfg: dict, num_classes: int):
        self.model_cfg = model_cfg
        self.num_classes = num_classes
        self.name = model_cfg["name"]
        self.pretrained = model_cfg.get("pretrained", True)
        self.freeze_backbone = model_cfg.get("freeze_backbone", False)

    def get_model(self) -> SingleModel:
        input_size = self.model_cfg.get("input_size", 224)
        dropout = self.model_cfg.get("dropout", 0.2)

        base_model = self.build_efficientnet(self.name, dropout)
        return SingleModel(base_model, self.name, input_size, dropout)

    def build_efficientnet(self, arch_name: str, dropout: float) -> nn.Module:
        try:
            weights_enum = get_model_weights(arch_name)
            weights = weights_enum.DEFAULT if self.pretrained else None
            model = get_model(arch_name, weights=weights)
        except Exception as e:
            raise ValueError(f"Erro ao construir '{arch_name}': {str(e)}")

        if self.freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, self.num_classes)
        )

        return model
