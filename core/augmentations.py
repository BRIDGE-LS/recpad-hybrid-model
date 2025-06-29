import albumentations as A
import yaml

from abc import ABC, abstractmethod
from typing import Callable
from albumentations.pytorch import ToTensorV2
from core.config_loader import ConfigLoader


class AugmentationStrategy(ABC):
    @abstractmethod
    def get_transform(self, class_name: str = None, input_size: int = None) -> Callable:
        pass


class PerClassAugmentation(AugmentationStrategy):
    def __init__(self, config: ConfigLoader):
        self.cfg = config.get_augmentation_paths()
        self.default_input_size = self.cfg.get("input_size", 224)
        self.mean = tuple(self.cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = tuple(self.cfg.get("std", [0.229, 0.224, 0.225]))
        self.class_transforms = self.cfg.get("classes", {})

    def get_transform(self, class_name: str = None, input_size: int = None) -> Callable:
        if class_name is None:
            raise ValueError("PerClassAugmentation precisa de um class_name.")

        size = input_size if input_size is not None else self.default_input_size
        transforms_config = self.class_transforms.get(class_name, {}).get("transforms", [])

        transforms = [A.Resize(size, size)]

        for t in transforms_config:
            for name, params in t.items():
                aug_class = getattr(A, name)
                if isinstance(params, dict):
                    transforms.append(aug_class(**params))
                else:
                    raise ValueError(f"[{class_name}] Parâmetros inválidos para '{name}': {params}")

        transforms.append(A.Normalize(mean=self.mean, std=self.std))
        transforms.append(ToTensorV2())

        return A.Compose(transforms)


class DefaultAugmentation(AugmentationStrategy):
    def __init__(self, config: ConfigLoader):
        self.cfg = config.get_augmentation_paths()
        self.default_input_size = self.cfg.get("input_size", 224)
        self.mean = tuple(self.cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = tuple(self.cfg.get("std", [0.229, 0.224, 0.225]))

    def get_transform(self, class_name: str = None, input_size: int = None) -> Callable:
        size = input_size if input_size is not None else self.default_input_size

        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])


class NoAugmentation(AugmentationStrategy):
    def __init__(self, config: ConfigLoader):
        self.cfg = config.get_augmentation_paths()
        self.default_input_size = self.cfg.get("input_size", 224)
        self.mean = tuple(self.cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = tuple(self.cfg.get("std", [0.229, 0.224, 0.225]))

    def get_transform(self, class_name: str = None, input_size: int = None) -> Callable:
        size = input_size if input_size is not None else self.default_input_size

        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
