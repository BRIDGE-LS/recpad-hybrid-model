
import torch
from core.model import ModelFactory

# Mock simplificado do config.yaml
mock_config = {
    "model": {
        "num_classes": 8,
        "pretrained": True,
        "freeze_backbone": False
    },
    "models": [
        {"name": "efficientnet_b0", "input_size": 224, "dropout": 0.25, "weight": 0.25},
        {"name": "efficientnet_b3", "input_size": 300, "dropout": 0.30, "weight": 0.35},
        {"name": "efficientnet_b4", "input_size": 380, "dropout": 0.40, "weight": 0.40}
    ]
}

def test_model_factory():
    factory = ModelFactory(mock_config)
    for model_name in ["efficientnet_b0", "efficientnet_b3", "efficientnet_b4"]:
        model_wrapper = factory.get_model(model_name)
        print(f"✅ Modelo carregado: {model_wrapper.name}")
        print(f"   - Input size : {model_wrapper.input_size}")
        print(f"   - Dropout    : {model_wrapper.dropout}")
        dummy_input = torch.randn(1, 3, model_wrapper.input_size, model_wrapper.input_size)
        output = model_wrapper(dummy_input)
        print(f"   - Output shape (dummy): {output.shape}")

if __name__ == "__main__":
    test_model_factory()
