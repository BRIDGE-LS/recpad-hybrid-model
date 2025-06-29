import yaml
import os


class ConfigLoader:
    """
    Responsável por carregar e acessar o arquivo config.yaml de forma segura e padronizada.
    Permite fallback para valores default.
    """

    def __init__(self, config_path: str):
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key: str, default=None):
        """Acesso genérico ao dicionário principal"""
        return self.config.get(key, default)

    def get_model_config(self, model_name: str):
        """Retorna o bloco de configuração de um modelo específico"""
        models = self.config.get("models", [])
        for model in models:
            if model.get("name") == model_name:
                return model
        raise ValueError(f"Modelo '{model_name}' não encontrado na seção 'models' do config.yaml.")

    def get_training_config(self):
        return self.config.get("training", {})

    def get_output_dir(self):
        return self.config.get("data", {}).get("output_dir", "outputs")


    def get_dataset_config(self):
        return self.config.get("dataset", {})
    
    def get_data_config(self):
        return self.config.get("data", {})

    def get_ensemble_config(self):
        return self.config.get("ensemble", {})

    def get_evaluation_config(self):
        return self.config.get("evaluation", {})

    def get_stats_config(self):
        return self.config.get("stats", {})

    def get_data_paths(self):
        return self.config.get("data", {})
    
    def get_augmentation_paths(self):
        return self.config.get("data", {})


    def get_augmentation_config(self, class_name: str):
        aug_config = self.config.get("augmentations", {})
        class_transforms = aug_config.get("classes", {}).get(class_name)
        
        if class_transforms is None:
            raise ValueError(f"Transformações para a classe '{class_name}' não encontradas em 'augmentations.classes'.")
        
        return class_transforms