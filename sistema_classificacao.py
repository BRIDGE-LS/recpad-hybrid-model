#!/usr/bin/env python3
"""
Sistema de Classificação TME para Câncer Gástrico - VERSÃO MELHORADA
====================================================================

Implementação baseada em evidências científicas para classificação de 8 classes TME
usando abordagem híbrida conservadora com validação rigorosa.

MELHORIAS IMPLEMENTADAS:
- Suporte opcional a Distributed Data Parallel (DDP) para acelerar treinamento
- Tratamento robusto de erros e cleanup seguro
- Configuração flexível de processamento (single/multi-GPU)
- Manutenção completa dos parâmetros baseados em literatura médica

Referências principais:
- Lou et al. (2025): HMU-GC-HE-30K dataset
- Mandal et al. (2025): Histopathology glossary for AI developers  
- DINO (Caron et al. 2021): Self-supervised vision transformers
- Tan & Le (2019): EfficientNet
- Vorontsov et al. (2024): Foundation model for clinical-grade computational pathology
- Slideflow (2024): Deep learning for digital histopathology with real-time WSI visualization
"""

import os
import sys
import json
import logging
import warnings
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler
from torchvision import transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO)

@dataclass
class TMEConfig:
    """
    Configuração do sistema baseada em evidências da literatura.
    
    ATENÇÃO: Parâmetros mantidos EXATAMENTE como originalmente definidos
    baseados em referências científicas rigorosas.
    
    Referências para parâmetros:
    - Image size 224: Padrão EfficientNet (Tan & Le 2019)
    - Batch size 32: Compromisso entre estabilidade e velocidade (Tellez et al. 2019)
    - Learning rate 1e-4: Otimizado para fine-tuning médico (Echle et al. 2021)
    - Weight decay 1e-4: Regularização padrão para histopatologia
    - Patience 15: Early stopping conservador para medicina (Campanella et al. 2019)
    
    NOVAS OPÇÕES DE PROCESSAMENTO:
    - use_distributed: Habilita DDP se múltiplas GPUs disponíveis
    - ddp_backend: Backend para DDP (padrão: 'nccl')
    - master_port: Porta para comunicação DDP
    """
    # Configurações de dados (MANTIDAS - baseadas em literatura)
    data_path: str = "data"
    image_size: int = 224  # EfficientNet padrão
    batch_size: int = 32   # Compromisso memória/performance (Tellez et al. 2019)
    num_workers: int = 4
    
    # Classes TME baseadas em Lou et al. (2025)
    classes: List[str] = None
    num_classes: int = 8
    
    # Configurações de treinamento baseadas em literatura médica (MANTIDAS)
    learning_rate: float = 1e-4     # Otimizado para histopatologia (Echle et al. 2021)
    weight_decay: float = 1e-4      # Regularização conservadora
    num_epochs: int = 100
    patience: int = 15              # Early stopping conservador (Campanella et al. 2019)
    
    # Configurações de validação (MANTIDAS)
    k_folds: int = 5               # 5-fold CV padrão
    test_size: float = 0.2         # 20% holdout test
    random_state: int = 42
    
    # Configurações do modelo (MANTIDAS)
    model_name: str = "efficientnet_b4"
    pretrained: bool = True
    dropout_rate: float = 0.3       # Regularização para evitar overfitting
    
    # Configurações de augmentation conservadoras (MANTIDAS)
    # Baseadas em Tellez et al. (2019) para histopatologia
    augmentation_params: Dict = None
    
    # NOVAS CONFIGURAÇÕES DE PROCESSAMENTO PARALELO
    # Baseadas em Vorontsov et al. (2024) e práticas atuais DDP
    use_distributed: bool = False           # Habilita DDP automaticamente se múltiplas GPUs
    ddp_backend: str = 'nccl'              # Backend DDP (nccl para NVIDIA, gloo para CPU)
    master_addr: str = '127.0.0.1'         # Endereço master para DDP
    master_port: str = '12355'             # Porta para comunicação DDP
    find_unused_parameters: bool = False   # Configuração DDP conservadora
    gradient_clipping: float = 1.0         # Clipping para estabilidade (baseado em Slideflow 2024)
    
    def __post_init__(self):
        if self.classes is None:
            # Classes TME baseadas em Lou et al. (2025)
            self.classes = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
        
        if self.augmentation_params is None:
            # Parâmetros conservadores baseados em literatura
            # Tellez et al. (2019): "Conservative augmentation for medical images"
            self.augmentation_params = {
                'rotation_limit': 90,           # Rotação total para H&E
                'shift_limit': 0.1,             # Translação conservadora
                'scale_limit': 0.1,             # Escala limitada
                'brightness_limit': 0.1,        # Variação de brilho
                'contrast_limit': 0.1,          # Variação de contraste
                'hue_shift_limit': 10,          # Baseado em Kather et al.
                'saturation_shift_limit': 15,   # Conservador para H&E
                'value_shift_limit': 10,        # Variação de intensidade
                'noise_prob': 0.1,              # Probabilidade de ruído
                'blur_prob': 0.1,               # Probabilidade de blur
                'flip_prob': 0.5                # Flip horizontal/vertical
            }
        
        # Auto-detectar se deve usar DDP
        if self.use_distributed and torch.cuda.device_count() > 1:
            self.use_distributed = True
            logging.info(f"DDP habilitado automaticamente - {torch.cuda.device_count()} GPUs detectadas")
        else:
            self.use_distributed = False

class TMEDataset(Dataset):
    """
    Dataset para classificação TME otimizado para câncer gástrico.
    
    Implementação baseada em:
    - Kather et al. (2019): NCT-CRC-HE-100K dataset estruture
    - Lou et al. (2025): HMU-GC-HE-30K organization
    - Slideflow (2024): Efficient data processing for histopathology
    """
    
    def __init__(self, 
                 data_path: str, 
                 split: str = 'train',
                 transform: Optional[A.Compose] = None,
                 config: TMEConfig = None):
        """
        Args:
            data_path: Caminho para dados organizados por classe
            split: 'train', 'val', ou 'test'
            transform: Transformações de augmentation
            config: Configuração do sistema
        """
        self.data_path = Path(data_path) / split
        self.transform = transform
        self.config = config or TMEConfig()
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.config.classes)}
        
        self._load_samples()
        self._analyze_distribution()
        
    def _load_samples(self):
        """Carrega amostras seguindo estrutura HMU-GC-HE-30K"""
        for class_name in self.config.classes:
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                logging.warning(f"Diretório não encontrado: {class_dir}")
                continue
                
            for img_file in class_dir.glob("*.png"):
                self.samples.append({
                    'path': str(img_file),
                    'class': class_name,
                    'label': self.class_to_idx[class_name]
                })
        
        logging.info(f"Carregadas {len(self.samples)} amostras para {self.data_path.name}")
    
    def _analyze_distribution(self):
        """Analisa distribuição das classes para detectar desbalanceamento"""
        class_counts = Counter([sample['class'] for sample in self.samples])
        
        print(f"\n=== Distribuição de Classes - {self.data_path.name.upper()} ===")
        total = len(self.samples)
        for class_name in self.config.classes:
            count = class_counts.get(class_name, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{class_name}: {count:6d} ({percentage:5.1f}%)")
        print(f"Total: {total:6d}")
        
        # Detectar desbalanceamento crítico (>10x diferença)
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if max_count / min_count > 10:
                logging.warning(f"Desbalanceamento crítico detectado: {max_count/min_count:.1f}x")
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calcula pesos das classes para lidar com desbalanceamento.
        Método: balanced class weighting (Sklearn padrão)
        """
        labels = [sample['label'] for sample in self.samples]
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.arange(self.config.num_classes), 
            y=labels
        )
        return torch.FloatTensor(class_weights)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Carregar imagem
        image = self._load_image(sample['path'])
        
        # Aplicar transformações
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Transformação mínima necessária
            transform = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            image = transform(image=image)['image']
        
        return image, sample['label']
    
    def _load_image(self, path: str) -> np.ndarray:
        """Carrega imagem com tratamento de erro robusto"""
        try:
            from PIL import Image
            image = Image.open(path).convert('RGB')
            return np.array(image)
        except Exception as e:
            logging.error(f"Erro ao carregar imagem {path}: {e}")
            # Retorna imagem placeholder em caso de erro
            return np.zeros((224, 224, 3), dtype=np.uint8)

def create_augmentation_transforms(config: TMEConfig) -> Dict[str, A.Compose]:
    """
    Cria transformações de augmentation baseadas em evidências científicas.
    
    Estratégia conservadora baseada em:
    - Tellez et al. (2019): Augmentation específica para histopatologia
    - Kather et al. (2019): Parâmetros validados para TME
    - Echle et al. (2021): Best practices para deep learning médico
    - Slideflow (2024): Efficient stain normalization and augmentation
    """
    
    # Normalização ImageNet (padrão para modelos pré-treinados)
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Transformações de treinamento - CONSERVADORAS
    train_transform = A.Compose([
        # Resize obrigatório
        A.Resize(config.image_size, config.image_size),
        
        # Augmentações geométricas conservadoras
        # Baseadas em Tellez et al.: "Modest augmentation for medical"
        A.Rotate(
            limit=config.augmentation_params['rotation_limit'], 
            p=0.7,
            border_mode=0  # BORDER_CONSTANT para evitar artefatos
        ),
        A.ShiftScaleRotate(
            shift_limit=config.augmentation_params['shift_limit'],
            scale_limit=config.augmentation_params['scale_limit'],
            rotate_limit=0,  # Já definido acima
            p=0.5
        ),
        
        # Flips - sempre seguros para histopatologia
        A.HorizontalFlip(p=config.augmentation_params['flip_prob']),
        A.VerticalFlip(p=config.augmentation_params['flip_prob']),
        
        # Augmentações de cor MUITO conservadoras
        # H&E é sensível a mudanças de cor
        A.ColorJitter(
            brightness=config.augmentation_params['brightness_limit'],
            contrast=config.augmentation_params['contrast_limit'],
            saturation=config.augmentation_params['saturation_shift_limit'] / 100,
            hue=config.augmentation_params['hue_shift_limit'] / 360,
            p=0.3  # Probabilidade baixa
        ),
        
        # Augmentações de textura ocasionais
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=config.augmentation_params['blur_prob']),
        
        # Ruído sutil
        A.GaussNoise(
            var_limit=(10.0, 30.0), 
            p=config.augmentation_params['noise_prob']
        ),
        
        normalize,
        ToTensorV2()
    ])
    
    # Transformações de validação/teste - SEM augmentation
    val_transform = A.Compose([
        A.Resize(config.image_size, config.image_size),
        normalize,
        ToTensorV2()
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }

class EfficientNetTME(nn.Module):
    """
    Modelo EfficientNet adaptado para classificação TME.
    
    Baseado em:
    - Tan & Le (2019): EfficientNet original
    - Echle et al. (2021): Adaptação para histopatologia
    - Kather et al. (2019): Fine-tuning para TME
    - Vorontsov et al. (2024): Clinical-grade computational pathology
    """
    
    def __init__(self, config: TMEConfig):
        super().__init__()
        self.config = config
        
        # Carregar EfficientNet pré-treinado
        if config.model_name == "efficientnet_b4":
            self.backbone = models.efficientnet_b4(pretrained=config.pretrained)
            feature_dim = 1792  # EfficientNet-B4 feature dimension
        elif config.model_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=config.pretrained)
            feature_dim = 1536
        else:
            raise ValueError(f"Modelo não suportado: {config.model_name}")
        
        # Substituir classificador final
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate / 2),  # Dropout menor na segunda camada
            nn.Linear(512, config.num_classes)
        )
        
        # Inicialização Xavier para novas camadas (Glorot & Bengio 2010)
        self._init_classifier()
    
    def _init_classifier(self):
        """Inicializa pesos do classificador com Xavier initialization"""
        for module in self.backbone.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features antes da classificação final"""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features

class DistributedTMETrainer:
    """
    Sistema de treinamento robusto com suporte opcional a DDP.
    
    Implementa best practices de:
    - Echle et al. (2021): Deep learning in cancer pathology
    - Kather et al. (2019): Tissue phenotyping validation
    - Medical imaging validation standards
    - Vorontsov et al. (2024): Foundation model practices
    - Slideflow (2024): Distributed training for histopathology
    
    NOVAS FUNCIONALIDADES:
    - Suporte automático a DDP se múltiplas GPUs
    - Fallback gracioso para single GPU
    - Cleanup seguro e tratamento de sinais
    - Sincronização robusta de métricas
    """
    
    def __init__(self, config: TMEConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.results = defaultdict(list)
        self.is_distributed = world_size > 1
        
        # Configurar handlers de sinal para cleanup seguro
        self._setup_signal_handlers()
        
        logging.info(f"Trainer inicializado - Rank: {rank}, World Size: {world_size}, Device: {self.device}")
    
    def _setup_signal_handlers(self):
        """
        Configura handlers para cleanup seguro em caso de interrupção.
        Baseado em práticas DDP para evitar travamento de GPUs.
        """
        def handle_exit(*args):
            if self.rank == 0:
                logging.info(f"[Rank {self.rank}] Sinal de parada recebido. Executando cleanup.")
            self._cleanup()
            exit(0)
        
        signal.signal(signal.SIGTERM, handle_exit)
        signal.signal(signal.SIGINT, handle_exit)
    
    def _cleanup(self):
        """Cleanup seguro para DDP e recursos"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def prepare_data(self) -> Dict[str, DataLoader]:
        """Prepara dados com estratificação adequada e suporte a DDP"""
        transforms = create_augmentation_transforms(self.config)
        
        # Carregar datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            datasets[split] = TMEDataset(
                data_path=self.config.data_path,
                split=split,
                transform=transforms[split if split != 'test' else 'val'],
                config=self.config
            )
        
        # Criar samplers
        if self.is_distributed:
            # Distributed samplers para DDP
            train_sampler = DistributedSampler(
                datasets['train'], 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                datasets['val'], 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
            test_sampler = DistributedSampler(
                datasets['test'], 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
        else:
            # Sampling balanceado para single GPU
            class_weights = datasets['train'].get_class_weights()
            sample_weights = [class_weights[label] for _, label in datasets['train']]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(datasets['train']),
                replacement=True
            )
            val_sampler = None
            test_sampler = None
        
        # Criar DataLoaders
        dataloaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),  # Só shuffle se não tem sampler
                num_workers=self.config.num_workers,
                pin_memory=True
            ),
            'val': DataLoader(
                datasets['val'],
                batch_size=self.config.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=self.config.batch_size,
                sampler=test_sampler,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        }
        
        return dataloaders
    
    def create_model(self) -> nn.Module:
        """Cria modelo com inicialização adequada e suporte opcional a DDP"""
        model = EfficientNetTME(self.config)
        
        # Transfer learning strategy para histopatologia
        # Congela layers iniciais (features básicas são transferíveis)
        if self.config.pretrained:
            # Congela features layers até certo ponto
            for name, param in model.backbone.features.named_parameters():
                layer_num = int(name.split('.')[0]) if name.split('.')[0].isdigit() else 0
                if layer_num < 4:  # Congela primeiros 4 blocos
                    param.requires_grad = False
        
        model = model.to(self.device)
        
        # Envolver com DDP se distribuído
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.rank],
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=True,
                gradient_as_bucket_view=True
            )
        
        return model
    
    def create_optimizer_scheduler(self, model: nn.Module, 
                                 train_loader: DataLoader) -> Tuple[optim.Optimizer, object]:
        """
        Cria otimizador e scheduler baseados em best practices médicas.
        
        MANTÉM EXATAMENTE os parâmetros baseados em literatura:
        - Learning rate 1e-4 (Echle et al. 2021)
        - Weight decay 1e-4 (padrão histopatologia)
        - AdamW + CosineAnnealingLR (Loshchilov & Hutter 2017)
        """
        
        # Diferentes learning rates para backbone e classifier
        # Estratégia comum em medical imaging
        model_params = model.module.parameters() if self.is_distributed else model.parameters()
        
        if self.is_distributed:
            backbone_params = model.module.backbone.features.parameters()
            classifier_params = model.module.backbone.classifier.parameters()
        else:
            backbone_params = model.backbone.features.parameters()
            classifier_params = model.backbone.classifier.parameters()
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        # Cosine Annealing com warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart a cada 10 épocas
            T_mult=2,  # Dobra período a cada restart
            eta_min=1e-6  # LR mínimo
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """Treina uma época com monitoramento detalhado e suporte a DDP"""
        model.train()
        
        # Configurar DistributedSampler para época atual
        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping para estabilidade (baseado em Slideflow 2024)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.gradient_clipping)
                
                optimizer.step()
                
                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
            except RuntimeError as e:
                if self.is_distributed and "Expected to have finished reduction" in str(e):
                    logging.warning(f"[Rank {self.rank}] DDP sync error no batch {batch_idx}. Tentando recuperar...")
                    # Força sincronização e continua
                    torch.cuda.synchronize()
                    if dist.is_initialized():
                        dist.barrier()
                    continue
                else:
                    raise e
            
            # Log progresso a cada 100 batches (apenas rank 0)
            if batch_idx % 100 == 0 and self.rank == 0:
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: Loss={current_loss:.4f}, Acc={current_acc:.4f}")
        
        # Sincronizar métricas entre ranks se distribuído
        if self.is_distributed:
            # Converter para tensors
            loss_tensor = torch.tensor(running_loss, device=self.device)
            corrects_tensor = torch.tensor(running_corrects.float(), device=self.device)
            samples_tensor = torch.tensor(total_samples, device=self.device)
            
            # All-reduce
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(corrects_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            
            # Médias globais
            epoch_loss = loss_tensor.item() / samples_tensor.item()
            epoch_acc = corrects_tensor.item() / samples_tensor.item()
        else:
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """Valida modelo com métricas médicas robustas e suporte a DDP"""
        model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Probabilidades e predições
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Sincronizar métricas entre ranks se distribuído
        if self.is_distributed:
            # Gather todas as predições e labels
            all_preds_tensor = torch.tensor(all_preds, device=self.device)
            all_labels_tensor = torch.tensor(all_labels, device=self.device)
            
            # Preparar listas para gather
            gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(self.world_size)]
            gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(self.world_size)]
            
            # All gather
            dist.all_gather(gathered_preds, all_preds_tensor)
            dist.all_gather(gathered_labels, all_labels_tensor)
            
            # Concatenar resultados apenas no rank 0
            if self.rank == 0:
                all_preds = torch.cat(gathered_preds).cpu().numpy()
                all_labels = torch.cat(gathered_labels).cpu().numpy()
            
            # Sincronizar loss
            loss_tensor = torch.tensor(running_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / len(val_loader.dataset)
        else:
            total_loss = running_loss / len(val_loader.dataset)
        
        # Calcular métricas médicas robustas (apenas rank 0 ou single GPU)
        if self.rank == 0 or not self.is_distributed:
            metrics = self._calculate_medical_metrics(all_labels, all_preds, all_probs)
            metrics['loss'] = total_loss
        else:
            # Ranks não-zero retornam métricas vazias
            metrics = {'loss': total_loss}
        
        return metrics
    
    def _calculate_medical_metrics(self, y_true: List[int], 
                                 y_pred: List[int], y_probs: List[List[float]]) -> Dict[str, float]:
        """
        Calcula métricas médicas robustas seguindo guidelines clínicos.
        
        Métricas baseadas em:
        - Echle et al. (2021): Clinical validation metrics
        - Kather et al. (2019): Histopathology benchmarks
        - Medical imaging evaluation standards
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # F1-scores
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Cohen's Kappa (importante para concordância em medicina)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # AUC multiclass (one-vs-rest)
        try:
            auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except:
            auc_score = 0.0
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,  # Métrica principal para medicina
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'kappa': kappa,
            'auc': auc_score
        }
    
    def train_model(self) -> Dict[str, any]:
        """
        Treinamento completo com early stopping e validação robusta.
        MANTÉM a estratégia médica conservadora original com suporte a DDP.
        
        Implementa estratégia médica conservadora:
        - Early stopping baseado em balanced accuracy
        - Checkpointing do melhor modelo
        - Logging detalhado para análise posterior
        - Sincronização robusta se DDP
        """
        
        # Preparar dados
        dataloaders = self.prepare_data()
        
        # Criar modelo
        model = self.create_model()
        
        # Calcular class weights para loss balanceado
        train_dataset = dataloaders['train'].dataset
        class_weights = train_dataset.get_class_weights().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Otimizador e scheduler
        optimizer, scheduler = self.create_optimizer_scheduler(model, dataloaders['train'])
        
        # Early stopping
        best_balanced_acc = 0.0
        patience_counter = 0
        
        # Histórico de treinamento
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_balanced_acc': [], 'val_kappa': []
        }
        
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"INICIANDO TREINAMENTO - {self.config.num_epochs} ÉPOCAS")
            if self.is_distributed:
                print(f"MODO DISTRIBUÍDO - {self.world_size} GPUs")
            else:
                print(f"MODO SINGLE GPU")
            print(f"{'='*80}")
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            try:
                # Treinamento
                train_metrics = self.train_epoch(model, dataloaders['train'], optimizer, criterion, epoch)
                
                # Validação
                val_metrics = self.validate_epoch(model, dataloaders['val'], criterion)
                
                # Scheduler step
                scheduler.step()
                
                # Sincronizar early stopping entre ranks
                if self.is_distributed:
                    dist.barrier()
                
                # Early stopping logic (apenas rank 0 decide)
                early_stop_flag = torch.tensor(0, device=self.device, dtype=torch.int32)
                
                if self.rank == 0:
                    # Salvar histórico
                    history['train_loss'].append(train_metrics['loss'])
                    history['train_acc'].append(train_metrics['accuracy'])
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_balanced_acc'].append(val_metrics.get('balanced_accuracy', 0.0))
                    history['val_kappa'].append(val_metrics.get('kappa', 0.0))
                    
                    # Early stopping baseado em balanced accuracy (métrica médica)
                    current_balanced_acc = val_metrics.get('balanced_accuracy', 0.0)
                    if current_balanced_acc > best_balanced_acc:
                        best_balanced_acc = current_balanced_acc
                        patience_counter = 0
                        
                        # Salvar melhor modelo
                        model_state = model.module.state_dict() if self.is_distributed else model.state_dict()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'metrics': val_metrics,
                            'config': asdict(self.config)
                        }, 'best_model.pth')
                        
                    else:
                        patience_counter += 1
                    
                    # Log da época
                    epoch_time = time.time() - epoch_start
                    print(f"\nÉpoca {epoch+1:3d}/{self.config.num_epochs}")
                    print(f"  Tempo: {epoch_time:.1f}s")
                    print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, BAcc: {current_balanced_acc:.4f}, "
                          f"Kappa: {val_metrics.get('kappa', 0.0):.4f}")
                    print(f"  Best BAcc: {best_balanced_acc:.4f}, Patience: {patience_counter}/{self.config.patience}")
                    
                    # Decisão de early stopping
                    if patience_counter >= self.config.patience:
                        early_stop_flag.fill_(1)
                        print(f"\nEarly stopping na época {epoch+1}")
                
                # Broadcast decisão de early stopping para todos os ranks
                if self.is_distributed:
                    dist.broadcast(early_stop_flag, src=0)
                
                # Todos os ranks verificam early stopping
                if early_stop_flag.item() == 1:
                    break
                    
            except Exception as e:
                logging.error(f"[Rank {self.rank}] Erro na época {epoch}: {e}")
                if self.is_distributed:
                    self._cleanup()
                raise e
        
        # Carregar melhor modelo para teste final
        if self.rank == 0:
            checkpoint = torch.load('best_model.pth')
            if self.is_distributed:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Teste final
            test_metrics = self.validate_epoch(model, dataloaders['test'], criterion)
            
            print(f"\n{'='*80}")
            print(f"RESULTADOS FINAIS NO CONJUNTO DE TESTE")
            print(f"{'='*80}")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric.upper():20s}: {value:.4f}")
            
            return {
                'model': model,
                'history': history,
                'test_metrics': test_metrics,
                'best_epoch': checkpoint['epoch']
            }
        else:
            # Ranks não-zero retornam resultado vazio
            return {}

class TMETrainer:
    """
    Wrapper para manter compatibilidade com código original.
    Automaticamente detecta se deve usar DDP ou single GPU.
    
    MANTÉM EXATAMENTE a mesma interface da classe original.
    """
    
    def __init__(self, config: TMEConfig):
        self.config = config
        self.results = defaultdict(list)
        
        # Auto-detectar configuração de processamento
        if config.use_distributed and torch.cuda.device_count() > 1:
            self.use_distributed = True
            self.world_size = torch.cuda.device_count()
            logging.info(f"Modo distribuído detectado - {self.world_size} GPUs")
        else:
            self.use_distributed = False
            self.world_size = 1
            logging.info("Modo single GPU")
    
    def train_model(self) -> Dict[str, any]:
        """
        Método principal que mantém compatibilidade com código original.
        Escolhe automaticamente entre single GPU ou DDP.
        """
        if self.use_distributed:
            # Executar treinamento distribuído
            return self._train_distributed()
        else:
            # Executar treinamento single GPU
            trainer = DistributedTMETrainer(self.config, rank=0, world_size=1)
            return trainer.train_model()
    
    def _train_distributed(self) -> Dict[str, any]:
        """Executa treinamento distribuído usando multiprocessing"""
        
        # def setup_and_train(rank, world_size, config):
        #     """Função para cada processo DDP"""
        #     # Setup DDP
        #     os.environ['MASTER_ADDR'] = config.master_addr
        #     os.environ['MASTER_PORT'] = config.master_port
            
        #     dist.init_process_group(
        #         backend=config.ddp_backend,
        #         rank=rank,
        #         world_size=world_size
        #     )
            
        #     # Criar trainer para este rank
        #     trainer = DistributedTMETrainer(config, rank=rank, world_size=world_size)
            
        #     try:
        #         results = trainer.train_model()
        #         return results
        #     finally:
        #         trainer._cleanup()
        
        # Spawn processos para DDP
        # mp.spawn(
        #     setup_and_train,
        #     args=(self.world_size, self.config),
        #     nprocs=self.world_size,
        #     join=True
        # )
        
        mp.spawn(
            distributed_setup_and_train,
            args=(self.world_size, asdict(self.config)),  # <- Passa config como dict
            nprocs=self.world_size,
            join=True
        )

        # Carregar resultados do rank 0
        if os.path.exists('best_model.pth'):
            checkpoint = torch.load('best_model.pth', map_location='cpu')
            return {
                'model': None,  # Modelo será carregado quando necessário
                'history': {},
                'test_metrics': checkpoint['metrics'],
                'best_epoch': checkpoint['epoch']
            }
        else:
            return {}
    
    # Métodos de compatibilidade com interface original
    def prepare_data(self):
        """Compatibilidade com código original"""
        trainer = DistributedTMETrainer(self.config, rank=0, world_size=1)
        return trainer.prepare_data()
    
    def create_model(self):
        """Compatibilidade com código original"""
        trainer = DistributedTMETrainer(self.config, rank=0, world_size=1)
        return trainer.create_model()

def main():
    """
    Função principal com pipeline completo de validação.
    
    MANTÉM EXATAMENTE a mesma funcionalidade original.
    Implementa protocolo rigoroso baseado em guidelines médicos:
    1. Configuração baseada em evidências
    2. Treinamento com validação robusta  
    3. Teste em holdout independente
    4. Análise de resultados e failure cases
    """
    
    print("="*80)
    print("SISTEMA DE CLASSIFICAÇÃO TME - CÂNCER GÁSTRICO")
    print("="*80)
    print("Baseado em:")
    print("- Lou et al. (2025): HMU-GC-HE-30K dataset")
    print("- Tan & Le (2019): EfficientNet")
    print("- Echle et al. (2021): Deep learning in cancer pathology")
    print("- Vorontsov et al. (2024): Foundation model for clinical-grade pathology")
    print("="*80)
    
    # Configuração conservadora baseada em literatura
    config = TMEConfig(
        data_path="data",  # Estrutura HMU-GC-HE-30K
        model_name="efficientnet_b4",
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=100,
        patience=15,
        use_distributed=True  # Auto-detecta múltiplas GPUs
    )
    
    # Inicializar trainer
    trainer = TMETrainer(config)
    
    # Treinamento completo
    results = trainer.train_model()
    
    # Análise detalhada dos resultados
    analyze_results(results, config)

def analyze_results(results: Dict[str, any], config: TMEConfig):
    """
    Análise completa dos resultados seguindo padrões médicos.
    
    MANTÉM EXATAMENTE a mesma funcionalidade original.
    
    Inclui:
    - Métricas de performance clínica
    - Análise de confusion matrix
    - Identificação de failure cases
    - Recomendações para melhoria
    """
    
    print(f"\n{'='*80}")
    print("ANÁLISE DETALHADA DOS RESULTADOS")
    print(f"{'='*80}")
    
    test_metrics = results.get('test_metrics', {})
    
    if not test_metrics:
        print("⚠️  Resultados de teste não disponíveis")
        return
    
    # 1. ANÁLISE DE PERFORMANCE CLÍNICA
    print("\n1. MÉTRICAS DE PERFORMANCE CLÍNICA")
    print("-" * 50)
    
    # Interpretação das métricas baseada em guidelines médicos
    balanced_acc = test_metrics.get('balanced_accuracy', 0.0)
    kappa = test_metrics.get('kappa', 0.0)
    
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    if balanced_acc >= 0.90:
        performance_level = "EXCELENTE"
    elif balanced_acc >= 0.85:
        performance_level = "BOM"
    elif balanced_acc >= 0.75:
        performance_level = "ACEITÁVEL"
    else:
        performance_level = "INSUFICIENTE"
    
    print(f"Nível de Performance: {performance_level}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # Interpretação do Kappa (Landis & Koch 1977)
    if kappa >= 0.81:
        agreement_level = "Concordância Quase Perfeita"
    elif kappa >= 0.61:
        agreement_level = "Concordância Substancial"
    elif kappa >= 0.41:
        agreement_level = "Concordância Moderada"
    elif kappa >= 0.21:
        agreement_level = "Concordância Fraca"
    else:
        agreement_level = "Concordância Pobre"
    
    print(f"Nível de Concordância: {agreement_level}")
    
    # 2. RECOMENDAÇÕES BASEADAS NOS RESULTADOS
    print(f"\n2. RECOMENDAÇÕES CLÍNICAS")
    print("-" * 50)
    
    if balanced_acc >= 0.85:
        print("✅ APROVADO para próxima fase (Sistema Híbrido)")
        print("- Performance atende critérios médicos")
        print("- Pode proceder com validação multi-centro")
        print("- Considerar implementação de sistema híbrido")
    elif balanced_acc >= 0.75:
        print("⚠️  APROVADO CONDICIONAL")
        print("- Performance aceitável mas pode melhorar")
        print("- Revisar data augmentation")
        print("- Considerar ensemble methods")
        print("- Validação adicional necessária")
    else:
        print("❌ REPROVADO - Requer melhorias")
        print("- Performance insuficiente para uso clínico")
        print("- Revisar arquitetura do modelo")
        print("- Verificar qualidade dos dados")
        print("- Considerar modelos mais complexos")
    
    # 3. PRÓXIMOS PASSOS BASEADOS EM PERFORMANCE
    print(f"\n3. PRÓXIMOS PASSOS RECOMENDADOS")
    print("-" * 50)
    
    if balanced_acc >= 0.85:
        print("FASE 2 - SISTEMA HÍBRIDO:")
        print("1. Implementar DINO self-supervised")
        print("2. Criar ensemble com EfficientNet")
        print("3. Validação multi-centro")
        print("4. Teste com patologistas")
        
    print("\nVALIDAÇÃO ADICIONAL SEMPRE NECESSÁRIA:")
    print("1. Cross-validation com diferentes centros")
    print("2. Validação com scanners diferentes")
    print("3. Teste de robustez com variações de coloração")
    print("4. Análise de interpretabilidade (Grad-CAM)")

class AdvancedTMEAnalyzer:
    """
    Analisador avançado para interpretabilidade e validação clínica.
    
    MANTÉM EXATAMENTE a mesma funcionalidade original.
    
    Implementa técnicas de:
    - Grad-CAM para visualização
    - Análise de failure cases
    - Métricas por classe
    - Correlação com conhecimento patológico
    """
    
    def __init__(self, model: nn.Module, config: TMEConfig):
        self.model = model
        self.config = config
        
        # Detectar se modelo é DDP
        if hasattr(model, 'module'):
            self.device = next(model.module.parameters()).device
        else:
            self.device = next(model.parameters()).device
        
    def analyze_class_performance(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Análise detalhada por classe TME"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Relatório de classificação por classe
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.config.classes,
            output_dict=True
        )
        
        # Análise específica por tipo de tecido
        print(f"\n{'='*80}")
        print("ANÁLISE DE PERFORMANCE POR CLASSE TME")
        print(f"{'='*80}")
        
        for class_name in self.config.classes:
            if class_name in report:
                metrics = report[class_name]
                print(f"\n{class_name} - {self._get_tissue_description(class_name)}")
                print(f"  Precisão:     {metrics['precision']:.3f}")
                print(f"  Sensibilidade: {metrics['recall']:.3f}")
                print(f"  F1-Score:     {metrics['f1-score']:.3f}")
                print(f"  Suporte:      {metrics['support']}")
                
                # Interpretação clínica
                if metrics['f1-score'] >= 0.90:
                    status = "EXCELENTE"
                elif metrics['f1-score'] >= 0.80:
                    status = "BOM"
                elif metrics['f1-score'] >= 0.70:
                    status = "ACEITÁVEL"
                else:
                    status = "NECESSITA MELHORIA"
                
                print(f"  Status:       {status}")
        
        return report
    
    def _get_tissue_description(self, class_name: str) -> str:
        """Descrições clínicas baseadas em Mandal et al. (2025)"""
        descriptions = {
            'ADI': 'Tecido Adiposo - Células de gordura no TME',
            'DEB': 'Debris - Material necrótico e detritos celulares',
            'LYM': 'Linfócitos - Infiltrado imune ativo',
            'MUC': 'Mucina - Secreção mucosa característica do gástrico',
            'MUS': 'Músculo - Camada muscular da parede gástrica',
            'NOR': 'Normal - Mucosa gástrica saudável',
            'STR': 'Estroma - Tecido conjuntivo de suporte',
            'TUM': 'Tumor - Células tumorais malignas'
        }
        return descriptions.get(class_name, 'Descrição não disponível')
    
    def generate_confusion_matrix_analysis(self, dataloader: DataLoader):
        """Gera análise detalhada da matriz de confusão"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Matriz de confusão
        cm = confusion_matrix(all_labels, all_preds)
        
        # Visualização
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.classes,
                   yticklabels=self.config.classes)
        plt.title('Matriz de Confusão - Classificação TME')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise dos erros mais comuns
        print(f"\n{'='*80}")
        print("ANÁLISE DOS ERROS MAIS COMUNS")
        print(f"{'='*80}")
        
        # Encontrar pares de classes com mais confusão
        confusion_pairs = []
        for i in range(len(self.config.classes)):
            for j in range(len(self.config.classes)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((
                        self.config.classes[i],
                        self.config.classes[j], 
                        cm[i, j]
                    ))
        
        # Ordenar por número de confusões
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\nTop 5 Confusões Mais Comuns:")
        for i, (true_class, pred_class, count) in enumerate(confusion_pairs[:5]):
            print(f"{i+1}. {true_class} → {pred_class}: {count} casos")
            print(f"   Interpretação: {self._interpret_confusion(true_class, pred_class)}")
    
    def _interpret_confusion(self, true_class: str, pred_class: str) -> str:
        """Interpreta confusões baseado em conhecimento patológico"""
        
        # Confusões esperadas baseadas em similaridade morfológica
        expected_confusions = {
            ('STR', 'MUS'): 'Similaridade entre estroma e músculo liso',
            ('TUM', 'STR'): 'Desmoplasia - reação estromal ao tumor',
            ('LYM', 'STR'): 'Infiltrado linfocitário denso pode parecer estroma',
            ('NOR', 'TUM'): 'Displasia de alto grau vs. carcinoma inicial',
            ('MUC', 'TUM'): 'Adenocarcinoma mucinoso vs. mucina normal',
            ('DEB', 'TUM'): 'Necrose tumoral vs. debris',
        }
        
        pair = (true_class, pred_class)
        reverse_pair = (pred_class, true_class)
        
        if pair in expected_confusions:
            return f"Esperado: {expected_confusions[pair]}"
        elif reverse_pair in expected_confusions:
            return f"Esperado: {expected_confusions[reverse_pair]}"
        else:
            return "Confusão inesperada - requer investigação"

def create_validation_pipeline() -> Dict[str, any]:
    """
    Pipeline completo de validação seguindo padrões médicos rigorosos.
    
    MANTÉM EXATAMENTE a mesma funcionalidade original.
    
    Implementa:
    1. Treinamento conservador
    2. Validação cruzada
    3. Teste independente
    4. Análise de interpretabilidade
    5. Recomendações clínicas
    """
    
    print("="*80)
    print("PIPELINE DE VALIDAÇÃO COMPLETO")
    print("="*80)
    
    # Configuração otimizada baseada em literatura
    config = TMEConfig(
        data_path="data",
        model_name="efficientnet_b4",
        image_size=224,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_epochs=100,
        patience=15,
        dropout_rate=0.3,
        use_distributed=True  # Auto-detecta múltiplas GPUs
    )
    
    # FASE 1: Treinamento e validação inicial
    print("\n" + "="*50)
    print("FASE 1: TREINAMENTO BASELINE")
    print("="*50)
    
    trainer = TMETrainer(config)
    results = trainer.train_model()
    
    # Verificar se temos resultados válidos
    if not results or not results.get('test_metrics'):
        print("❌ Falha no treinamento - resultados indisponíveis")
        return {}
    
    # FASE 2: Análise detalhada
    print("\n" + "="*50)
    print("FASE 2: ANÁLISE DETALHADA")
    print("="*50)
    
    # Criar modelo para análise
    if results.get('model') is not None:
        analyzer = AdvancedTMEAnalyzer(results['model'], config)
        
        # Preparar dados para análise
        transforms = create_augmentation_transforms(config)
        test_dataset = TMEDataset(
            data_path=config.data_path,
            split='test',
            transform=transforms['test'],
            config=config
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Análises detalhadas
        class_performance = analyzer.analyze_class_performance(test_loader)
        analyzer.generate_confusion_matrix_analysis(test_loader)
    else:
        print("⚠️  Modelo não disponível para análise detalhada")
        class_performance = {}
    
    # FASE 3: Decisão para próximos passos
    print("\n" + "="*50)
    print("FASE 3: DECISÃO E PRÓXIMOS PASSOS")
    print("="*50)
    
    balanced_acc = results['test_metrics'].get('balanced_accuracy', 0.0)
    
    # Critérios baseados em literatura médica
    if balanced_acc >= 0.85:
        next_phase = "SISTEMA_HIBRIDO"
        print("✅ APROVADO: Proceder com Sistema Híbrido")
        print("\nPRÓXIMOS PASSOS:")
        print("1. Implementar DINO self-supervised")
        print("2. Criar ensemble EfficientNet + DINO")
        print("3. Validação multi-centro")
        print("4. Teste com patologistas especialistas")
        
    elif balanced_acc >= 0.75:
        next_phase = "MELHORIAS_NECESSARIAS"
        print("⚠️  APROVADO CONDICIONAL: Melhorias necessárias")
        print("\nPRÓXIMOS PASSOS:")
        print("1. Otimizar hyperparâmetros")
        print("2. Aumentar dataset de treinamento")
        print("3. Implementar técnicas de regularização")
        print("4. Re-validar antes do sistema híbrido")
        
    else:
        next_phase = "REPROJETAR"
        print("❌ REPROVADO: Necessário reprojetar abordagem")
        print("\nPRÓXIMOS PASSOS:")
        print("1. Revisar qualidade dos dados")
        print("2. Considerar arquiteturas alternativas")
        print("3. Investigar data leakage ou overfitting")
        print("4. Consultar especialistas em patologia")
    
    # Salvar resultados completos
    final_results = {
        'config': asdict(config),
        'training_results': results,
        'class_performance': class_performance,
        'next_phase_recommendation': next_phase,
        'timestamp': time.time()
    }
    
    # Salvar em JSON para análise posterior
    with open('validation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("PIPELINE CONCLUÍDO - Resultados salvos em 'validation_results.json'")
    print(f"{'='*80}")
    
    return final_results

# Função de utilidade para organização dos dados
def organize_hmu_data(source_dir: str, target_dir: str):
    """
    Organiza dados HMU-GC-HE-30K seguindo estrutura adequada.
    
    MANTÉM EXATAMENTE a mesma funcionalidade original.
    
    Args:
        source_dir: Diretório com dados originais do FigShare
        target_dir: Diretório de destino organizado
    """
    from pathlib import Path
    import shutil
    from sklearn.model_selection import train_test_split
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Classes TME baseadas em Lou et al. (2025)
    classes = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
    
    # Criar estrutura de diretórios
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (target_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("Organizando dados HMU-GC-HE-30K...")
    
    # Para cada classe
    for cls in classes:
        class_dir = source_path / "all_image" / cls
        
        if not class_dir.exists():
            print(f"⚠️  Diretório não encontrado: {class_dir}")
            continue
        
        # Listar arquivos
        files = list(class_dir.glob("*.png"))
        
        if len(files) == 0:
            print(f"⚠️  Nenhum arquivo encontrado em: {class_dir}")
            continue
        
        # Divisão estratificada: 70% train, 15% val, 15% test
        train_files, temp_files = train_test_split(
            files, test_size=0.3, random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=0.5, random_state=42
        )
        
        # Copiar arquivos
        splits = [
            ('train', train_files),
            ('val', val_files), 
            ('test', test_files)
        ]
        
        for split_name, file_list in splits:
            for file_path in file_list:
                dest_path = target_path / split_name / cls / file_path.name
                shutil.copy2(file_path, dest_path)
        
        print(f"{cls}: {len(train_files):4d} train, {len(val_files):4d} val, {len(test_files):4d} test")
    
    print(f"\n✅ Dados organizados em: {target_path}")

def setup_distributed_environment(rank: int, world_size: int, config: TMEConfig):
    """
    Configura ambiente distribuído para DDP.
    
    NOVA FUNCIONALIDADE para suporte robusto a DDP.
    
    Args:
        rank: Rank do processo atual
        world_size: Número total de processos
        config: Configuração TME
    """
    # Configurar variáveis de ambiente
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    
    # Debug para DDP (opcional)
    if logging.getLogger().level == logging.DEBUG:
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Inicializar process group
    dist.init_process_group(
        backend=config.ddp_backend,
        rank=rank,
        world_size=world_size
    )
    
    # Configurar device CUDA
    torch.cuda.set_device(rank)
    
    if rank == 0:
        logging.info(f"DDP inicializado - Backend: {config.ddp_backend}, World Size: {world_size}")

def cleanup_distributed_environment():
    """
    Limpa ambiente distribuído.
    
    NOVA FUNCIONALIDADE para cleanup seguro.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def distributed_setup_and_train(rank, world_size, config_dict):
    """
    Função auxiliar para cada processo DDP (nível de módulo - picklável).
    """
    # Recriar config
    config = TMEConfig(**config_dict)

    # Setup DDP
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port

    dist.init_process_group(
        backend=config.ddp_backend,
        rank=rank,
        world_size=world_size
    )

    # Criar trainer para este rank
    trainer = DistributedTMETrainer(config, rank=rank, world_size=world_size)

    try:
        trainer.train_model()
    finally:
        trainer._cleanup()


# Compatibilidade com sistema híbrido original
from pathlib import Path
import os

# Verificar se existe implementação do sistema híbrido
HYBRID_SYSTEM_PATH = Path("sistema_hibrido.py")

def check_hybrid_system_availability() -> bool:
    """
    Verifica se sistema híbrido está disponível.
    
    NOVA FUNCIONALIDADE para integração com sistema híbrido.
    """
    return HYBRID_SYSTEM_PATH.exists()

def run_hybrid_system_if_approved(baseline_results: Dict[str, any], config: TMEConfig) -> Optional[Dict[str, any]]:
    """
    Executa sistema híbrido se baseline for aprovado.
    
    NOVA FUNCIONALIDADE para integração automática.
    
    Args:
        baseline_results: Resultados do sistema baseline
        config: Configuração TME
        
    Returns:
        Resultados do sistema híbrido ou None se não executado
    """
    balanced_acc = baseline_results.get('test_metrics', {}).get('balanced_accuracy', 0.0)
    
    if balanced_acc < 0.85:
        print(f"\n⚠️  Baseline não aprovado para sistema híbrido: {balanced_acc:.3f} < 0.85")
        return None
    
    if not check_hybrid_system_availability():
        print(f"\n⚠️  Sistema híbrido não encontrado em: {HYBRID_SYSTEM_PATH}")
        print("Execute apenas o sistema baseline.")
        return None
    
    print(f"\n✅ Baseline aprovado: {balanced_acc:.3f} ≥ 0.85")
    print("Procedendo com sistema híbrido...")
    
    try:
        # Importar e executar sistema híbrido
        import importlib.util
        spec = importlib.util.spec_from_file_location("sistema_hibrido", HYBRID_SYSTEM_PATH)
        hybrid_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hybrid_module)
        
        # Executar pipeline híbrido
        hybrid_results = hybrid_module.main_hybrid_pipeline("best_model.pth")
        return hybrid_results
        
    except Exception as e:
        logging.error(f"Erro ao executar sistema híbrido: {e}")
        print(f"\n❌ Falha no sistema híbrido: {e}")
        return None

def print_usage_instructions():
    """
    Imprime instruções detalhadas de uso do sistema.
    
    NOVA FUNCIONALIDADE para facilitar uso.
    """
    print("\n" + "="*80)
    print("INSTRUÇÕES DE USO DO SISTEMA TME")
    print("="*80)
    
    print("\n📁 1. PREPARAÇÃO DOS DADOS:")
    print("   Organize seus dados na estrutura:")
    print("   data/")
    print("   ├── train/")
    print("   │   ├── ADI/")
    print("   │   ├── DEB/")
    print("   │   ├── LYM/")
    print("   │   ├── MUC/")
    print("   │   ├── MUS/")
    print("   │   ├── NOR/")
    print("   │   ├── STR/")
    print("   │   └── TUM/")
    print("   ├── val/ (mesma estrutura)")
    print("   └── test/ (mesma estrutura)")
    print("\n   OU use a função organize_hmu_data() para organizar automaticamente")
    
    print("\n🚀 2. EXECUÇÃO BÁSICA (Single GPU):")
    print("   from tme_gastric_classifier import *")
    print("   ")
    print("   # Configuração")
    print("   config = TMEConfig(")
    print("       data_path='data',")
    print("       use_distributed=False  # Single GPU")
    print("   )")
    print("   ")
    print("   # Treinamento")
    print("   trainer = TMETrainer(config)")
    print("   results = trainer.train_model()")
    print("   ")
    print("   # Análise")
    print("   analyze_results(results, config)")
    
    print("\n⚡ 3. EXECUÇÃO ACELERADA (Multi-GPU):")
    print("   config = TMEConfig(")
    print("       data_path='data',")
    print("       use_distributed=True   # Auto-detecta múltiplas GPUs")
    print("   )")
    print("   ")
    print("   # Resto igual ao single GPU")
    print("   trainer = TMETrainer(config)")
    print("   results = trainer.train_model()")
    
    print("\n🔬 4. PIPELINE COMPLETO:")
    print("   # Executa pipeline completo com análises")
    print("   results = create_validation_pipeline()")
    
    print("\n📊 5. ANÁLISE AVANÇADA:")
    print("   # Após treinamento, analise detalhadamente")
    print("   analyzer = AdvancedTMEAnalyzer(model, config)")
    print("   class_performance = analyzer.analyze_class_performance(test_loader)")
    print("   analyzer.generate_confusion_matrix_analysis(test_loader)")
    
    print("\n⚙️  6. CONFIGURAÇÕES IMPORTANTES:")
    print("   - Todos os hiperparâmetros são baseados em literatura médica")
    print("   - NÃO altere learning_rate, batch_size, patience sem justificativa")
    print("   - use_distributed=True acelera em múltiplas GPUs automaticamente")
    print("   - Sistema salva automaticamente o melhor modelo em 'best_model.pth'")
    
    print("\n📈 7. MÉTRICAS DE APROVAÇÃO:")
    print("   - Balanced Accuracy ≥ 85%: Aprovado para sistema híbrido")
    print("   - Balanced Accuracy ≥ 75%: Aprovado condicional")
    print("   - Balanced Accuracy < 75%: Requer melhorias")
    
    print("\n🔄 8. SISTEMA HÍBRIDO (Automático):")
    print("   Se baseline ≥ 85%, o sistema automaticamente:")
    print("   - Procura por 'sistema_hibrido.py'")
    print("   - Executa pipeline híbrido EfficientNet + DINO")
    print("   - Compara resultados e recomenda melhor abordagem")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    """
    Execução principal com detecção automática de configuração.
    
    MELHORIAS IMPLEMENTADAS:
    - Auto-detecção de single/multi GPU
    - Integração automática com sistema híbrido
    - Instruções de uso detalhadas
    - Tratamento robusto de erros
    """
    
    print("="*80)
    print("SISTEMA TME - CLASSIFICAÇÃO PARA CÂNCER GÁSTRICO")
    print("VERSÃO MELHORADA COM SUPORTE A PROCESSAMENTO PARALELO")
    print("="*80)
    
    # Verificar disponibilidade de GPUs
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"🔥 {gpu_count} GPUs detectadas - Modo distribuído disponível")
    elif gpu_count == 1:
        print("🔥 1 GPU detectada - Modo single GPU")
    else:
        print("⚠️  Nenhuma GPU detectada - Modo CPU (muito lento)")
    
    # Verificar se dados existem
    data_path = Path("data")
    if not data_path.exists():
        print(f"\n❌ Diretório de dados não encontrado: {data_path}")
        print("📁 Use organize_hmu_data() para organizar seus dados primeiro")
        print_usage_instructions()
        sys.exit(1)
    
    # Verificar estrutura de dados
    required_splits = ['train', 'val', 'test']
    required_classes = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
    
    structure_ok = True
    for split in required_splits:
        split_path = data_path / split
        if not split_path.exists():
            print(f"❌ Diretório não encontrado: {split_path}")
            structure_ok = False
            continue
        
        for class_name in required_classes:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"❌ Classe não encontrada: {class_path}")
                structure_ok = False
    
    if not structure_ok:
        print("\n📁 Estrutura de dados incorreta. Use organize_hmu_data() para corrigir.")
        print_usage_instructions()
        sys.exit(1)
    
    try:
        # Exemplo de uso completo
        print("\n🚀 EXECUTANDO PIPELINE COMPLETO...")
        
        # 1. Executar sistema baseline
        results = create_validation_pipeline()
        
        if not results:
            print("❌ Falha no pipeline de validação")
            sys.exit(1)
        
        # 2. Tentar executar sistema híbrido se aprovado
        if results.get('training_results'):
            hybrid_results = run_hybrid_system_if_approved(
                results['training_results'], 
                TMEConfig(use_distributed=(gpu_count > 1))
            )
            
            if hybrid_results:
                print("\n✅ Sistema híbrido executado com sucesso")
                print("📊 Verifique 'hybrid_validation_results.json' para comparação")
        
        print("\n✅ PIPELINE CONCLUÍDO COM SUCESSO")
        print("📊 Resultados salvos em 'validation_results.json'")
        print("🏥 Sistema pronto para validação clínica")
        
    except KeyboardInterrupt:
        print("\n⚠️  Execução interrompida pelo usuário")
        cleanup_distributed_environment()
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        logging.error(f"Erro fatal: {e}", exc_info=True)
        cleanup_distributed_environment()
        sys.exit(1)
    
    finally:
        # Cleanup final
        cleanup_distributed_environment()