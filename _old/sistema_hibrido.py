#!/usr/bin/env python3
"""
Sistema Híbrido TME - Fase 2
===========================

Sistema híbrido DINO + EfficientNet para classificação TME em câncer gástrico.
Implementado apenas se baseline EfficientNet atingir ≥85% balanced accuracy.

Baseado em:
- Caron et al. (2021): DINO - Self-supervised Vision Transformers
- Tan & Le (2019): EfficientNet
- Chen et al. (2022): Multi-modal fusion for medical imaging
- Tellez et al. (2019): Self-supervised learning in histopathology
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm

# Importar componentes do sistema baseline
from tme_gastric_classifier import TMEConfig, TMEDataset, create_augmentation_transforms

logging.basicConfig(level=logging.INFO)

@dataclass
class HybridConfig(TMEConfig):
    """
    Configuração estendida para sistema híbrido.
    
    Adiciona parâmetros específicos para:
    - DINO self-supervised learning
    - Ensemble methods
    - Multi-modal fusion
    """
    
    # DINO configuration
    dino_model: str = "dino_vits16"  # DINO ViT-Small/16
    dino_pretrained: bool = True
    dino_feature_dim: int = 384      # ViT-S/16 feature dimension
    
    # Ensemble configuration
    ensemble_method: str = "weighted_average"  # "weighted_average", "stacking", "voting"
    efficientnet_weight: float = 0.6           # Weight for EfficientNet predictions
    dino_weight: float = 0.4                   # Weight for DINO predictions
    
    # Fusion strategy
    fusion_method: str = "feature_fusion"      # "feature_fusion", "prediction_fusion"
    fusion_hidden_dim: int = 512               # Hidden dimension for fusion layer
    
    # Self-supervised pre-training (optional)
    ssl_epochs: int = 50                       # Self-supervised pre-training epochs
    ssl_learning_rate: float = 1e-4            # SSL learning rate
    use_ssl_pretraining: bool = False          # Whether to do SSL pre-training
    
    # Advanced training
    use_mixup: bool = True                     # Mixup augmentation
    mixup_alpha: float = 0.2                   # Mixup alpha parameter
    use_cutmix: bool = True                    # CutMix augmentation
    cutmix_alpha: float = 1.0                  # CutMix alpha parameter

class DINOFeatureExtractor(nn.Module):
    """
    DINO Vision Transformer feature extractor.
    
    Baseado em:
    - Caron et al. (2021): DINO self-supervised learning
    - Dosovitskiy et al. (2021): Vision Transformer
    - Tellez et al. (2019): Self-supervised histopathology
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Carregar DINO pré-treinado
        if config.dino_model == "dino_vits16":
            # DINO ViT-Small/16 pre-trained on ImageNet
            self.backbone = timm.create_model(
                'vit_small_patch16_224.dino',
                pretrained=config.dino_pretrained,
                num_classes=0  # Remove classification head
            )
            self.feature_dim = 384
            
        elif config.dino_model == "dino_vitb16":
            # DINO ViT-Base/16 (mais expressivo, mais caro)
            self.backbone = timm.create_model(
                'vit_base_patch16_224.dino',
                pretrained=config.dino_pretrained,
                num_classes=0
            )
            self.feature_dim = 768
        else:
            raise ValueError(f"DINO model não suportado: {config.dino_model}")
        
        # Projection head para TME-specific features
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Classificador TME
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_hidden_dim, config.num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicialização Xavier para novas camadas"""
        for module in [self.projection_head, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # Extract DINO features
        dino_features = self.backbone(x)  # [B, feature_dim]
        
        # Project to TME-specific space
        projected_features = self.projection_head(dino_features)  # [B, fusion_hidden_dim]
        
        if return_features:
            return projected_features
        
        # Classify
        logits = self.classifier(projected_features)  # [B, num_classes]
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai attention maps do DINO para interpretabilidade.
        
        Baseado em Caron et al. (2021): "DINO learns semantic segmentation"
        """
        # Get attention weights from last transformer block
        attention_weights = self.backbone.blocks[-1].attn.get_attention_map(x)
        return attention_weights

class EfficientNetDINOFusion(nn.Module):
    """
    Sistema híbrido que combina EfficientNet e DINO.
    
    Estratégias de fusão:
    1. Feature-level fusion: Combina features antes da classificação
    2. Prediction-level fusion: Combina predições dos dois modelos
    
    Baseado em:
    - Chen et al. (2022): Multi-modal fusion strategies
    - Huang & Wang (2017): Fusion architectures
    """
    
    def __init__(self, config: HybridConfig, efficientnet_checkpoint: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # EfficientNet branch (carregado do baseline)
        self.efficientnet = self._load_efficientnet(efficientnet_checkpoint)
        
        # DINO branch
        self.dino = DINOFeatureExtractor(config)
        
        # Fusion strategy
        if config.fusion_method == "feature_fusion":
            self._setup_feature_fusion()
        elif config.fusion_method == "prediction_fusion":
            self._setup_prediction_fusion()
        else:
            raise ValueError(f"Fusion method não suportado: {config.fusion_method}")
    
    def _load_efficientnet(self, checkpoint_path: Optional[str]) -> nn.Module:
        """Carrega EfficientNet treinado do baseline"""
        from tme_gastric_classifier import EfficientNetTME
        
        # Criar modelo EfficientNet
        efficientnet = EfficientNetTME(self.config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Carregar pesos do baseline treinado
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            efficientnet.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"EfficientNet carregado de: {checkpoint_path}")
            
            # Congelar EfficientNet inicialmente (fine-tuning conservador)
            for param in efficientnet.parameters():
                param.requires_grad = False
        
        return efficientnet
    
    def _setup_feature_fusion(self):
        """
        Configura fusão em nível de features.
        
        Features do EfficientNet + DINO → Fusion layer → Classification
        """
        # Dimensões das features
        efficientnet_dim = 1792  # EfficientNet-B4 features
        dino_dim = self.config.fusion_hidden_dim  # DINO projected features
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(efficientnet_dim + dino_dim, self.config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.fusion_hidden_dim, self.config.fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate / 2)
        )
        
        # Final classifier
        self.final_classifier = nn.Linear(
            self.config.fusion_hidden_dim // 2, 
            self.config.num_classes
        )
        
        self._init_fusion_weights()
    
    def _setup_prediction_fusion(self):
        """
        Configura fusão em nível de predições.
        
        EfficientNet predictions + DINO predictions → Weighted combination
        """
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(
            torch.tensor([self.config.efficientnet_weight, self.config.dino_weight])
        )
        
        # Optional meta-classifier for prediction fusion
        self.meta_classifier = nn.Sequential(
            nn.Linear(self.config.num_classes * 2, self.config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.fusion_hidden_dim, self.config.num_classes)
        )
        
        self._init_fusion_weights()
    
    def _init_fusion_weights(self):
        """Inicializa pesos das camadas de fusão"""
        for module in [getattr(self, 'fusion_layer', None), 
                      getattr(self, 'final_classifier', None),
                      getattr(self, 'meta_classifier', None)]:
            if module is not None:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor, return_individual: bool = False) -> torch.Tensor:
        """
        Forward pass do sistema híbrido.
        
        Args:
            x: Input images [B, 3, H, W]
            return_individual: Se retorna predições individuais
        """
        if self.config.fusion_method == "feature_fusion":
            return self._forward_feature_fusion(x, return_individual)
        else:
            return self._forward_prediction_fusion(x, return_individual)
    
    def _forward_feature_fusion(self, x: torch.Tensor, return_individual: bool = False):
        """Feature-level fusion forward pass"""
        # EfficientNet features
        efficientnet_features = self.efficientnet.get_features(x)  # [B, 1792]
        
        # DINO features
        dino_features = self.dino(x, return_features=True)  # [B, fusion_hidden_dim]
        
        # Concatenate features
        fused_features = torch.cat([efficientnet_features, dino_features], dim=1)
        
        # Fusion processing
        fused_features = self.fusion_layer(fused_features)
        
        # Final classification
        logits = self.final_classifier(fused_features)
        
        if return_individual:
            # Return individual predictions for analysis
            efficientnet_logits = self.efficientnet(x)
            dino_logits = self.dino(x)
            return {
                'fused': logits,
                'efficientnet': efficientnet_logits,
                'dino': dino_logits
            }
        
        return logits
    
    def _forward_prediction_fusion(self, x: torch.Tensor, return_individual: bool = False):
        """Prediction-level fusion forward pass"""
        # Individual predictions
        efficientnet_logits = self.efficientnet(x)  # [B, num_classes]
        dino_logits = self.dino(x)  # [B, num_classes]
        
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted combination
        fused_logits = weights[0] * efficientnet_logits + weights[1] * dino_logits
        
        # Optional meta-classifier
        if hasattr(self, 'meta_classifier'):
            # Concatenate predictions for meta-learning
            concat_preds = torch.cat([efficientnet_logits, dino_logits], dim=1)
            meta_logits = self.meta_classifier(concat_preds)
            fused_logits = 0.7 * fused_logits + 0.3 * meta_logits
        
        if return_individual:
            return {
                'fused': fused_logits,
                'efficientnet': efficientnet_logits,
                'dino': dino_logits,
                'weights': weights
            }
        
        return fused_logits

class MixupCutmixAugmentation:
    """
    Implementa Mixup e CutMix para augmentation avançada.
    
    Baseado em:
    - Zhang et al. (2018): Mixup - Beyond Empirical Risk Minimization
    - Yun et al. (2019): CutMix - Regularization Strategy
    - Adaptado para histopatologia médica
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.mixup_alpha = config.mixup_alpha
        self.cutmix_alpha = config.cutmix_alpha
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Aplica Mixup augmentation.
        
        Returns:
            mixed_x: Imagens mixadas
            y_a, y_b: Labels originais  
            lam: Lambda parameter
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Aplica CutMix augmentation.
        
        Corta região retangular e cola de outra imagem.
        """
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Calcular bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

class HybridTMETrainer:
    """
    Trainer para sistema híbrido com funcionalidades avançadas.
    
    Inclui:
    - Progressive unfreezing strategy
    - Advanced augmentation (Mixup/CutMix)
    - Multi-component loss functions
    - Ensemble training and validation
    
    Baseado em:
    - Howard & Ruder (2018): Universal Language Model Fine-tuning
    - Smith & Topin (2019): Super-convergence training
    - Chen et al. (2022): Multi-modal medical imaging
    """
    
    def __init__(self, config: HybridConfig, efficientnet_checkpoint: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.efficientnet_checkpoint = efficientnet_checkpoint
        
        # Advanced augmentation
        if config.use_mixup or config.use_cutmix:
            self.augmenter = MixupCutmixAugmentation(config)
        
        logging.info(f"Hybrid trainer initialized - Device: {self.device}")
    
    def create_hybrid_model(self) -> nn.Module:
        """Cria e configura modelo híbrido"""
        model = EfficientNetDINOFusion(
            config=self.config,
            efficientnet_checkpoint=self.efficientnet_checkpoint
        )
        
        return model.to(self.device)
    
    def create_progressive_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, object]:
        """
        Cria otimizador com learning rates diferenciados.
        
        Estratégia progressive unfreezing:
        1. Frozen EfficientNet + Active DINO + Active Fusion
        2. Active EfficientNet (low LR) + Active DINO + Active Fusion  
        3. Active all components
        """
        
        # Learning rates diferenciados
        param_groups = [
            # EfficientNet parameters (frozen initially)
            {
                'params': model.efficientnet.parameters(),
                'lr': self.config.learning_rate * 0.01,  # Very low initially
                'name': 'efficientnet'
            },
            # DINO parameters  
            {
                'params': model.dino.backbone.parameters(),
                'lr': self.config.learning_rate * 0.1,   # Lower for pre-trained
                'name': 'dino_backbone'
            },
            # DINO projection + classifier
            {
                'params': list(model.dino.projection_head.parameters()) + 
                         list(model.dino.classifier.parameters()),
                'lr': self.config.learning_rate,
                'name': 'dino_head'
            }
        ]
        
        # Fusion parameters
        if hasattr(model, 'fusion_layer'):
            param_groups.append({
                'params': list(model.fusion_layer.parameters()) + 
                         list(model.final_classifier.parameters()),
                'lr': self.config.learning_rate,
                'name': 'fusion'
            })
        elif hasattr(model, 'meta_classifier'):
            param_groups.append({
                'params': model.meta_classifier.parameters(),
                'lr': self.config.learning_rate,
                'name': 'meta_fusion'
            })
        
        # AdamW optimizer
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,      # Longer cycles for hybrid training
            T_mult=2,
            eta_min=1e-7
        )
        
        return optimizer, scheduler
    
    def progressive_unfreeze(self, model: nn.Module, epoch: int):
        """
        Implementa progressive unfreezing do EfficientNet.
        
        Baseado em Howard & Ruder (2018):
        - Epochs 0-20: EfficientNet frozen
        - Epochs 21-40: EfficientNet partially unfrozen  
        - Epochs 41+: All parameters active
        """
        
        if epoch == 20:
            logging.info("Progressive unfreezing: Activating EfficientNet")
            for param in model.efficientnet.parameters():
                param.requires_grad = True
                
        elif epoch == 40:
            logging.info("Progressive unfreezing: Full model active")
            # Increase EfficientNet learning rate
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'efficientnet':
                    param_group['lr'] = self.config.learning_rate * 0.1
    
    def mixup_cutmix_loss(self, pred: torch.Tensor, y_a: torch.Tensor, 
                         y_b: torch.Tensor, lam: float, criterion: nn.Module) -> torch.Tensor:
        """
        Calcula loss para Mixup/CutMix.
        
        loss = λ * loss(pred, y_a) + (1-λ) * loss(pred, y_b)
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train_epoch_hybrid(self, model: nn.Module, train_loader: DataLoader,
                          optimizer: optim.Optimizer, criterion: nn.Module, 
                          epoch: int) -> Dict[str, float]:
        """
        Treina uma época do modelo híbrido com augmentation avançada.
        """
        model.train()
        
        # Progressive unfreezing
        self.progressive_unfreeze(model, epoch)
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Componentes do loss
        loss_components = defaultdict(float)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Advanced augmentation
            if self.config.use_mixup and np.random.random() < 0.5:
                inputs, y_a, y_b, lam = self.augmenter.mixup_data(inputs, labels)
                use_mixup = True
            elif self.config.use_cutmix and np.random.random() < 0.5:
                inputs, y_a, y_b, lam = self.augmenter.cutmix_data(inputs, labels)
                use_mixup = True
            else:
                use_mixup = False
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, EfficientNetDINOFusion):
                outputs = model(inputs, return_individual=True)
                main_outputs = outputs['fused']
                
                # Multi-component loss
                if use_mixup:
                    # Main loss (fused)
                    main_loss = self.mixup_cutmix_loss(
                        main_outputs, y_a, y_b, lam, criterion
                    )
                    
                    # Auxiliary losses (individual models)
                    efficientnet_loss = self.mixup_cutmix_loss(
                        outputs['efficientnet'], y_a, y_b, lam, criterion
                    )
                    dino_loss = self.mixup_cutmix_loss(
                        outputs['dino'], y_a, y_b, lam, criterion
                    )
                else:
                    main_loss = criterion(main_outputs, labels)
                    efficientnet_loss = criterion(outputs['efficientnet'], labels)
                    dino_loss = criterion(outputs['dino'], labels)
                
                # Combined loss with auxiliary supervision
                total_loss = (
                    0.7 * main_loss +           # Primary fusion loss
                    0.15 * efficientnet_loss +  # Auxiliary EfficientNet loss
                    0.15 * dino_loss           # Auxiliary DINO loss
                )
                
                # Track loss components
                loss_components['main'] += main_loss.item()
                loss_components['efficientnet'] += efficientnet_loss.item()
                loss_components['dino'] += dino_loss.item()
                
            else:
                # Fallback for single model
                main_outputs = model(inputs)
                if use_mixup:
                    total_loss = self.mixup_cutmix_loss(
                        main_outputs, y_a, y_b, lam, criterion
                    )
                else:
                    total_loss = criterion(main_outputs, labels)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item() * inputs.size(0)
            
            if not use_mixup:
                _, preds = torch.max(main_outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
            else:
                # Approximate accuracy for mixup
                _, preds = torch.max(main_outputs, 1)
                running_corrects += torch.sum(
                    lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()
                )
            
            total_samples += inputs.size(0)
            
            # Log progresso
            if batch_idx % 50 == 0:
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Loss={current_loss:.4f}, Acc={current_acc:.4f}")
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        # Log componentes do loss
        for component, value in loss_components.items():
            loss_components[component] = value / len(train_loader)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc.item(),
            'loss_components': dict(loss_components)
        }
    
    def validate_epoch_hybrid(self, model: nn.Module, val_loader: DataLoader,
                             criterion: nn.Module) -> Dict[str, float]:
        """
        Valida modelo híbrido com análise detalhada de componentes.
        """
        model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Análise individual dos componentes
        individual_predictions = {
            'efficientnet': [],
            'dino': [],
            'fused': []
        }
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if isinstance(model, EfficientNetDINOFusion):
                    outputs = model(inputs, return_individual=True)
                    main_outputs = outputs['fused']
                    
                    # Store individual predictions for analysis
                    individual_predictions['efficientnet'].extend(
                        torch.max(outputs['efficientnet'], 1)[1].cpu().numpy()
                    )
                    individual_predictions['dino'].extend(
                        torch.max(outputs['dino'], 1)[1].cpu().numpy()
                    )
                    individual_predictions['fused'].extend(
                        torch.max(main_outputs, 1)[1].cpu().numpy()
                    )
                else:
                    main_outputs = model(inputs)
                
                loss = criterion(main_outputs, labels)
                
                # Probabilities and predictions
                probs = F.softmax(main_outputs, dim=1)
                _, preds = torch.max(main_outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calcular métricas
        metrics = self._calculate_medical_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = running_loss / len(val_loader.dataset)
        
        # Análise de componentes individuais
        if individual_predictions['efficientnet']:
            metrics['component_analysis'] = self._analyze_individual_components(
                all_labels, individual_predictions
            )
        
        return metrics
    
    def _calculate_medical_metrics(self, y_true: List[int], 
                                 y_pred: List[int], y_probs: List[List[float]]) -> Dict[str, float]:
        """Métricas médicas robustas (reutilizada do baseline)"""
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score,
            cohen_kappa_score, roc_auc_score
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except:
            metrics['auc'] = 0.0
        
        return metrics
    
    def _analyze_individual_components(self, y_true: List[int], 
                                     predictions: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
        """
        Analisa performance dos componentes individuais.
        
        Útil para entender:
        - Qual componente contribui mais
        - Se fusion está realmente melhorando
        - Complementaridade entre EfficientNet e DINO
        """
        component_metrics = {}
        
        for component_name, preds in predictions.items():
            if preds:
                component_metrics[component_name] = {
                    'accuracy': accuracy_score(y_true, preds),
                    'balanced_accuracy': balanced_accuracy_score(y_true, preds),
                    'f1_macro': f1_score(y_true, preds, average='macro'),
                    'kappa': cohen_kappa_score(y_true, preds)
                }
        
        return component_metrics
    
    def train_hybrid_model(self) -> Dict[str, any]:
        """
        Pipeline completo de treinamento híbrido.
        
        Fases:
        1. Preparação dos dados
        2. Criação do modelo híbrido
        3. Treinamento com progressive unfreezing
        4. Validação com análise de componentes
        5. Teste final e comparação com baseline
        """
        
        print(f"\n{'='*80}")
        print("TREINAMENTO DO SISTEMA HÍBRIDO TME")
        print(f"{'='*80}")
        print(f"EfficientNet checkpoint: {self.efficientnet_checkpoint}")
        print(f"DINO model: {self.config.dino_model}")
        print(f"Fusion method: {self.config.fusion_method}")
        print(f"{'='*80}")
        
        # 1. Preparar dados (reutiliza do baseline)
        from tme_gastric_classifier import TMETrainer
        base_trainer = TMETrainer(self.config)
        dataloaders = base_trainer.prepare_data()
        
        # 2. Criar modelo híbrido
        model = self.create_hybrid_model()
        
        # 3. Configurar otimização
        self.optimizer, scheduler = self.create_progressive_optimizer(model)
        
        # 4. Loss function
        train_dataset = dataloaders['train'].dataset
        class_weights = train_dataset.get_class_weights().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 5. Training loop
        best_balanced_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_balanced_acc': [], 'val_kappa': [],
            'component_analysis': []
        }
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch_hybrid(
                model, dataloaders['train'], self.optimizer, criterion, epoch
            )
            
            # Validation
            val_metrics = self.validate_epoch_hybrid(
                model, dataloaders['val'], criterion
            )
            
            # Scheduler step
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
            history['val_kappa'].append(val_metrics['kappa'])
            
            if 'component_analysis' in val_metrics:
                history['component_analysis'].append(val_metrics['component_analysis'])
            
            # Early stopping
            if val_metrics['balanced_accuracy'] > best_balanced_acc:
                best_balanced_acc = val_metrics['balanced_accuracy']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': asdict(self.config)
                }, 'best_hybrid_model.pth')
                
            else:
                patience_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"\nÉpoca {epoch+1:3d}/{self.config.num_epochs}")
            print(f"  Tempo: {epoch_time:.1f}s")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            # Log loss components se disponível
            if 'loss_components' in train_metrics:
                components = train_metrics['loss_components']
                print(f"    Components - Main: {components.get('main', 0):.4f}, "
                      f"ENet: {components.get('efficientnet', 0):.4f}, "
                      f"DINO: {components.get('dino', 0):.4f}")
            
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"BAcc: {val_metrics['balanced_accuracy']:.4f}, "
                  f"Kappa: {val_metrics['kappa']:.4f}")
            
            # Component analysis
            if 'component_analysis' in val_metrics:
                comp_analysis = val_metrics['component_analysis']
                print("  Components Performance:")
                for comp_name, comp_metrics in comp_analysis.items():
                    print(f"    {comp_name:12s}: BAcc={comp_metrics['balanced_accuracy']:.4f}, "
                          f"Kappa={comp_metrics['kappa']:.4f}")
            
            print(f"  Best BAcc: {best_balanced_acc:.4f}, "
                  f"Patience: {patience_counter}/{self.config.patience}")
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping na época {epoch+1}")
                break
        
        # 6. Load best model and final test
        checkpoint = torch.load('best_hybrid_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test
        test_metrics = self.validate_epoch_hybrid(model, dataloaders['test'], criterion)
        
        print(f"\n{'='*80}")
        print("RESULTADOS FINAIS DO SISTEMA HÍBRIDO")
        print(f"{'='*80}")
        for metric, value in test_metrics.items():
            if metric != 'component_analysis':
                print(f"{metric.upper():20s}: {value:.4f}")
        
        # Component analysis
        if 'component_analysis' in test_metrics:
            print(f"\nANÁLISE DE COMPONENTES:")
            comp_analysis = test_metrics['component_analysis']
            for comp_name, comp_metrics in comp_analysis.items():
                print(f"{comp_name.upper():15s}: "
                      f"BAcc={comp_metrics['balanced_accuracy']:.4f}, "
                      f"F1={comp_metrics['f1_macro']:.4f}, "
                      f"Kappa={comp_metrics['kappa']:.4f}")
        
        return {
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'best_epoch': checkpoint['epoch'],
            'config': self.config
        }

def compare_baseline_vs_hybrid(baseline_results: Dict, hybrid_results: Dict) -> Dict[str, float]:
    """
    Compara performance do baseline vs. sistema híbrido.
    
    Analisa:
    - Improvement em métricas chave
    - Significância estatística
    - Cost-benefit analysis
    """
    
    baseline_metrics = baseline_results['test_metrics']
    hybrid_metrics = hybrid_results['test_metrics']
    
    improvements = {}
    
    for metric in ['balanced_accuracy', 'f1_macro', 'kappa', 'auc']:
        if metric in baseline_metrics and metric in hybrid_metrics:
            baseline_val = baseline_metrics[metric]
            hybrid_val = hybrid_metrics[metric]
            improvement = hybrid_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100
            
            improvements[metric] = {
                'baseline': baseline_val,
                'hybrid': hybrid_val,
                'absolute_improvement': improvement,
                'relative_improvement_pct': improvement_pct
            }
    
    print(f"\n{'='*80}")
    print("COMPARAÇÃO: BASELINE vs. SISTEMA HÍBRIDO")
    print(f"{'='*80}")
    
    for metric, values in improvements.items():
        print(f"\n{metric.upper()}:")
        print(f"  Baseline: {values['baseline']:.4f}")
        print(f"  Híbrido:  {values['hybrid']:.4f}")
        print(f"  Melhoria: {values['absolute_improvement']:+.4f} "
              f"({values['relative_improvement_pct']:+.2f}%)")
        
        # Interpretação da melhoria
        if values['absolute_improvement'] > 0.02:  # >2% improvement
            status = "✅ MELHORIA SIGNIFICATIVA"
        elif values['absolute_improvement'] > 0.01:  # >1% improvement
            status = "✅ MELHORIA MODERADA"
        elif values['absolute_improvement'] > 0:
            status = "⚠️  MELHORIA MARGINAL"
        else:
            status = "❌ SEM MELHORIA"
        
        print(f"  Status:   {status}")
    
    # Recomendação final
    balanced_acc_improvement = improvements.get('balanced_accuracy', {}).get('absolute_improvement', 0)
    
    print(f"\n{'='*50}")
    print("RECOMENDAÇÃO FINAL:")
    print(f"{'='*50}")
    
    if balanced_acc_improvement > 0.02:
        print("✅ SISTEMA HÍBRIDO RECOMENDADO")
        print("- Melhoria significativa justifica complexidade adicional")
        print("- Proceder com validação multi-centro")
        print("- Considerar deployment clínico")
        
    elif balanced_acc_improvement > 0.01:
        print("⚠️  SISTEMA HÍBRIDO CONDICIONAL")
        print("- Melhoria moderada pode justificar em cenários específicos")
        print("- Avaliar cost-benefit para deployment")
        print("- Considerar otimizações adicionais")
        
    else:
        print("❌ BASELINE SUFICIENTE")
        print("- Sistema híbrido não oferece vantagem significativa")
        print("- Manter baseline EfficientNet")
        print("- Focar em validação multi-centro do baseline")
    
    return improvements

def main_hybrid_pipeline(baseline_checkpoint: str = "best_model.pth"):
    """
    Pipeline principal para sistema híbrido.
    
    Executa apenas se baseline atingiu critérios de performance.
    """
    
    # Verificar se baseline existe e atende critérios
    if not os.path.exists(baseline_checkpoint):
        print(f"❌ Checkpoint baseline não encontrado: {baseline_checkpoint}")
        print("Execute primeiro o sistema baseline até atingir ≥85% balanced accuracy")
        return None
    
    # Carregar métricas do baseline
    baseline_checkpoint_data = torch.load(baseline_checkpoint, map_location='cpu')
    baseline_balanced_acc = baseline_checkpoint_data['metrics']['balanced_accuracy']
    
    if baseline_balanced_acc < 0.85:
        print(f"❌ Baseline não atende critérios: {baseline_balanced_acc:.3f} < 0.85")
        print("Sistema híbrido só deve ser implementado se baseline ≥85% balanced accuracy")
        return None
    
    print(f"✅ Baseline aprovado: {baseline_balanced_acc:.3f} ≥ 0.85")
    print("Procedendo com sistema híbrido...")
    
    # Configuração híbrida
    config = HybridConfig(
        data_path="data",
        model_name="efficientnet_b4",
        dino_model="dino_vits16",
        fusion_method="feature_fusion",
        batch_size=24,  # Menor devido à complexidade
        learning_rate=5e-5,  # Mais conservador
        num_epochs=80,
        patience=20,
        use_mixup=True,
        use_cutmix=True
    )
    
    # Treinar sistema híbrido
    trainer = HybridTMETrainer(config, baseline_checkpoint)
    hybrid_results = trainer.train_hybrid_model()
    
    # Comparar com baseline
    baseline_results = {
        'test_metrics': baseline_checkpoint_data['metrics']
    }
    
    comparison = compare_baseline_vs_hybrid(baseline_results, hybrid_results)
    
    # Salvar resultados
    final_results = {
        'baseline_results': baseline_results,
        'hybrid_results': hybrid_results,
        'comparison': comparison,
        'config': asdict(config),
        'timestamp': time.time()
    }
    
    with open('hybrid_validation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("PIPELINE HÍBRIDO CONCLUÍDO")
    print("Resultados salvos em 'hybrid_validation_results.json'")
    print(f"{'='*80}")
    
    return final_results

if __name__ == "__main__":
    # Executar pipeline híbrido
    results = main_hybrid_pipeline("best_model.pth")