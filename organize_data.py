#!/usr/bin/env python3
"""
Script para organizar dados HMU-GC-HE-30K na estrutura correta
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def organize_hmu_data(source_dir: str, target_dir: str):
    """
    Organiza dados HMU-GC-HE-30K seguindo estrutura adequada.
    
    Args:
        source_dir: Diretório com dados originais (hmu_gc_data)
        target_dir: Diretório de destino organizado (data)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Classes TME baseadas em Lou et al. (2025)
    classes = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
    
    print("="*60)
    print("ORGANIZANDO DADOS HMU-GC-HE-30K")
    print("="*60)
    print(f"Origem: {source_path}")
    print(f"Destino: {target_path}")
    print()
    
    # Criar estrutura de diretórios
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (target_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Verificar se diretório origem existe
    all_image_dir = source_path / "all_image"
    if not all_image_dir.exists():
        print(f"❌ Diretório não encontrado: {all_image_dir}")
        print("Verifique se o caminho está correto!")
        return
    
    print("📁 Criando divisões train/val/test...")
    total_files = 0
    
    # Para cada classe
    for cls in classes:
        class_dir = all_image_dir / cls
        
        if not class_dir.exists():
            print(f"⚠️  Diretório não encontrado: {class_dir}")
            continue
        
        # Listar arquivos de imagem
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
        files = []
        for ext in image_extensions:
            files.extend(list(class_dir.glob(f"*{ext}")))
            files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        if len(files) == 0:
            print(f"⚠️  Nenhuma imagem encontrada em: {class_dir}")
            continue
        
        print(f"📂 {cls}: {len(files)} imagens encontradas")
        
        # Shuffle para randomizar
        random.shuffle(files)
        
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
            dest_dir = target_path / split_name / cls
            
            for i, file_path in enumerate(file_list):
                # Criar nome único para evitar conflitos
                dest_path = dest_dir / f"{cls}_{i:06d}.png"
                
                try:
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"Erro ao copiar {file_path}: {e}")
                    continue
        
        print(f"  ✅ {len(train_files):4d} train, {len(val_files):4d} val, {len(test_files):4d} test")
        total_files += len(files)
    
    print(f"\n✅ Organização concluída!")
    print(f"📊 Total de arquivos processados: {total_files}")
    print(f"📁 Dados organizados em: {target_path}")
    
    # Verificar estrutura final
    print(f"\n📋 Estrutura final:")
    for split in ['train', 'val', 'test']:
        split_path = target_path / split
        total_split = 0
        for cls in classes:
            class_path = split_path / cls
            if class_path.exists():
                count = len(list(class_path.glob("*.png")))
                total_split += count
        print(f"  {split}: {total_split} imagens")

if __name__ == "__main__":
    # Configurar random seed para reprodutibilidade
    random.seed(42)
    
    # Caminhos baseados na sua estrutura
    source_dir = "hmu_gc_data"
    target_dir = "data"
    
    # Verificar se diretório origem existe
    if not Path(source_dir).exists():
        print(f"❌ Diretório não encontrado: {source_dir}")
        print("Verifique se o nome está correto!")
        exit(1)
    
    # Organizar dados
    organize_hmu_data(source_dir, target_dir)
    
    print(f"\n🚀 Pronto! Agora você pode executar:")
    print(f"python sistema_classificacao.py")