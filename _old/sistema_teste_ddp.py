#!/usr/bin/env python3
"""
TESTE RÁPIDO DDP - Detecta problemas em 5-10 minutos
=====================================================

Execute este script ANTES do treinamento completo para identificar
se haverá problemas de timeout NCCL.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import datetime
from pathlib import Path

def test_ddp_basic(rank, world_size):
    """Teste básico de DDP com operações que causam timeout"""
    try:
        print(f"[Rank {rank}] Iniciando teste...")
        
        # Configurar ambiente
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        
        # TESTE 1: Init process group (pode travar aqui)
        print(f"[Rank {rank}] Teste 1: Inicializando process group...")
        start_time = time.time()
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=2)  # Timeout curto para teste
        )
        
        init_time = time.time() - start_time
        print(f"[Rank {rank}] ✅ Init OK em {init_time:.2f}s")
        
        # Configurar device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # TESTE 2: Operações coletivas básicas
        print(f"[Rank {rank}] Teste 2: Operações coletivas...")
        
        # AllReduce
        tensor = torch.randn(1000, 1000, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] ✅ AllReduce OK")
        
        # AllGather (operação que falhou no erro original)
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        print(f"[Rank {rank}] ✅ AllGather OK")
        
        # TESTE 3: Broadcast
        if rank == 0:
            broadcast_tensor = torch.randn(2000, 2000, device=device)
        else:
            broadcast_tensor = torch.zeros(2000, 2000, device=device)
        
        dist.broadcast(broadcast_tensor, src=0)
        print(f"[Rank {rank}] ✅ Broadcast OK")
        
        # TESTE 4: Barrier (pode travar)
        print(f"[Rank {rank}] Teste 4: Barrier...")
        dist.barrier()
        print(f"[Rank {rank}] ✅ Barrier OK")
        
        # TESTE 5: Simular carga de trabalho desbalanceada
        print(f"[Rank {rank}] Teste 5: Carga desbalanceada...")
        if rank == 0:
            time.sleep(2)  # Rank 0 demora mais
        
        dist.barrier()  # Todos esperam
        print(f"[Rank {rank}] ✅ Carga desbalanceada OK")
        
        print(f"[Rank {rank}] 🎉 TODOS OS TESTES PASSARAM!")
        return True
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ ERRO: {e}")
        return False
    finally:
        try:
            dist.destroy_process_group()
        except:
            pass

def test_nccl_availability():
    """Testa se NCCL está funcionando corretamente"""
    print("🔍 Testando disponibilidade NCCL...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA não disponível")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"📊 GPUs detectadas: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠️  Menos de 2 GPUs - DDP não será usado")
        return False
    
    # Testar NCCL backend
    try:
        if dist.is_nccl_available():
            print("✅ NCCL disponível")
        else:
            print("❌ NCCL não disponível")
            return False
    except:
        print("❌ Erro ao verificar NCCL")
        return False
    
    return True

def run_quick_ddp_test():
    """Executa teste rápido completo"""
    print("="*60)
    print("🧪 TESTE RÁPIDO DDP - DETECÇÃO DE PROBLEMAS")
    print("="*60)
    
    # Teste 1: Verificações básicas
    if not test_nccl_availability():
        print("\n❌ Falha nos testes básicos - use single GPU")
        return False
    
    # Teste 2: DDP real
    world_size = min(torch.cuda.device_count(), 2)  # Máximo 2 GPUs para teste
    
    print(f"\n🚀 Testando DDP com {world_size} GPUs...")
    print("⏱️  Isso deve levar 30-60 segundos...")
    
    try:
        # Configurações NCCL para teste
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_DEBUG'] = 'WARN'
        
        mp.spawn(test_ddp_basic, args=(world_size,), nprocs=world_size, join=True)
        
        print("\n🎉 TESTE DDP PASSOU - Sistema pronto para treinamento distribuído!")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE DDP FALHOU: {e}")
        print("\n💡 RECOMENDAÇÃO: Use single GPU (use_distributed=False)")
        return False

if __name__ == "__main__":
    success = run_quick_ddp_test()
    
    if success:
        print("\n" + "="*60)
        print("✅ RESULTADO: Pode usar DDP com segurança")
        print("📝 Configure: use_distributed=True")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  RESULTADO: Use single GPU")
        print("📝 Configure: use_distributed=False")
        print("="*60)