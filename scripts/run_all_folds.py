import os
import sys
import time
import subprocess
import itertools
import argparse
import yaml
import json
import torch

from pathlib import Path
from pathlib import Path

from core.analyze import aggregate_fold_summaries

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_gpu_queue(num_gpus):
    from itertools import cycle
    return cycle(range(num_gpus))


def launch_process(model_name, fold, config_path, gpu_id, dry_run=False):
    """
    Lança um processo de treinamento por fold com o modelo e GPU especificados.
    """
    script_dir = Path(__file__).parent
    training_script = script_dir.parent / "core" / "training.py"

    cmd = [
        "python", str(training_script),
        "--config", config_path,
        "--model", model_name,
        "--fold", str(fold)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🔄 Iniciando {model_name} fold {fold} na GPU {gpu_id}...")
    if dry_run:
        print("Comando:", " ".join(cmd))
        return None
    else:
        return subprocess.Popen(cmd, env=env), " ".join(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = [m["name"] for m in cfg["models"]]
    n_folds = cfg["dataset"]["n_folds"]
    jobs = list(itertools.product(models, range(n_folds)))

    if args.gpus is not None:
        num_gpus = args.gpus
    else:
        num_gpus = torch.cuda.device_count()
        print(f"🖥️ GPUs detectadas automaticamente: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("Nenhuma GPU detectada. Use --gpus 1 para forçar execução em CPU (não recomendado).")

    gpu_queue = get_gpu_queue(num_gpus)

    # === Execução dos jobs ===
    start_time = time.time()
    running = []
    executed_commands = []
    execution_log = []

    for model_name, fold in jobs:
        gpu_id = next(gpu_queue)
        result = launch_process(model_name, fold, args.config, gpu_id, dry_run=args.dry_run)
        if result and not args.dry_run:
            p, cmd_str = result
            running.append((p, model_name, fold, gpu_id, cmd_str))

    for p, model_name, fold, gpu_id, cmd_str in running:
        retcode = p.wait()
        status = "success" if retcode == 0 else "error"
        print(f"✔️ Concluído: {model_name} fold {fold} (status: {status})")
        executed_commands.append(cmd_str)
        execution_log.append({
            "model": model_name,
            "fold": fold,
            "gpu": gpu_id,
            "command": cmd_str,
            "status": status,
            "log_file": f"logs/{model_name}_fold{fold}.out"
        })

    # === Registro dos comandos executados ===
    Path("logs").mkdir(exist_ok=True)
    with open("logs/comandos_executados.txt", "w") as f:
        for cmd in executed_commands:
            f.write(cmd + "\n")

    # === Salvar execução ===
    meta_exec = {
        "config_used": args.config,
        "n_gpus": num_gpus,
        "models": models,
        "n_folds": n_folds,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "duration_minutes": round((time.time() - start_time) / 60, 2),
        "jobs": execution_log
    }
    with open("logs/execucao.json", "w") as f:
        json.dump(meta_exec, f, indent=4)

    # === Agregação final ===
    output_dir = cfg["data"]["output_dir"]
    for model_name in models:
        aggregate_fold_summaries(output_dir, model_name, n_folds)

    print("✅ Execução concluída e registrada com sucesso.")
