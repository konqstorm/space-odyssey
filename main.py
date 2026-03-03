# main.py
import argparse
import torch
import numpy as np

from src.env import SpaceEnv
from src.agents import REINFORCEAgent, TRPOAgent
from src.training import train
from src.manual_control import manual_control
from src.simulation import watch_agent
from src.config import (
    load_env_config,
    load_reinforce_config,
    load_trpo_config,
    load_runtime_config,
)


def _resolve_watch_agent(agent_name):
    name = str(agent_name).strip().lower()
    if name == "reinforce":
        return REINFORCEAgent
    if name == "trpo":
        return TRPOAgent
    raise ValueError(f"Unsupported watch agent: {agent_name}. Use 'reinforce' or 'trpo'.")


def main():
    parser = argparse.ArgumentParser(
        description="RL Space Navigation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --manual                           # Поиграть самому
  python main.py --train-reinforce                  # Обучить REINFORCE (параметры из configs/reinforce_training.yaml)
  python main.py --train-trpo                       # Обучить TRPO (параметры из configs/trpo_training.yaml)
  python main.py --watch                            # Смотреть агента (параметры из configs/runtime.yaml)
        """
    )
    
    # Режимы
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--manual",
        action="store_true",
        help="Режим ручного управления (стрелки вверх/влево/вправо)"
    )
    mode_group.add_argument(
        "--train-reinforce",
        action="store_true",
        help="Обучение REINFORCE агента"
    )
    mode_group.add_argument(
        "--train-trpo",
        action="store_true",
        help="Обучение TRPO агента"
    )
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Просмотр обученного агента"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Директория с YAML-конфигами (по умолчанию configs)"
    )
    
    args = parser.parse_args()
    runtime_config = load_runtime_config(args.config_dir)
    env_config = load_env_config(args.config_dir)
    reinforce_config = load_reinforce_config(args.config_dir)
    trpo_config = load_trpo_config(args.config_dir)

    torch.set_num_threads(int(runtime_config["torch_num_threads"]))
    seed = runtime_config["seed"]

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Создание окружения
    env = SpaceEnv(
        space_size=(int(env_config["width"]), int(env_config["height"])),
        num_asteroids=int(env_config["num_asteroids"]),
        max_steps=int(env_config["max_steps"]),
    )
    
    # Выбор режима и выполнение
    if args.manual:
        print("=" * 60)
        print("РЕЖИМ: Ручное управление")
        print("=" * 60)
        print("Управление: ↑ (вперёд), ← (влево), → (вправо)")
        print("Закрыть окно для выхода")
        print()
        manual_control(env)
        
    elif args.train_reinforce:
        runs_root = runtime_config["train"].get("runs_dir", "runs")
        print("=" * 60)
        print("РЕЖИМ: Обучение REINFORCE")
        print("=" * 60)
        print(f"Эпизодов: {reinforce_config['episodes']}")
        print(f"Batch size: {reinforce_config['batch_size']}")
        print(f"Learning Rate: {reinforce_config['lr']}")
        print(f"Entropy coeff: {reinforce_config['entropy_coeff']}")
        print(f"Gamma: {reinforce_config['gamma']}")
        print(f"Seed: {'random' if seed is None else seed}")
        print(f"Runs directory: {runs_root}")
        print()
        train(
            env,
            "reinforce",
            reinforce_config,
            runs_root=runs_root,
            used_configs={
                "env": env_config,
                "runtime": runtime_config,
                "reinforce_training": reinforce_config,
            },
        )
        
    elif args.train_trpo:
        runs_root = runtime_config["train"].get("runs_dir", "runs")
        print("=" * 60)
        print("РЕЖИМ: Обучение TRPO")
        print("=" * 60)
        print(f"Эпизодов: {trpo_config['episodes']}")
        print(f"Learning Rate: {trpo_config['lr']}")
        print(f"Gamma: {trpo_config['gamma']}")
        print(f"Delta: {trpo_config['delta']}")
        print(f"Seed: {'random' if seed is None else seed}")
        print(f"Runs directory: {runs_root}")
        print()
        train(
            env,
            "trpo",
            trpo_config,
            runs_root=runs_root,
            used_configs={
                "env": env_config,
                "runtime": runtime_config,
                "trpo_training": trpo_config,
            },
        )
        
    elif args.watch:
        watch_cfg = runtime_config["watch"]
        agent_cls = _resolve_watch_agent(watch_cfg.get("agent", "reinforce"))
        model_path = watch_cfg["model_path"]
        print("=" * 60)
        print("РЕЖИМ: Просмотр обученного агента")
        print("=" * 60)
        print(f"Агент: {watch_cfg.get('agent', 'reinforce')}")
        print(f"Загрузка модели из: {model_path}")
        print("ESC или закрытие окна для выхода")
        print()
        watch_agent(env, agent_cls, model_path)


if __name__ == "__main__":
    main()
