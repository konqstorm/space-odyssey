# main.py
import argparse
import torch
import numpy as np

from environment import SpaceEnv
from agents import REINFORCEAgent, TRPOAgent
from training import train_reinforce, train_trpo
from manual_control import manual_control
from simulation import watch_agent

torch.set_num_threads(12)


def main():
    parser = argparse.ArgumentParser(
        description="RL Space Navigation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --manual                           # Поиграть самому
  python main.py --train-reinforce --episodes 2000  # Обучить REINFORCE
  python main.py --train-trpo --episodes 1000       # Обучить TRPO
  python main.py --watch --model best_reinforce.pth # Смотреть обученного агента
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
    
    # Параметры среды
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Ширина игрового поля (по умолчанию 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Высота игрового поля (по умолчанию 1080)"
    )
    parser.add_argument(
        "--asteroids",
        type=int,
        default=0,
        help="Количество астероидов (по умолчанию 0)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Максимальное число шагов в эпизоде (по умолчанию 1000)"
    )
    
    # Параметры обучения
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Количество эпизодов для обучения (по умолчанию 2000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Размер батча эпизодов для REINFORCE (по умолчанию 24)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (по умолчанию 3e-4)"
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.002,
        help="Коэффициент энтропии для REINFORCE (по умолчанию 0.002)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed для воспроизводимости. Если не задан, инициализация полностью случайная"
    )
    
    # Параметры Наблюдения
    parser.add_argument(
        "--model",
        type=str,
        default="models/reinforce/best_reinforce.pth",
        help="Путь к файлу модели для загрузки (по умолчанию models/reinforce/best_reinforce.pth)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Путь для сохранения обученной модели (по умолчанию models/reinforce или models/trpo)"
    )
    
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Создание окружения
    env = SpaceEnv(
        space_size=(args.width, args.height),
        num_asteroids=args.asteroids,
        max_steps=args.max_steps
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
        save_path = args.save_model or "models/reinforce"
        print("=" * 60)
        print("РЕЖИМ: Обучение REINFORCE")
        print("=" * 60)
        print(f"Эпизодов: {args.episodes}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Entropy coeff: {args.entropy_coeff}")
        print(f"Seed: {'random' if args.seed is None else args.seed}")
        print(f"Сохранение модели в: {save_path}")
        print()
        train_reinforce(
            env,
            episodes=args.episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            entropy_coeff=args.entropy_coeff,
            save_path=save_path
        )
        
    elif args.train_trpo:
        save_path = args.save_model or "models/trpo"
        print("=" * 60)
        print("РЕЖИМ: Обучение TRPO")
        print("=" * 60)
        print(f"Эпизодов: {args.episodes}")
        print(f"Learning Rate: {args.lr}")
        print(f"Seed: {'random' if args.seed is None else args.seed}")
        print(f"Сохранение модели в: {save_path}")
        print()
        train_trpo(env, episodes=args.episodes, save_path=save_path)
        
    elif args.watch:
        print("=" * 60)
        print("РЕЖИМ: Просмотр обученного агента")
        print("=" * 60)
        print(f"Загрузка модели из: {args.model}")
        print("ESC или закрытие окна для выхода")
        print()
        watch_agent(env, REINFORCEAgent, args.model)


if __name__ == "__main__":
    main()
