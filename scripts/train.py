from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from snake_rl.agent import TrainConfig, parse_args, train


def main():
    args = parse_args()
    cfg = TrainConfig(
        max_games=args.games,
        render_training=args.render,
        enable_plot=args.plot,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
