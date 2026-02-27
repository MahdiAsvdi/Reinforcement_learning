import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .game import BLOCK_SIZE, Direction, Point, SnakeGameAI
from .helper import plot
from .model import Linear_QNet, QTrainer


STATE_SIZE = 20
ACTION_SIZE = 3


@dataclass
class TrainConfig:
    max_games: int = 2000
    width: int = 640
    height: int = 480
    render_training: bool = False
    render_eval: bool = False
    render_speed: int = 180

    # DQN
    hidden_sizes: tuple = (512, 256)
    lr: float = 5e-4
    gamma: float = 0.97
    tau: float = 0.01
    grad_clip: float = 5.0
    weight_decay: float = 1e-5

    # Replay
    max_memory: int = 200_000
    batch_size: int = 2048
    min_replay_size: int = 5000
    train_every_steps: int = 1

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995

    # Logging / eval
    print_every: int = 20
    eval_every: int = 100
    eval_episodes: int = 10
    enable_plot: bool = False

    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.state_size = state_size

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.position = 0
        self.size = 0

    def push(self, state, action_idx, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action_idx
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


class Agent:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.n_games = 0
        self.n_steps = 0
        self.record = 0
        self.total_score = 0
        self.best_mean_score = float("-inf")
        self.epsilon = config.epsilon_start

        self.memory = ReplayBuffer(config.max_memory, STATE_SIZE)
        self.model = Linear_QNet(STATE_SIZE, config.hidden_sizes, ACTION_SIZE)
        self.trainer = QTrainer(
            self.model,
            lr=config.lr,
            gamma=config.gamma,
            tau=config.tau,
            grad_clip=config.grad_clip,
            weight_decay=config.weight_decay,
        )

        self.recent_losses = deque(maxlen=200)

    @staticmethod
    def _point_in_direction(point, direction, steps=1):
        if direction == Direction.RIGHT:
            return Point(point.x + BLOCK_SIZE * steps, point.y)
        if direction == Direction.LEFT:
            return Point(point.x - BLOCK_SIZE * steps, point.y)
        if direction == Direction.DOWN:
            return Point(point.x, point.y + BLOCK_SIZE * steps)
        return Point(point.x, point.y - BLOCK_SIZE * steps)

    def get_state(self, game):
        head = game.snake[0]
        clock_wise = (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP)
        idx = clock_wise.index(game.direction)

        straight_dir = clock_wise[idx]
        right_dir = clock_wise[(idx + 1) % 4]
        left_dir = clock_wise[(idx - 1) % 4]

        point_s1 = self._point_in_direction(head, straight_dir, steps=1)
        point_r1 = self._point_in_direction(head, right_dir, steps=1)
        point_l1 = self._point_in_direction(head, left_dir, steps=1)

        point_s2 = self._point_in_direction(head, straight_dir, steps=2)
        point_r2 = self._point_in_direction(head, right_dir, steps=2)
        point_l2 = self._point_in_direction(head, left_dir, steps=2)

        point_abs_left = self._point_in_direction(head, Direction.LEFT, steps=1)
        point_abs_right = self._point_in_direction(head, Direction.RIGHT, steps=1)
        point_abs_up = self._point_in_direction(head, Direction.UP, steps=1)
        point_abs_down = self._point_in_direction(head, Direction.DOWN, steps=1)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = np.array(
            [
                # Relative hazards
                game.is_collision(point_s1),
                game.is_collision(point_r1),
                game.is_collision(point_l1),
                game.is_collision(point_s2),
                game.is_collision(point_r2),
                game.is_collision(point_l2),
                # Absolute hazards around head
                game.is_collision(point_abs_left),
                game.is_collision(point_abs_right),
                game.is_collision(point_abs_up),
                game.is_collision(point_abs_down),
                # Direction one-hot
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                # Food relative position
                game.food.x < game.head.x,
                game.food.x > game.head.x,
                game.food.y < game.head.y,
                game.food.y > game.head.y,
                # Normalized food vector (dense signal)
                (game.food.x - game.head.x) / game.w,
                (game.food.y - game.head.y) / game.h,
            ],
            dtype=np.float32,
        )
        return state

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.push(state, action_idx, reward, next_state, done)

    def update_epsilon(self):
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start * (self.config.epsilon_decay**self.n_games),
        )

    def get_action(self, state, explore=True):
        if explore:
            self.update_epsilon()
        else:
            self.epsilon = 0.0

        if explore and random.random() < self.epsilon:
            action_idx = random.randint(0, ACTION_SIZE - 1)
        else:
            state0 = torch.as_tensor(
                state, dtype=torch.float32, device=self.trainer.device
            ).unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state0)
            action_idx = int(torch.argmax(prediction, dim=1).item())

        final_move = [0, 0, 0]
        final_move[action_idx] = 1
        return final_move, action_idx

    def train_step(self):
        if len(self.memory) < self.config.min_replay_size:
            return None
        if self.n_steps % self.config.train_every_steps != 0:
            return None

        batch = self.memory.sample(self.config.batch_size)
        loss = self.trainer.train_step(*batch)
        self.recent_losses.append(loss)
        return loss


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eval(agent, cfg):
    eval_game = SnakeGameAI(
        w=cfg.width,
        h=cfg.height,
        render=cfg.render_eval,
        speed=cfg.render_speed,
    )
    scores = []
    for _ in range(cfg.eval_episodes):
        eval_game.reset()
        done = False
        while not done:
            state = agent.get_state(eval_game)
            move, _ = agent.get_action(state, explore=False)
            _, done, score = eval_game.play_step(move)
        scores.append(score)

    return float(np.mean(scores)), int(np.max(scores))


def train(cfg: TrainConfig):
    set_global_seed(cfg.seed)

    plot_scores = []
    plot_mean_scores = []

    agent = Agent(cfg)
    game = SnakeGameAI(
        w=cfg.width,
        h=cfg.height,
        render=cfg.render_training,
        speed=cfg.render_speed,
    )

    while agent.n_games < cfg.max_games:
        state_old = agent.get_state(game)
        final_move, action_idx = agent.get_action(state_old, explore=True)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.remember(state_old, action_idx, reward, state_new, done)
        agent.n_steps += 1
        agent.train_step()

        if done:
            game.reset()
            agent.n_games += 1
            agent.total_score += score
            mean_score = agent.total_score / agent.n_games

            if score > agent.record:
                agent.record = score
                agent.model.save("best_score_model.pth")

            if mean_score > agent.best_mean_score:
                agent.best_mean_score = mean_score
                agent.model.save("best_mean_model.pth")

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            if cfg.enable_plot:
                plot(plot_scores, plot_mean_scores)

            if agent.n_games % cfg.print_every == 0 or score == agent.record:
                avg_loss = (
                    float(np.mean(agent.recent_losses))
                    if len(agent.recent_losses) > 0
                    else 0.0
                )
                print(
                    f"Game {agent.n_games:4d} | "
                    f"Score {score:3d} | Record {agent.record:3d} | "
                    f"Mean {mean_score:6.2f} | Eps {agent.epsilon:.3f} | "
                    f"Loss {avg_loss:.4f} | Memory {len(agent.memory)}"
                )

            if cfg.eval_every > 0 and agent.n_games % cfg.eval_every == 0:
                eval_mean, eval_max = run_eval(agent, cfg)
                print(
                    f"[Eval] games={agent.n_games} avg_score={eval_mean:.2f} max_score={eval_max}"
                )

    agent.model.save("final_model.pth")
    print(
        f"Training finished at game {agent.n_games}. "
        f"Best score={agent.record}, best mean score={agent.best_mean_score:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a high-performance Snake RL agent.")
    parser.add_argument("--games", type=int, default=2000, help="Number of training games.")
    parser.add_argument(
        "--render", action="store_true", help="Render training visually (slower)."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Enable live training plot (slower)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Run greedy evaluation every N games (0 disables).",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of episodes for each evaluation."
    )
    return parser.parse_args()


if __name__ == "__main__":
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
