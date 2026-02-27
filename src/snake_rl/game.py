import random
from collections import namedtuple
from enum import Enum
from pathlib import Path

import numpy as np
import pygame

pygame.init()


def _load_font(size=25):
    project_root = Path(__file__).resolve().parents[2]
    font_path = project_root / "assets" / "fonts" / "arial.ttf"
    if font_path.exists():
        return pygame.font.Font(str(font_path), size)
    return pygame.font.SysFont("arial", size)


font = _load_font(25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
DEFAULT_RENDER_SPEED = 120


class SnakeGameAI:
    def __init__(
        self,
        w=640,
        h=480,
        render=True,
        speed=DEFAULT_RENDER_SPEED,
        frame_limit_factor=120,
        step_penalty=-0.01,
        closer_reward=0.05,
        farther_penalty=-0.05,
        food_reward=10.0,
        death_penalty=-10.0,
    ):
        self.w = w
        self.h = h
        self.render = render
        self.speed = speed
        self.frame_limit_factor = frame_limit_factor
        self.step_penalty = step_penalty
        self.closer_reward = closer_reward
        self.farther_penalty = farther_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty

        self.grid_w = self.w // BLOCK_SIZE
        self.grid_h = self.h // BLOCK_SIZE
        if self.grid_w < 4 or self.grid_h < 4:
            raise ValueError("Grid is too small for the snake game.")

        self.display = None
        self.clock = pygame.time.Clock()
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI")

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        start_x = (self.grid_w // 2) * BLOCK_SIZE
        start_y = (self.grid_h // 2) * BLOCK_SIZE
        self.head = Point(start_x, start_y)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.snake_set = set(self.snake)

        self.score = 0
        self.frame_iteration = 0
        self.steps_without_food = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        # Random retry is fast on this grid and avoids recursion.
        while True:
            x = random.randint(0, self.grid_w - 1) * BLOCK_SIZE
            y = random.randint(0, self.grid_h - 1) * BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake_set:
                self.food = candidate
                return

    def _distance_to_food(self, point):
        return abs(point.x - self.food.x) + abs(point.y - self.food.y)

    def _handle_events(self):
        if not self.render:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_without_food += 1

        # 1. Handle window events
        self._handle_events()

        # 2. Move
        old_distance = self._distance_to_food(self.head)
        self._move(action)
        self.snake.insert(0, self.head)
        self.snake_set.add(self.head)

        # 3. Check terminal conditions
        reward = self.step_penalty
        game_over = False
        if self.is_collision() or self.frame_iteration > self.frame_limit_factor * len(self.snake):
            game_over = True
            reward = self.death_penalty
            return reward, game_over, self.score

        # 4. Food logic + shaping
        if self.head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward = self.food_reward
            self._place_food()
        else:
            tail = self.snake.pop()
            self.snake_set.discard(tail)

            new_distance = self._distance_to_food(self.head)
            if new_distance < old_distance:
                reward += self.closer_reward
            elif new_distance > old_distance:
                reward += self.farther_penalty

        # 5. Optional rendering
        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Hits boundary
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True

        # Hits body.
        # For the current head, we must explicitly check the body list because
        # `pt == self.head` can still be a collision when the head overlaps a body segment.
        if pt == self.head:
            if pt in self.snake[1:]:
                return True
        elif pt in self.snake_set:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(
            self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        )

        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action: [straight, right, left]
        clock_wise = (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP)
        idx = clock_wise.index(self.direction)
        move_idx = int(np.argmax(action))

        if move_idx == 0:
            new_dir = clock_wise[idx]
        elif move_idx == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
