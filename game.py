import pygame, sys
import random
from pygame.math import Vector2
import numpy as np
from enum import Enum

pygame.init()

# direction
class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# color
APPLE = (200, 0, 0)
HEAD_OUTSIDE = (0, 0, 0)
BODY_OUTSIDE = (0, 0, 255)
INSIDE = (0, 100, 255)

font = pygame.font.SysFont('arial', 20)

BLOCK_SIZE = 20
SPEED = 63

class SnakeGameAI:

    def __init__(self):
        self.width = 640
        self.height = 480

        self.row_count = self.height // BLOCK_SIZE
        self.column_count = self.width // BLOCK_SIZE

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.score = 0

        self.head = Vector2(self.width/2, self.height/2)
        self.snake = [self.head,
                      Vector2(self.head.x-BLOCK_SIZE, self.head.y),
                      Vector2(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.apple = None
        self._place_apple()
        self.frame_iteration = 0

    def play_scene(self, action):
        self.frame_iteration += 1
        reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game_over = True
                    return reward, game_over, self.score

        self._move_snake(action)
        game_over = self.is_game_over()
        if game_over or self.frame_iteration > 100*len(self.snake):
            reward = -10 # original = -10
            return reward, game_over, self.score
        
        if self.head == self.apple:
            self.score += 1
            reward = 12 # 리워드 수정할까?
            self._place_apple()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _place_apple(self):
        x = random.randint(0, self.column_count-1)*BLOCK_SIZE
        y = random.randint(0, self.row_count-1)*BLOCK_SIZE
        self.apple =Vector2(x, y)
        if self.apple in self.snake:
            self._place_apple()

    def _move_snake(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx-1) % 4
            new_dir = clock_wise[next_idx] #right
        else:
            next_idx = (idx+1) % 4
            new_dir = clock_wise[next_idx] #left
        
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

        self.head = Vector2(x, y) # 업데이트된 좌표를 헤드에 반영, 헤드의움직임
        self.snake.insert(0, self.head)

    def is_game_over(self, vt=None):
        if vt is None:
            vt = self.head

        if vt.x > self.width - BLOCK_SIZE or vt.x < 0 or vt.y > self.height - BLOCK_SIZE or vt.y < 0:
            return True
        
        if vt in self.snake[1:]:
            return True
        
        return False

    def _update_ui(self):
        self._draw_background()
        self._draw_apple()
        self._draw_snake()

        score_board = font.render("Score: " + str(self.score), True, (0, 0, 0))
        self.screen.blit(score_board, [0, 0])

        pygame.display.update()

    def _draw_background(self):
        self.screen.fill((175, 215, 70))
        for col in range(self.column_count):
            for row in range(self.row_count):
                if (row % 2 == 0 and col % 2 == 0) or (row % 2 != 0 and col % 2 != 0):
                    pygame.draw.rect(self.screen, (167, 209, 61), pygame.Rect(col*BLOCK_SIZE, row*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def _draw_snake(self):
        for vec2 in self.snake:
            if vec2 == self.head:
                pygame.draw.rect(self.screen, HEAD_OUTSIDE, pygame.Rect(vec2.x, vec2.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(self.screen, BLOCK_SIZE, pygame.Rect(vec2.x, vec2.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, INSIDE, pygame.Rect(vec2.x+4, vec2.y+4, 12, 12))

    def _draw_apple(self):
        pygame.draw.rect(self.screen, APPLE, pygame.Rect(self.apple.x, self.apple.y, BLOCK_SIZE, BLOCK_SIZE))