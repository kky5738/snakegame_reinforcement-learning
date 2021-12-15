import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction
from model import Linear_QNet, QTrainer
from pygame.math import Vector2
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005 # 0.001 # 0.0005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.85 # discount rate # 0.8~0.9
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11, 128, 3)
        self.model_target = Linear_QNet(11, 128, 3)

        # 25~29 line 테스트할 경우 주석 해제, 학습시킬 경우 주석
        # self.model.load_state_dict(torch.load('./model/module_best.pth'))
        # self.model.eval()

        # self.model_target.load_state_dict(torch.load('./model/module_target_best.pth'))
        # self.model.eval()
        self.trainer = QTrainer(self.model, self.model_target, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Vector2(head.x - 20, head.y)
        point_r = Vector2(head.x + 20, head.y)
        point_u = Vector2(head.x, head.y - 20)
        point_d = Vector2(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_game_over(point_r)) or 
            (dir_l and game.is_game_over(point_l)) or 
            (dir_u and game.is_game_over(point_u)) or 
            (dir_d and game.is_game_over(point_d)),

            # Danger right
            (dir_u and game.is_game_over(point_r)) or 
            (dir_d and game.is_game_over(point_l)) or 
            (dir_l and game.is_game_over(point_u)) or 
            (dir_r and game.is_game_over(point_d)),

            # Danger left
            (dir_d and game.is_game_over(point_r)) or 
            (dir_u and game.is_game_over(point_l)) or 
            (dir_r and game.is_game_over(point_u)) or 
            (dir_l and game.is_game_over(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Apple location 
            game.apple.x < game.head.x,  # apple left
            game.apple.x > game.head.x,  # apple right
            game.apple.y < game.head.y,  # apple up
            game.apple.y > game.head.y  # apple down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        for i in range(2):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 85 - self.n_games
        final_move = [0,0,0]
        
        # 학습시킬 경우 주석 해제, 테스트할 경우 100~107번째 라인 주석
        # if random.randint(0, 150) < self.epsilon:
        #     move = random.randint(0, 2)
        #     final_move[move] = 1 
        # else:
        #     state0 = torch.tensor(state, dtype=torch.float)
        #     prediction = self.model(state0)
        #     move = torch.argmax(prediction).item()
        #     final_move[move] = 1

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move

    def target_update(self):
        self.model_target.load_state_dict(self.model.state_dict())

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_scene(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                if score > 65:
                    agent.model.save('module_best.pth')
                    agent.model_target.save('module_target_best.pth')
            if agent.n_games%12==0:
                agent.target_update()
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()