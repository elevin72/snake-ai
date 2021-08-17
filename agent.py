import torch
import random
import numpy as np
from collections import deque
import time
from game import *
from model import *
from helper import plot

MAX_MEMORY = 10000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
EXPLORATION_NUMBER = 80

game_info = namedtuple('game_info', 'state, action, reward, next_state, game_over')

class Agent:

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # control randomness
        self.gamma = 0.9 # discount rate??
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)


    def get_state(self, game: Snake):
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y - BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def get_action(self, state): # do_action()??
        # random moves -- tradeoff between exploration and exploitation
        self.epsilon = EXPLORATION_NUMBER - self.num_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action[move] = 1
        return action


def train():
    start_time = time.time()
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Snake()
    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)
        
        #train short term
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        #whats this?
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            # train long memory
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.num_games, 'Score', score, 'Record', record)
            print('Elapsed time', time.time() - start_time)

            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.num_games
            plot_avg_scores.append(avg_score)
            plot(plot_scores, plot_avg_scores)



if __name__ == '__main__':
    train()
