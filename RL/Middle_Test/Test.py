# import pacman game
from pacman import Directions
from pacmanUtils import *
from game import Agent
import game

# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import other libraries
import random
import numpy as np
import time
from collections import deque, namedtuple

# model parameters
DISCOUNT_RATE = 0.99  # discount factor
LEARNING_RATE = 0.0005  # learning rate parameter
REPLAY_MEMORY = 50000  # Replay buffer 의 최대 크기
LEARNING_STARTS = 5000  # 5000 스텝 이후 training 시작
TARGET_UPDATE_ITER = 400  # update target network

EPSILON_START = 0.8
EPSILON_END = 0.01
import math

###############################################
# Additional Template Codes                   #
# You may or may not use below skeleton code. #
###############################################


class Pacman(PacmanUtils):
    def __init__(self):
        print("Started just Pacman")
        print("just sample")

        self.step = 10000
        self.model = None

        self.epsilon = EPSILON_START
        self.win_counter = 0  # number of victory episodes
        self.steps_taken = 0  # steps taken across episodes
        self.steps_per_epi = 0  # steps taken in one episodes
        self.episode_number = 0
        self.episode_rewards = []

        self.policy = np.zeros()

        def predict(self, state):
            print(state)
            # if np.random.random()<self.epsilon:
            if np.random.random() < 1:  # For construction
                self.action = select_policy(state)
            else:
                self.action = np.random.randint(0, 4)  # random value between 0 and 3

            return self.action

        def select_policy(self):
            action = None
            return action

        def update_epsilon(self):
            sample = random.random()
            self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                           math.exp(-1. * self.steps_taken / 200)
            self.steps_taken += 1

        def step(self, next_state, reward, done):
            # next_state = self.state에 self.action 을 적용하여 나온 state
            # reward = self.state에 self.action을 적용하여 얻어낸 점수.
            if self.action is None:
                self.state = self.preprocess(next_state)
            else:
                self.next_state = self.preprocess(next_state)

                self.state = self.next_state

            # next
            self.episode_reward += reward
            self.steps_taken += 1
            self.steps_per_epi += 1
            self.update_epsilon()

            if (self.trained_model == False):
                self.train()
                self.update_epsilon()
                if (self.steps_taken % TARGET_UPDATE_ITER == 0):
                    # UPDATE target network
                    pass


        def reset(self):
            # 새로운 episode 시작시 불리는 함수.
            self.last_score = 0
            self.current_score = 0
            self.episode_reward = 0

            self.episode_number += 1
            self.steps_per_epi = 0

        def final(self, state):
            # episode 종료시 불리는 함수.
            done = True
            reward = self.getScore(state)
            if reward >= 0:  # not eaten by ghost when the game ends
                self.win_counter += 1

            self.step(state, reward, done)
            self.episode_rewards.append(self.episode_reward)
            win_rate = float(self.win_counter) / 500.0
            avg_reward = np.mean(np.array(self.episode_rewards))
            # print episode information
            if (self.episode_number % 500 == 0):
                print(
                    "Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(
                        self.episode_number,
                        self.win_counter, win_rate, avg_reward, self.epsilon))
                self.win_counter = 0
                self.episode_rewards = []
                if (self.trained_model == False and self.episode_number % 1000 == 0):
                    # Save model here
                    # torch.save(self.YOUR_NETWORK, self.model)
                    pass

        def blank_grid(self, state, N):
            width, height = state.data.layout.width, state.data.layout.height
            grid = np.zeros((N, width, height))
            return grid

        def pacman_grid(self, state):
            return state.getPacmanPosition()

        def ghost_grid(self, state):
            return state.getScaredTimer()

        def preprocess(self, state):
            # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.
            pacman_state = state.getPacmanPosition()
            ghost_state = state.getGhostPositions()[0]
            food_state = state.getFood()
            walls_state = state.getWalls()
            capsule_state = state.getCapsules()
            # reward          = state.getScore()
            state = {"pacman": pacman_state, "ghost": ghost_state, "food": food_state, "wall": walls_state,
                     "capsule": capsule_state}
            print(state)
            return state

if __name__=="__main__":


