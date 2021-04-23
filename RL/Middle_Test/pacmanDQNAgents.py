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

#import other libraries
import random
import numpy as np
import time
from collections import deque, namedtuple

# model parameters
DISCOUNT_RATE = 0.99        # discount factor
LEARNING_RATE = 0.0005      # learning rate parameter
REPLAY_MEMORY = 50000       # Replay buffer 의 최대 크기
LEARNING_STARTS = 5000 	    # 5000 스텝 이후 training 시작
TARGET_UPDATE_ITER = 400   # update target network

EPSILON_START = 0.8
EPSILON_END = 0.01
import math

###############################################
# Additional Template Codes                   #
# You may or may not use below skeleton code. #
###############################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class ReplayMemory:
    def __init__(self):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def write(self):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
###############################################
# End of Additional Template Codes            #
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
            # state를 넣어 policy에 따라 action을 반환 (epsilon greedy)
            # Hint: network에 state를 input으로 넣기 전에 preprocessing 해야합니다.
            # print(state)
            if np.random.random()<self.epsilon:
                self.action = self.policy()

            act = np.random.randint(0, 4)  # random value between 0 and 3

            self.action = act  # save action
            return act

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

        def train(self):
            # replay_memory로부터 mini batch를 받아 policy를 업데이트
            if (self.steps_taken > LEARNING_STARTS):
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

            else
                return

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


class PacmanDQN(PacmanUtils):
    def __init__(self, args):        
        print("Started Pacman DQN algorithm")
        #print(args)
        self.double = args['double']
        self.multistep = args['multistep']
        self.n_steps = args['n_steps']
        self.model = args['model']

        self.trained_model = args['trained_model']
        if self.trained_model:
            mode = "Test trained model"
        else:
            mode = "Training model"
        
        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}     Model : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode, args['model']))
        print("=" * 100)

        # Target 네트워크와 Local 네트워크, epsilon 값을 설정
        if self.trained_model:  # Test
            # self.YOUR_NETWORK = torch.load(self.model)
            self.epsilon = 0
        else:                   # Train
            self.epsilon = EPSILON_START  # epsilon init value

        # statistics
        self.win_counter = 0       # number of victory episodes
        self.steps_taken = 0       # steps taken across episodes
        self.steps_per_epi = 0     # steps taken in one episodes   
        self.episode_number = 0
        self.episode_rewards =[]  
        
        self.epsilon = EPSILON_START  # epsilon init value



    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환 (epsilon greedy)
        # Hint: network에 state를 input으로 넣기 전에 preprocessing 해야합니다.
        #print(state)
        act = np.random.randint(0, 4)  # random value between 0 and 3
        self.action = act # save action
        return act
    
    
    def update_epsilon(self):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
                # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
                # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)
    
    
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

        if(self.trained_model == False):
            self.train()
            self.update_epsilon()
            if(self.steps_taken % TARGET_UPDATE_ITER == 0):
                # UPDATE target network 
                pass
	
    
    def train(self):
        # replay_memory로부터 mini batch를 받아 policy를 업데이트
        if (self.steps_taken > LEARNING_STARTS):
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

        else
            return
    
    
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
        if reward >= 0: # not eaten by ghost when the game ends
            self.win_counter +=1

        self.step(state, reward, done)
        self.episode_rewards.append(self.episode_reward)
        win_rate = float(self.win_counter) / 500.0
        avg_reward = np.mean(np.array(self.episode_rewards))
		# print episode information
        if(self.episode_number%500 == 0):
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number,
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon))
            self.win_counter = 0
            self.episode_rewards= []
            if(self.trained_model==False and self.episode_number%1000 == 0):
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
        pacman_state    = state.getPacmanPosition()
        ghost_state     = state.getGhostPositions()[0]
        food_state      = state.getFood()
        walls_state     = state.getWalls()
        capsule_state   = state.getCapsules()
        # reward          = state.getScore()
        state = {"pacman" : pacman_state, "ghost": ghost_state,"food": food_state, "wall":walls_state, "capsule": capsule_state}
        print(state)
        return state


