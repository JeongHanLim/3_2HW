from pacman import Directions
from pacmanUtils import *
from game import Agent
import game
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class DQN(torch.nn.Module):
    def __init__(self, r, w, h, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convr = conv2d_size_out(conv2d_size_out(conv2d_size_out(r)))
        linear_input_size = convw * convh * convr * 32
        self.head =nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def write(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.memory, 64)

    def __len__(self):
        return len(self.memory)

    def print(self):
        print(self.memory[self.position-i])


class PacmanDQN(PacmanUtils):
    def __init__(self, args):
        super().__init__()
        print("Started Pacman DQN algorithm")
        self.double     = args['double']
        self.multistep  = args['multistep']
        self.n_steps    = args['n_steps']
        self.model  = args['model']

        self.trained_model = args['trained_model']
        if self.trained_model:
            mode = "Test trained model"
        else:
            mode = "Training model"
            self.YOUR_NETWORK = DQN(7, 7, 7, 1).to(device)
            self.optimizer = optim.SGD(self.YOUR_NETWORK.parameters(), lr = 0.001, momentum=0.9)

        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}     Model : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode, args['model']))
        print("=" * 100)

        if self.trained_model:  # Test
            #self.YOUR_NETWORK = torch.load(self.model)
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

        self.action = 4
        #optimizer = optim.RMSprop(self.YOUR_NETWORK.state_dict())
        self.memory = ReplayMemory(REPLAY_MEMORY)

    def predict(self, state):
            act = np.random.randint(0, 4)  # random value between 0 and 3
            self.action = act # save action
            return act

    def update_epsilon(self):

        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_taken / 200)

        return

    def step(self, next_state, reward, done):
        # next_state = self.state에 self.action 을 적용하여 나온 state
        # reward = self.state에 self.action을 적용하여 얻어낸 점수.
        if self.action is None:
            self.state = self.preprocess(next_state)
        else:
            self.next_state = self.preprocess(next_state)
            self.memory.write(self.state, self.action, self.next_state, reward)
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

        else:

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
                torch.save(self.YOUR_NETWORK, self.model+str(self.episode_number))


    def blank_grid(self, state, N):
        width, height = state.data.layout.width, state.data.layout.height
        grid = np.zeros((N, width, height))
        return grid

    def grid2array(self, blank_grid, index, method):
        blank_grid[index] = [x[:] for x in method]
        return blank_grid

    def scalar2array(self, blank_grid, index, method, expected=None):
        if expected is None:
            expected = [0, 0]

        pos = method

        if index != 4:
            x = blank_grid.shape[1]-1-pos[1]+expected[0]
            y = pos[0] + expected[1]
            blank_grid[index][int(x)][int(y)] = 1

        if index == 4:
            if pos != []:
                x = blank_grid.shape[1]-1-pos[0][1]+expected[0]
                y = pos[0][0]+expected[1]
                blank_grid[index][int(x)][int(y)] = 1

        return blank_grid

    def state2array(self, blank_grid, index, method, state):
        data = method.__str__()
        data = data.split(", ")[2]
        expected_pos = [0, 0]
        if data == "West":
            expected_pos = [-1, 0]
        elif data == "East":
            expected_pos = [1, 0]
        elif data == "North":
            expected_pos = [0, 1]
        elif data == "South":
            expected_pos = [0, -1]

        if index == 5:
            blank_grid = self.scalar2array(blank_grid, 5, state.getPacmanPosition(), expected=expected_pos)
        elif index == 6:
            blank_grid = self.scalar2array(blank_grid, 6, state.getGhostPositions()[0], expected=expected_pos)
        else:
            print("Something wrong here")
        return blank_grid

    def preprocess(self, state):
        # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.
        blank_grid = self.blank_grid(state, 7)
        blank_grid  = self.scalar2array(blank_grid, 0, state.getPacmanPosition())
        blank_grid  = self.scalar2array(blank_grid, 1, state.getGhostPositions()[0])
        blank_grid  = self.grid2array(blank_grid, 2, state.getFood())
        blank_grid  = self.grid2array(blank_grid, 3, state.getWalls())
        blank_grid  = self.scalar2array(blank_grid, 4, state.getCapsules())
        blank_grid  = self.state2array(blank_grid, 5, state.data.agentStates[0], state)
        blank_grid  = self.state2array(blank_grid, 6, state.data.agentStates[1], state)
        return blank_grid

if __name__=="__main__":

    print("3")