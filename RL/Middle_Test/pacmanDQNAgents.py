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
LEARNING_STARTS = 5000	    # 5000 스텝 이후 training 시작
TARGET_UPDATE_ITER = 4000  # update target network
BATCH_SIZE = 64
EPSILON_START = 0.8
EPSILON_END = 0.01
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ConvBlock(nn.Sequential):
    def __init__(self, inchannel, outchannel, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__(*[nn.Conv2d(inchannel, outchannel, kernel_size, stride=stride),
                                          #nn.BatchNorm2d(outchannel),
                                          nn.LeakyReLU()
                                          ])


class DQN(torch.nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(*[ConvBlock(6, 16, 3), ConvBlock(16, 32, 3), ConvBlock(32, 16, 3)])
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, outputs)

    def forward(self, x):
        z = self.conv_layers(x)
        z = self.pool(z)
        z = self.flatten(z)
        z = self.fc1(z)
        z = F.leaky_relu(z)
        z = self.fc3(z)
        return z


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

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class PacmanDQN(PacmanUtils):
    def __init__(self, args):
        super().__init__()
        print("Started Pacman DQN algorithm")
        self.double, self.multistep, self.n_steps, self.trained_model, self.model = None, None, None, None, None
        for k in args.keys():
            setattr(self, k, args[k])

        self.losses = []
        self.memory = ReplayMemory(REPLAY_MEMORY)
        self.win_counter = 0       # number of victory episodes
        self.steps_taken = 0       # steps taken across episodes
        self.steps_per_epi = 0     # steps taken in one episodes
        self.episode_number = 0
        self.episode_rewards =[]
        self.batch_size = BATCH_SIZE

        if self.trained_model:
            mode = "Test trained model"
            self.target_network=DQN(4)
            self.target_network  = torch.load(self.model)
            self.epsilon = 0

        else:
            mode = "Training model"
            self.policy_network = DQN(4)
            self.target_network = DQN(4)
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)
            self.epsilon = EPSILON_START  # epsilon init value

        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}     Model : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode, args['model']))
        print("=" * 100)

    def predict(self, state):
        if np.random.random() > self.epsilon:

            with torch.no_grad():
                state = self.preprocess(state)
                state = state.unsqueeze(0)
                state = state.float()
                result = self.target_network(state)
                self.action = result.max(1)[1].view(1,1)
        else:
            self.action = torch.tensor([np.random.randint(0, 4)]).long()

        return self.action.item()


    def update_epsilon(self):
        if self.trained_model:
            self.epsilon = 0
        else:
            self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_taken / 40000)


        return

    def step(self, next_state, reward, done):
        if self.action is None:
            self.state = self.preprocess(next_state)
        else:
            self.next_state = self.preprocess(next_state)
            self.memory.write(self.state, self.action, self.next_state, reward, done)
            self.state = self.next_state

        self.episode_reward += reward
        self.steps_taken += 1
        self.steps_per_epi += 1
        self.update_epsilon()

        if not self.trained_model:
            self.update_epsilon()
            self.train()
            if self.steps_taken % TARGET_UPDATE_ITER == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self):
        if self.steps_taken > LEARNING_STARTS:
            batched = Transition(*zip(*self.memory.sample(self.batch_size)))
            state_batch = torch.stack(batched.state).float()
            action_batch = torch.tensor(batched.action).long().unsqueeze(1)
            reward_batch = torch.tensor(batched.reward).float().unsqueeze(1)
            next_state_batch = torch.stack(batched.next_state).float()
            num_done = torch.tensor(batched.done).reshape(-1, 1)

            if self.double:
                next_state_actions = self.target_network(next_state_batch).max(1)[1].view(64,1).long()
                next_sv = self.target_network(next_state_batch).gather(1, next_state_actions)
                next_sv = next_sv.float()
            else:
                next_sv = self.target_network(next_state_batch).detach().max(dim=1)[0]
                next_sv = next_sv.unsqueeze(1).float()

            expected_sa_values = next_sv * DISCOUNT_RATE*(1-num_done*1) + reward_batch
            current_q = self.policy_network(state_batch).gather(1, action_batch)
            # if self.steps_taken % 1000 == 0:
            #     print("current", current_q[15:25],"expected", expected_sa_values[15:25])
            loss = F.smooth_l1_loss(current_q, expected_sa_values)
            self.losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def reset(self):
        # 새로운 episode 시작시 불리는 함수.
        self.last_score = 0
        self.current_score = 0
        self.episode_reward = 0
        self.episode_number += 1
        self.steps_per_epi = 0

    def final(self, state):
        done = True
        reward = self.getScore(state)
        if reward >= 0:
            self.win_counter +=1

        self.step(state, reward, done)
        self.episode_rewards.append(self.episode_reward)
        win_rate = float(self.win_counter) / 500.0
        avg_reward = np.mean(np.array(self.episode_rewards))

        if self.episode_number % 500 == 0:
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number,
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon))
            print("Episode no = {:>5};, recent average loss {}".format(self.episode_number,
                                                                       np.mean(self.losses[-500:-1])))
            # print(self.episode_rewards)

        if self.episode_number%500==0:
            self.win_counter = 0
            self.episode_rewards= []
            if self.trained_model==False:
                torch.save(self.target_network, self.model+str(self.episode_number))

    def blank_grid(self, state, N):
        width, height = state.data.layout.width, state.data.layout.height
        grid = np.zeros((N, width, height))
        return grid

    def grid2array(self, blank_grid, index, state):
        for i in range(7):
            for j in range(7):
                blank_grid[index][i][j] =state.data.layout.walls[j][6-i]
                blank_grid[index+1][i][j]=state.data.food[j][6-i]
        return blank_grid

    def scalar2array(self, blank_grid, index, method, expected=None, data=1):
        if expected is None:
            expected = [0, 0]

        pos = method

        if index != 3 : #For Capsule
            x = blank_grid.shape[1]-1-pos[1]+expected[0]
            y = pos[0] + expected[1]
            blank_grid[index][int(x)][int(y)] = data

        if index == 3:
            if pos != []:
                x = blank_grid.shape[1  ]-1-pos[0][1]+expected[0]
                y = pos[0][0]+expected[1]
                blank_grid[index][int(x)][int(y)] = 1

        return blank_grid

    def ghost_scared(self, blank_grid, index, state):
        if state.data.agentStates[1].scaredTimer == 0:
            blank_grid = self.scalar2array(blank_grid, index, state.getGhostPositions()[0])

        else:
            blank_grid = self.scalar2array(blank_grid, index+1, state.getGhostPositions()[0], data=1)
            #print(state.data.agentStates[1].scaredTimer)
        return blank_grid

    # def state2array(self, blank_grid, index, method, state):
    #     data = method.__str__()
    #     data = data.split(", ")[2]
    #     expected_pos = [0, 0]
    #     if data == "West":
    #         expected_pos = [0, -1]
    #     elif data == "East":
    #         expected_pos = [0, 1]
    #     elif data == "North":
    #         expected_pos = [-1, 0]
    #     elif data == "South":
    #         expected_pos = [1, 0]
    #
    #     if index == 4:
    #         blank_grid = self.scalar2array(blank_grid, 4, state.getPacmanPosition(), expected=expected_pos)
    #     elif index == 5:
    #         if state.data.agentStates[1].scaredTimer == 0:
    #             blank_grid = self.scalar2array(blank_grid, 5, state.getGhostPositions()[0], expected=expected_pos)
    #         else:
    #             blank_grid = self.scalar2array(blank_grid, 6, state.getGhostPositions()[0], expected = expected_pos)
    #     else:
    #         print("Something wrong here")
    #     return blank_grid

    def preprocess(self, state):
        # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.
        blank_grid = self.blank_grid(state, 6)
        blank_grid  = self.scalar2array(blank_grid, 0, state.getPacmanPosition())
        blank_grid  = self.grid2array(blank_grid, 1, state)
        blank_grid  = self.scalar2array(blank_grid, 3, state.getCapsules())
        blank_grid = self.ghost_scared(blank_grid, 4, state)
        #print(blank_grid)
        return torch.tensor(blank_grid)

