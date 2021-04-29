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
LEARNING_STARTS = 1000	    # 5000 스텝 이후 training 시작
TARGET_UPDATE_ITER = 1000  # update target network

EPSILON_START = 0.99
EPSILON_END = 0.01
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ConvBlock(nn.Sequential):
    def __init__(self, inchannel, outchannel, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__(*[nn.Conv2d(inchannel, outchannel, kernel_size, stride=stride),
                                          #nn.BatchNorm2d(outchannel),
                                          nn.ReLU(inplace=True)
                                          ])


class DQN(torch.nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(*[ConvBlock(6, 7, 3), ConvBlock(7, 7, 3), ConvBlock(7,1,3)])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16, 8)
        #self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, outputs)

    def forward(self, x):
        z = self.conv_layers(x)
        z = self.pool(z)
        z = self.flatten(z)
        z = self.fc1(z)
        z = F.relu(z)
        # z = self.fc2(z)
        # z = F.relu(z)
        z = self.fc3(z)
        #return x.view(x.size(0), -1)
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

    def __len__(self):
        return len(self.memory)

    def print(self):
        print(self.memory[self.position])


class PacmanDQN(PacmanUtils):
    def __init__(self, args):
        super().__init__()
        print("Started Pacman DQN algorithm")
        for k in args.keys():
            setattr(self, k, args[k])

        self.losses = []
        self.memory = ReplayMemory(REPLAY_MEMORY)

        if self.trained_model:
            mode = "Test trained model"
        else:
            mode = "Training model"
            self.policy_network = DQN(4)
            self.target_network = DQN(4) #TODO: optimizer place needs to be replaced.
            self.optimizer = optim.RMSprop(self.policy_network.parameters())

        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}     Model : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode, args['model']))
        print("=" * 100)

        if self.trained_model:  # Test
            self.target_network = torch.load(self.model)
            self.epsilon = 0
        else:                   # Train
            self.epsilon = EPSILON_START  # epsilon init value

        # statistics
        self.win_counter = 0       # number of victory episodes
        self.steps_taken = 0       # steps taken across episodes
        self.steps_per_epi = 0     # steps taken in one episodes
        self.episode_number = 0
        self.episode_rewards =[]
        self.batch_size = 128

        self.epsilon = EPSILON_START  # epsilon init value
        self.action = None
        self.state = None
        self.num_action = 4

    def predict(self, state):

        if np.random.random() < self.epsilon:
            with torch.no_grad():
                state = self.preprocess(state)
                state = state.unsqueeze(0)
                state = state.float()
                #print(state)
                result = self.target_network(state)
                # print(result, result.max(1)[1].view(1,1))
                self.action = result.max(1)[1].view(1,1)
                if self.steps_taken%1000==0:
                    print(self.action)
        else:
            self.action = torch.tensor([np.random.randint(0, 4)]).long()
        return self.action


    def update_epsilon(self):

        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_taken / 10000)

        return

    def step(self, next_state, reward, done):
        # next_state = self.state에 self.action 을 적용하여 나온 state
        # reward = self.state에 self.action을 적용하여 얻어낸 점수.
        if self.action is None: #First.

            self.state = self.preprocess(next_state)

        else:
            self.next_state = self.preprocess(next_state)
            self.memory.write(self.state, self.action, self.next_state, reward, done)
            self.state = self.next_state

        # next
        self.episode_reward += reward
        self.steps_taken += 1
        self.steps_per_epi += 1
        self.update_epsilon()

        if not self.trained_model:
            self.train()
            self.update_epsilon()
            if self.steps_taken % TARGET_UPDATE_ITER == 0:
                print("model updated")
                self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self):
        if self.steps_taken > LEARNING_STARTS:
            # Get data from Memory by random sampling
            transitions = self.memory.sample(self.batch_size)
            # make zip file using namedtuple, which is faster than for loop
            batched = Transition(*zip(*transitions))
            # As data needs to be reshaped by batch size, concat with reshaping
            state_batch = torch.cat(batched.state).reshape(self.batch_size, -1, 7, 7).float()
            # Due to some reason, concat does not work, so I just reshaped.
            # For action_batch, there's no need to reshape
            # Because it will be used by index.
            action_batch = torch.tensor(batched.action)
            # reward reshaped
            reward_batch = torch.tensor(batched.reward).reshape(-1, 1).float()

            num_done = torch.tensor(batched.done).reshape(-1, 1)
            #As Target Network, stop grading.
            with torch.no_grad():
                #Predict next q using target network
                next_q = self.target_network(state_batch)
                next_q, _ = next_q.max(dim=1)
                next_q = next_q.reshape(-1, 1).float()
                # target_q = next_q * 0.99 * num_done + reward_batch
                target_q = next_q * 0.99*num_done + reward_batch

            # calculate current q using policy network
            current_q = self.policy_network(state_batch)
            # calculate best q on action on every batch.
            # Size needs to be batch * 1
            policy_set = torch.tensor(np.zeros((len(current_q), 1))).float()
            for i in range(len(current_q)):
                policy_set[i]=current_q[i][action_batch[i].long()]
            if self.steps_taken%1000==0:
                print(policy_set[32:41],target_q[32:41])

            loss = F.smooth_l1_loss(policy_set, target_q) #TODO: Might Be Broadcasted.
            self.losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

        if self.episode_number % 500 == 0:
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number,
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon))

        if self.episode_number%100==0:
            print("Episode no = {:>5};, recent average loss {}".format(self.episode_number, np.mean(self.losses[-100:-1])))
        self.win_counter = 0
        self.episode_rewards= []
        if self.trained_model==False and self.episode_number%1000 == 0:
            torch.save(self.target_network, self.model)


    def blank_grid(self, state, N):
        width, height = state.data.layout.width, state.data.layout.height
        grid = np.zeros((N, width, height))
        return grid

    def grid2array(self, blank_grid, index, state):
        for i in range(7):
            for j in range(7):
                blank_grid[index][i][j] =state.data.layout.walls[j][6-i]
                blank_grid[index+1][i][j]=state.data.layout.food[j][6-i]
        return blank_grid

    def scalar2array(self, blank_grid, index, method, expected=None):
        if expected is None:
            expected = [0, 0]

        pos = method

        if index != 3 : #For Capsule
            x = blank_grid.shape[1]-1-pos[1]+expected[0]
            y = pos[0] + expected[1]
            blank_grid[index][int(x)][int(y)] = 1

        if index == 3:
            if pos != []:
                x = blank_grid.shape[1]-1-pos[0][1]+expected[0]
                y = pos[0][0]+expected[1]
                blank_grid[index][int(x)][int(y)] = 1

        return blank_grid

    def ghost_scared(self, blank_grid, index, state):
        if state.data.agentStates[1].scaredTimer == 0:
            blank_grid = self.scalar2array(blank_grid, index, state.getGhostPositions()[0])
        else:
            blank_grid = self.scalar2array(blank_grid, index+1, state.getGhostPositions()[0])
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
        #blank_grid  = self.state2array(blank_grid, 4, state.data.agentStates[0], state)
        #blank_grid  = self.state2array(blank_grid, 5, state.data.agentStates[1], state)

        #print(blank_grid[0])
        #print(blank_grid[4])
        #print(blank_grid[7])
        #print(blank_grid[5])
        #print(blank_grid[8])
        #print("==========================")
            #time.sleep(2)
        return torch.tensor(blank_grid)

#9001->1560...?
