import numpy as np
import random
class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.learning_rate = 0.0271
        self.gamma = 0.9401
        self.e = 0.004102

    def select_action(self, state):
        """
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        rand = random.random()
        p_state = np.ones(self.n_actions)*self.e/self.n_actions

        #Eplison Greedy
        if rand<self.e:
            for i in range(self.n_actions):
                p_state[i] = random.random()
            sumofp = sum(p_state)
            for i in range(self.n_actions):
                p_state[i] /= sumofp
        #maxQ
        else:
            best_a = np.argmax(self.Q[state])
            p_state[best_a] += 1-self.e
            sumofp = sum(p_state)
            for i in range(self.n_actions):
                p_state[i] /= sumofp


        return np.random.choice(np.arange(self.n_actions), p = p_state)
        #return action

    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        #For constant, and for shortening sentence..
        gamma, lr = self.gamma, self.learning_rate

        next_action = self.select_action(next_state)
        #model free : compare action.
        grad_q = self.Q[next_state][next_action] - self.Q[state][action]
        self.Q[state][action] += lr*(reward+gamma*grad_q)

