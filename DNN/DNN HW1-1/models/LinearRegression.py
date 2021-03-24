import numpy as np
import matplotlib.pyplot as plt
import random

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================

        loss_total = []

        for epoch in range(epochs):
            mseloss = 0
            cnt = 0
            tmp = [[x_t, y_t] for x_t, y_t in zip(x, y)]
            random.shuffle(tmp)
            x = [x[0] for x in tmp]
            y = [x[1] for x in tmp]
            x = np.asarray(x)
            y = np.asarray(y)

            for i in range(0, x.shape[0]-batch_size, batch_size):

                x_batch = x[i: i + batch_size]
                y_batch = y[i: i + batch_size]
                y_batch = np.asarray(y_batch).reshape(10, 1)
                y_pred = np.matmul(x_batch, self.W)
                m_grad = -1 / batch_size * sum(np.dot(x_batch.transpose(), (y_batch-y_pred)))
                mseloss += np.mean(np.square(y_batch-y_pred))
                self.W = optim.update(self.W, m_grad, lr)
                cnt += 1
            mseloss = mseloss/cnt
            print(epoch, "in loss", mseloss)
            loss_total.append(mseloss)

        # ============================================================

        return mseloss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # ========================= EDIT HERE ========================
        y_predicted = np.matmul(x, self.W)
        # ============================================================
        return y_predicted
