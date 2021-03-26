import numpy as np
import random
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        total_loss = []
        total_acc = []
        for epoch in range(epochs):
            loss = 0
            acc = 0
            #DATA SHUFFLING=====================


            tmp = [[x_t, y_t] for x_t, y_t in zip(x, y)]
            random.shuffle(tmp)
            x = [x[0] for x in tmp]
            y = [x[1] for x in tmp]
            x = np.asarray(x)
            y = np.asarray(y)

            #TRAIN==================================

            for i in range(0, x.shape[0]-batch_size+1, batch_size):

                x_batch = x[i: i + batch_size]
                y_batch = y[i: i + batch_size]
                y_batch = np.asarray(y_batch).reshape(batch_size, 1)

                sig_y_pred = self.forward(x_batch)
                diff = sig_y_pred - y_batch
                diff = diff.reshape(-1, 1)
                m_grad = np.matmul(x_batch.transpose(), diff)
                self.W = optim.update(self.W, m_grad, lr)
                loss += (-y_batch * np.log(sig_y_pred + epsilon) - (1 - y_batch) * np.log
                (1 - sig_y_pred + epsilon)).mean()
                for i in range(len(y_batch)):
                    if y_batch[i] == sig_y_pred[i]:
                        acc +=1

            loss /= x.shape[0]
            acc /= x.shape[0]
            total_loss.append(loss)
            total_acc.append(acc)
        label = [(i+1)*30 for i in range(10)]
        total_acc_resize = [total_acc[30*i] for i in range(10)]
        total_loss_resize = [total_loss[30*i] for i in range(10)]
        plt.plot(label, total_loss_resize)
        plt.title("minibatch==10")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xticks(range(0, 301, 30))
        plt.show()
        plt.title("minibatch==10")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.xticks(range(0, 301, 30))
        plt.plot(label, total_acc_resize)
        plt.show()

        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================

        y_predicted = LogisticRegression._sigmoid(self, np.matmul(x,self.W))

        for i in range(len(y_predicted)):
            if y_predicted[i]> threshold:
                y_predicted[i] = 1
            else:
                y_predicted[i] = 0
        #print(y_predicted)


        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid  = 1 / (1 + np.exp(-x))
        # ============================================================
        return sigmoid
