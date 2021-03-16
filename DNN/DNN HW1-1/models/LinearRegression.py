import numpy as np
import matplotlib.pyplot as plt

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
        mseloss = 0
        #b = np.random.rand(10, 1)
        loss_total = []
        for epoch in range(epochs):
            pick_rand_data = np.random.randint(0, len(x)-batch_size)

            x_batch = x[pick_rand_data: pick_rand_data + batch_size]
            y_batch = y[pick_rand_data: pick_rand_data + batch_size]
            y_batch = np.asarray(y_batch).reshape(-1, 1)


            y_pred = np.matmul(x_batch, self.W)
            m_grad = -2*np.matmul(x, np.matmul(self.W, x))

            print("shape of gradient m is ", m_grad.shape)

            #w = optim(w=w, grad=m_grad, lr=lr)

            self.W = self.W-m_grad*lr


        #For Check.
        plt.plot(loss_total)
        plt.show()
        final_loss = mseloss




        # ============================================================

        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # ========================= EDIT HERE ========================
        y_predicted = np.matmul(x, self.W)
        # ============================================================
        return y_predicted
