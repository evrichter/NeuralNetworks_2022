import numpy as np
from numpy.random import rand

np.random.seed(0)


class Network:
    def __init__(self, n_input, n_output, lr=0.01):
        self.n_input = n_input  # input size
        self.n_output = n_output  # output size

        self.lr = lr

        # Define weights and biases
        self.w1 = rand(self.n_input, self.n_output)
        self.b1 = rand(1)

        self.one_vector = np.ones((self.n_input,1))

    def forward(self, x):
        """To Do: implement me"""
        #done
        h=np.dot(self.w1,x)+self.b1*self.one_vector
        return h,self.sigmoid(h)

    def sigmoid(self, x):
        """To Do: implement me"""
        #done
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        """To Do: implement me"""
        #done
        return np.dot(self.sigmoid(x),(1-self.sigmoid(x)))

    def loss(self, y, y_hat):
        """To Do: implement me"""
        #done
        return 0.5*(np.linalg.norm(y_hat-y))**2

    def backward(self, x, y, h):
        """To Do: implement me"""
        gradient_w1 = np.zeros(self.w1.shape)
        gradient_b1 = 0
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                gradient_w1[i,j] = (self.sigmoid(h[i]) - y[i])*self.sigmoid_gradient(h[i])*x[j]
        for i in range(x.shape[0]):
            gradient_b1 += (self.sigmoid(h[i]) - y[i])*self.sigmoid_gradient(h[i])
      
        return gradient_w1, gradient_b1
