import numpy as np
import csv
from PIL import Image as im
from matplotlib import pyplot as plt


class FullyConnected:
    def __init__(self, sizein, sizeout):
        self.grad = None
        self.gradw = None
        self.gradb = None
        self.data_in = None
        self.sw, self.rw, self.sb, self.rb = 0, 0, 0, 0
        self.p_o = 0.9
        self.p_t = 0.999
        self.delta = 0.00000001
        rng = np.random.default_rng(seed=0)
        self.__weights = 0.0001 * (rng.random([sizein, sizeout]) - 0.5)
        self.__biases = 0.0001 * (rng.random([1, sizeout]) - 0.5)

    def forwardpropagate(self, data_in):
        self.data_in = data_in
        self.out = (data_in @ self.__weights) + self.__biases
        return self.out

    def backpropagation(self, grad, lr, epoch, flag):

        if flag == 1:
            self.gradw = np.transpose(self.data_in) @ grad
            self.gradb = np.transpose(np.ones([self.data_in.shape[0], 1])) @ grad
            self.grad = grad @ np.transpose(self.__weights)
            # ADAM ALGORITHM TO UPDATE WEIGHT
            self.sw = (self.p_o * self.sw) + (1 - self.p_o) * self.gradw
            self.rw = (self.p_t * self.rw) + (1 - self.p_t) * (self.gradw * self.gradw)
            self.__weights = self.__weights - lr * (
                    (self.sw / (1 - pow(self.p_o, epoch))) / (
                    np.sqrt(self.rw / (1 - pow(self.p_t, epoch))) + self.delta))

            # ADAM ALGORITHM TO UPDATE BIAS
            self.sb = (self.p_o * self.sb) + (1 - self.p_o) * self.gradb
            self.rb = (self.p_t * self.rb) + (1 - self.p_t) * (self.gradb * self.gradb)
            self.__biases = self.__biases - lr * (
                    (self.sb / (1 - pow(self.p_o, epoch))) / (
                    np.sqrt(self.rb / (1 - pow(self.p_t, epoch))) + self.delta))

        if flag == 0:
            self.grad = grad @ np.transpose(self.__weights)

        return self.grad


class Relu:
    def __init__(self):
        self.out = None
        self.grad = None

    def forwardpropagate(self, h):
        self.out = np.maximum(0, h)
        return self.out

    def gradient(self):
        self.grad = np.where(self.out > 0, 1, 0)
        return self.grad

    def backpropagation(self, grad):
        self.grad = grad * self.grad
        return self.grad


class Sigmoid:
    def __init__(self):
        self.out = None
        self.grad = None

    def forwardpropagate(self, h):
        self.out = 1 / (1 + np.exp(np.negative(h)))
        return self.out

    def gradient(self):
        self.grad = self.out * (np.subtract(1, self.out))
        return self.grad

    def backpropagation(self, grad):
        self.grad = grad * self.grad
        return self.grad


class LogLoss:
    def __init__(self, y):
        self.y = y
        self.yhat = None

    def forwardpropagate(self, yhat):
        self.yhat = yhat
        return self.yhat

    def eval(self):
        self.l_l = -(self.y * (np.log(self.yhat + np.finfo(float).eps)) + ((1 - self.y) * (
            np.log(1 - (self.yhat + np.finfo(float).eps)))))
        return np.mean(self.l_l)

    def gradient(self):
        grad = - (np.divide(self.y, self.yhat) - np.divide(1 - self.y, 1 - self.yhat))
        return grad


class Generative_Af:
    def __init__(self, yhat):
        self.yhat = yhat

    def eval(self):
        self.l_l = -(np.log(self.yhat + np.finfo(float).eps))
        return np.mean(self.l_l)

    def gradient(self):
        grad = - (np.divide(1, self.yhat + np.finfo(float).eps))
        return grad


class loadFile:
    def load(self, x):
        with open(x, 'r') as f:
            x = list(csv.reader(f, delimiter=","))
        train_dataset = np.array(x, dtype='float32')
        return train_dataset


class Splitdata:
    def split_f(self, x):
        data_X = x[:, 1:]
        data_Y = np.empty([x.shape[0], 1])
        data_Y[:, 0] = x[:, 0]
        return data_X.astype(float), data_Y.astype(float)


if __name__ == "__main__":
    # ----------------------------------------------- TRAINING SET -----------------------------------------------------
    # loading the csv file
    h = loadFile().load('mnist_2/mnist_train.csv')
    # Extracting only the rows and columns of a specific number
    h_0 = h[h[:, 0] == 9, :]
    # Splitting the data
    h_0_x, h_0_y = Splitdata().split_f(h_0)
    # Generating the random values between 0 and 255
    i = np.random.uniform(0, 255, h_0_x.shape)
    # Extracting 50 of the random and the real values
    i = i[:50, :]
    j = h_0_x[:50, :]
    # Creating the fully connected layer for the generator
    fully_g = FullyConnected(h_0_x.shape[1], h_0_x.shape[1])
    # Creating the Relu Function Layer
    r = Relu()
    # Creating the fully connected layer for the discriminator
    fully_d = FullyConnected(h_0_x.shape[1], 1)
    # Creating the Sigmoid Function Layer
    s = Sigmoid()
    # Setting up target values
    y_t = np.empty([(j.shape[0] + i.shape[0]), 1])
    y_t[0:j.shape[0], 0], y_t[j.shape[0]:, 0] = 1, 0
    # Creating Log Loss for discriminator
    log_loss = LogLoss(y_t)
    log_eval = []
    gen_obj_eval = []
    epoch = 1
    max_index_col = 0
    count = 0
    while epoch < 25:
        # GENERATOR
        fully_one = fully_g.forwardpropagate(i)
        relu_one = r.forwardpropagate(fully_one)
        array = np.reshape(relu_one[max_index_col, :], (28, 28))
        data = im.fromarray(array)
        data = data.convert('L')
        data.save(f'image_one/{count}.png')
        count += 1

        # DISCRIMINATOR
        input_d = np.concatenate((j, relu_one), axis=0)
        fully_two = fully_d.forwardpropagate(input_d)
        sig_one = s.forwardpropagate(fully_two)
        log_one = log_loss.forwardpropagate(sig_one)
        log_eval.append(log_loss.eval())

        # GRADIENT FOR DISCRIMINATOR
        log_grad = log_loss.gradient()
        sig_grad = s.gradient()

        # BACKPROPAGATION FOR DISCRIMINATOR
        sig_back = s.backpropagation(log_grad)
        fully_two_back = fully_d.backpropagation(sig_back, 0.0001, epoch, 1)

        # FORWARD PASSING FAKE DATA THROUGH THE ENTIRE ARCH
        fully_two_g = fully_d.forwardpropagate(relu_one)
        sig_one_g = s.forwardpropagate(fully_two_g)
        max_index_col = np.argmax(sig_one_g, axis=0)

        gen_obj = Generative_Af(sig_one_g)
        gen_obj_eval.append(gen_obj.eval())

        # GRADIENT FOR GENERATOR
        gen_obj_grad = gen_obj.gradient()
        sig_grad_g = s.gradient()
        relu_grad = r.gradient()

        # BACKPROPAGATION FOR GENERATOR
        sig_back_g = s.backpropagation(gen_obj_grad)
        fully_two_g_back = fully_d.backpropagation(sig_back_g, 0.0001, epoch, 0)
        r_back = r.backpropagation(fully_two_g_back)
        fully_one_back = fully_g.backpropagation(r_back, 0.0001, epoch, 1)

        epoch += 1

    epochi = []
    for i in range(len(log_eval)):
        epochi.append(i)

    plt.plot(epochi, log_eval)
    plt.plot(epochi, gen_obj_eval)
    plt.xlabel('EPOCH')
    plt.ylabel('Objective Function (J)')
    plt.legend(['Discriminator', 'Generator'])
    plt.show()
