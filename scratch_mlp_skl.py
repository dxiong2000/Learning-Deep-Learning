from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r


# convert data from integer to a vector. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
def convert_y_to_vector(y):
    vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        vect[i, y[i]] = 1
    return vect


# sigmoid activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))


# initializes W and b matrices to a random distribution
def initialize_weights_bias(nn_structure):
    W = {}
    b = {}

    # W^l = len(layer l+1) by len(layer l) matrix
    # b^l = len(layer l+1) vector
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l], ))
    return W, b


# initializes dW and db matrices to zeros
def initialize_weight_bias_delta(nn_structure):
    delta_W = {}
    delta_b = {}
    for l in range(1, len(nn_structure)):
        delta_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        delta_b[l] = np.zeros(nn_structure[l])
    return delta_W, delta_b


# feed forward function, x is input, W is weight matrix, b is bias matrix
def feed_forward(x, W, b):
    # h = node outputs
    h = {1: x}
    z = {}

    # for each layer in nn
    for l in range(1, len(W) + 1):
        # if it is the first hidden layer, then the node input is x
        # otherwise, the node input is the output h[l] (output from previous layer)
        if l == 1:
            node_in = x
        else:
            node_in = h[l]

        z[l+1] = W[l].dot(node_in) + b[l]
        h[l+1] = sigmoid(z[l+1])
    return h, z


# calculates the delta value for output layer
# delta^n = -(y - h)*f'(z)
def calc_output_delta(y, h_out, z_out):
    return -1*(y - h_out) * sigmoid_deriv(z_out)


# calculates delta values for hidden layers
# delta^l = delta^(l+1) * W^l * f'(z^l)
def calc_hidden_delta(delta_plus_one, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_one) * sigmoid_deriv(z_l)


def train_nn(nn_structure, x, y, iter_num=3000, step_size=0.25):
    W, b = initialize_weights_bias(nn_structure)
    iter_count = 0
    m_samples = len(y)
    # for visualizing the change in error
    avg_cost_func = []

    # runs for iter_num iterations
    while iter_count < iter_num:
        # prints iteration number every 1000
        if iter_count%500 == 0:
            print("iteration {} of {}".format(iter_count, iter_num))

        delta_W, delta_b = initialize_weight_bias_delta(nn_structure)
        avg_cost = 0

        # for samples in training data
        for i in range(len(y)):
            # delta values
            delta = {}
            # feed forward pass
            h, z = feed_forward(x[i, :], W, b)

            # back propagation from n_l to 1
            for l in range(len(nn_structure), 0, -1):
                # if outer layer, calculate the outer layer delta value
                if l == len(nn_structure):
                    delta[l] = calc_output_delta(y[i, :], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i, :] - h[l]))
                else:
                    # if not the first hidden layer, calculate the delta value based on prior layers
                    # else, the delta value will not need to be stored
                    if l > 1:
                        delta[l] = calc_hidden_delta(delta[l+1], W[l], z[l])

                    delta_W[l] += np.dot(delta[l+1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
                    delta_b[l] += delta[l+1]

        # gradient descent
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -step_size * (1.0/m_samples * delta_W[l])
            b[l] += -step_size * (1.0/m_samples * delta_b[l])

        avg_cost = 1.0/m_samples * avg_cost
        avg_cost_func.append(avg_cost)
        iter_count += 1

    return W, b, avg_cost_func


# test trained model
def test_nn(W, b, x_test, y_test, n_layers):
    testcases = x_test.shape[0]
    outputs = np.zeros((testcases, ))

    # runs feed forward over all testcases
    for i in range(testcases):
        h, z = feed_forward(x_test[i, :], W, b)
        # uses argmax to find the digit with the highest probability
        outputs[i] = np.argmax(h[n_layers])

    # prints accuracy
    print(accuracy_score(y_test, outputs) * 100)


def main():
    # loads digits
    digits = load_digits()

    # scales down to [-2, 2]
    x_scale = StandardScaler()
    x = x_scale.fit_transform(digits.data)

    # loads target values
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    y_train_vectors = convert_y_to_vector(y_train)

    # Set MLP structure to 64 node input layer, 30 node hidden layer, and 10 node output layer
    nn_structure = [64, 30, 10]

    # Train networks. Learning rates of 0.25 and 0.5
    W, b, avg_cost_func = train_nn(nn_structure, x_train, y_train_vectors, iter_num=5000, step_size=0.25)
    W1, b1, avg_cost_func1 = train_nn(nn_structure, x_train, y_train_vectors, iter_num=5000, step_size=0.5)

    '''
    # Save model into respective files
    np.save("w.npy", W)
    np.save("b.npy", b)
    np.save("w1.npy", W1)
    np.save("b1.npy", b1)
    np.save("cost.npy", avg_cost_func)
    np.save("cost1.npy", avg_cost_func1)
    
    # Load model into respective vars
    W = np.load("w.npy", allow_pickle=True).item()
    b = np.load("b.npy", allow_pickle=True).item()
    W1 = np.load("w1.npy", allow_pickle=True).item()
    b1 = np.load("b1.npy", allow_pickle=True).item()
    avg_cost_func = np.load("cost.npy", allow_pickle=True)
    avg_cost_func1 = np.load("cost1.npy", allow_pickle=True)
    '''

    # test trained model
    test_nn(W, b, x_test, y_test, n_layers=3)
    test_nn(W1, b1, x_test, y_test, n_layers=3)

    # plot learning rate 0.25 and 0.5 models
    plt.plot(avg_cost_func, label="0.25")
    plt.ylabel('Average Cost')
    plt.xlabel('Iteration number')
    plt.plot(avg_cost_func1, label="0.5")
    plt.ylabel('Average Cost')
    plt.xlabel('Iteration number')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
