import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

if __name__ == '__main__':
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([0, 1, 1, 0])  # output as row vector

    lr = 3  #learning rate
    epochs = 1000

    n_x = 2 #features in input
    m = 4   #training samples
    neurons_l1 = 2  #neurons in first layer
    neurons_l2 = 1  #neurons in output layer

    W1 = np.random.random((n_x, neurons_l1))
    B1 = np.zeros((neurons_l1, 1))

    W2 = np.random.random((neurons_l1, neurons_l2))
    B2 = np.zeros((neurons_l2, 1))

    for i in range(epochs):
        #forward pass
        Z1 = np.dot(W1.T, X) + B1
        A1 = sigmoid(Z1)

        Z2 = np.dot(W2.T, A1) + B2
        A2 = sigmoid(Z2)

        #backward pass
        DZ2 = A2 - Y        #using cross-entropy loss
        DW2 = (1/m)*np.dot(A1, DZ2.T)
        DB2 = (1/m)*np.sum(DZ2)

        W2 = W2 - lr*DW2
        B2 = B2 - lr*DB2

        # backward pass for the first layer
        DZ1 = np.dot(W2, DZ2) * A1 * (1 - A1)
        DW1 = (1/m) * np.dot(X, DZ1.T)
        DB1 = (1/m) * np.sum(DZ1)

        W1 = W1 - lr*DW1
        B1 = B1 - lr*DB1

        if i%100 == 0:
            logloss = np.sum(-(Y*(np.log(A2)) + (1 - Y) * (np.log(1 - A2)))) #log loss
            print('epoch = ', i)
            print('loss = ', logloss)

print('\nXOR neural network final predictions after rounding:\n', np.concatenate((X, np.round(A2))).T)