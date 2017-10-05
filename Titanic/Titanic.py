import numpy as np
import matplotlib.pyplot as plt
np.seterr(all='raise')

def initialize_parameters(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/float(layers_dims[l-1]),dtype = np.float128)
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1),dtype = np.float128)
    return parameters


def initialize_adam(parameters):
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def sigmoid(z):
    A = 1/(1+np.exp(-z))
    return A, A.copy()

def relu(z):
    A = z * (z > 0)
    return A, A.copy()

def sigmoid_backward(dA,cache):
    out = cache
    dx = dA*(out*(1-out))
    return dx

def relu_backward(dA,cache):
    dx, x = None, cache
    dx = np.array(dA, copy=True)
    dx[x <= 0] = 0
    return dx

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches += [cache]

    AL, cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches += [cache]
    return AL, caches

def compute_cost(AL, Y):
    m = float(Y.shape[1])
    x = 1-AL
    x[x == 0] = 1.0**-20
    cost = (-1/m)*(np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(x).T))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = float(A_prev.shape[1])

    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = np.sum(dZ,axis=1,keepdims=True)*(1/m)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    x = 1 - AL
    x[x == 0] = 1.0 ** -10
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, x))

    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],current_cache,"relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads["dW" + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads["db" + str(l + 1)], 2)

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
        v_corrected["dW" + str(l + 1)] / ((np.sqrt(s_corrected["dW" + str(l + 1)])) - epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
        v_corrected["db" + str(l + 1)] / ((np.sqrt(s_corrected["db" + str(l + 1)])) - epsilon))

    return parameters, v, s

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*grads["dW"+str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*grads["db"+str(l)]
    return parameters

def predict(X,Y,parameters):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    AL, caches = L_model_forward(X, parameters)
    for i in range(AL.shape[1]):
        Y_prediction[0, i] = 1 if AL[0, i] > 0.5 else 0
    return Y_prediction


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000,beta1 = .9,beta2 = .999,epsilon = 1e-8, print_cost=False,plot = False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)
    v, s = initialize_adam(parameters)
    t = 0
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X,parameters)

        cost = compute_cost(AL,Y)
    
        grads = L_model_backward(AL,Y,caches)

        t = t + 1  # Adam counter
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                       t, learning_rate, beta1, beta2, epsilon)
        #parameters = update_parameters(parameters,grads,learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    if plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters
if __name__ == "__main__":
    my_data = np.genfromtxt('train_edit.csv', delimiter=',')
    train_x = my_data.T[2:9][:]
    train_y = np.atleast_2d(my_data.T[1][:])
    m = train_x.shape[1]

    layers_dims = [7, 20, 7, 5, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims,learning_rate=.0009, num_iterations = 2500)

    pred_train = predict(train_x, train_y, parameters)
    correct = 0

    for i in range(m):
        if pred_train[0][i] == train_y[0][i]:
            correct += 1
    print("train: ",correct/float(m))

my_data_test = np.genfromtxt('test_edit.csv',delimiter=',')
test_x = np.atleast_2d(my_data_test.T[1:8][:])
test_ID = np.atleast_2d(my_data_test.T[0][:])
test_ID = test_ID.astype(int)
AL,caches = L_model_forward(test_x,parameters)
pred_test = np.zeros((1, AL.shape[1]))
for i in range(AL.shape[1]):
    pred_test[0, i] = 1 if AL[0, i] > 0.5 else 0
pred_test = pred_test.astype(int)

final = np.concatenate((test_ID.T.astype(int),pred_test.T.astype(int)),axis=1)

labels = [["PassengerID","Survived"]]
#final = np.concatenate((labels,final),axis=0)
print(final)
np.savetxt("pred_test.csv", final, delimiter=",",fmt='%d')