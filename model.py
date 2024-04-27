import numpy as np
import json

# 保存和读取模型参数（json）
def save_model(model, model_name):
    for k,v in model.items():
        model[k] = v.tolist()
    with open(f'{model_name}.json', 'w') as f:
        json.dump(model, f)

def load_model(model_name):
    with open(f'{model_name}.json', 'r') as f:
        model = json.load(f)
    for k,v in model.items():
        model[k] = np.array(v)
    return model


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def compute_loss(y_true, y_pred, model, beta):
    weight_loss = beta / 2 * (np.sum(np.square(model['W1'])) + np.sum(np.square(model['W2'])) + np.sum(np.square(model['W3'])))
    loss = np.mean(-np.sum(y_true * np.log(y_pred), axis=1)) + weight_loss
    return loss

def init_model(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
    b2 = np.zeros(hidden_dim)
    W3 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b3 = np.zeros(output_dim)
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    return model


def forward_propagation(model, X, activation='relu'):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    z1 = np.dot(X, W1) + b1

    if activation == 'relu':
        a1 = relu(z1)
    elif activation == 'sigmoid':
        a1 = sigmoid(z1)
    else:
        raise ValueError("Unsupported activation function")

    z2 = np.dot(a1, W2) + b2

    if activation == 'relu':
        a2 = relu(z2)
    elif activation == 'sigmoid':
        a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    y_hat = softmax(z3)
    cache = {'a1': a1, 'a2': a2, 'z1': z1, 'z2': z2}
    return y_hat, cache

def backward_propagation(model, cache, X, y, y_hat, beta, activation='relu'):
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    a1, a2, z1, z2 = cache['a1'], cache['a2'], cache['z1'], cache['z2']

    dz3 = y_hat - y
    dW3 = np.dot(a2.T, dz3) + beta * W3
    db3 = np.sum(dz3, axis=0)

    da2 = np.dot(dz3, W3.T)
    if activation == 'relu':
        dz2 = np.where(a2 > 0, da2, 0)
    elif activation == 'sigmoid':
        dz2 = sigmoid_derivative(a2) * da2

    dW2 = np.dot(a1.T, dz2) + beta * W2
    db2 = np.sum(dz2, axis=0)

    da1 = np.dot(dz2, W2.T)
    if activation == 'relu':
        dz1 = np.where(a1 > 0, da1, 0)
    elif activation == 'sigmoid':
        dz1 = sigmoid_derivative(a1) * da1

    dW1 = np.dot(X.T, dz1) + beta * W1
    db1 = np.sum(dz1, axis=0)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

def update_model(model, grads, learning_rate):
    model['W1'] -= learning_rate * grads['W1']
    model['b1'] -= learning_rate * grads['b1']
    model['W2'] -= learning_rate * grads['W2']
    model['b2'] -= learning_rate * grads['b2']
    model['W3'] -= learning_rate * grads['W3']
    model['b3'] -= learning_rate * grads['b3']
    return model

def compute_accuracy(y_hat, y):
    return np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))

def one_hot(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1

    return one_hot_labels
