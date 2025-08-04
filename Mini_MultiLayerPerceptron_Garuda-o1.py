import numpy as np

def relu(x):
   return np.maximum(0, x)

x = np.array([[0.4, 0.7], [0.3, 0.1]])

w1 = np.array([[0.3, -0.2], [0.5, 0.6]])
b1 = np.array([0.0, 0.0])
a1 = np.dot(x, w1) + b1
z1 = relu(a1)

w2 = np.array([[-0.5, 0.1], [-0.1, 0.8]])
b2 = np.array([0.0, 0.0])
z2 = np.dot(z1, w2) + b2

print("Output MLP =", z2)