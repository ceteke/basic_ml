import math
import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.weights = np.random.normal(0, 1e-3, [2, 3])
        self.bias = np.zeros(3)
        self.learning_rate = 0.01

    def _softmax(self, x):
        return np.exp(x) / (np.sum(np.exp(x)))

    def forward(self, x):
        return self._softmax(np.matmul(x, self.weights) + self.bias)

    def predict(self, x):
        return np.argmax(np.matmul(x, self.weights) + self.bias, axis=1)

    def train(self, x, y):
        N = x.shape[0]
        total_loss = 0.0
        dW = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        for t in range(N):
            y_hat = self.forward(x[t])
            total_loss -= np.dot(y[t], np.log(y_hat))

            delta = y[t] - y_hat
            dW += np.matmul(x[t].reshape([1,-1]).T, delta.reshape([1,-1]))
            db += delta

        self.weights += self.learning_rate * dW
        self.bias += self.learning_rate * db
        return total_loss


N1 = 100
N2 = 100
N3 = 100

mean_vector1 = [0.0, 1.5]

mean_vector2 = [-2.5, -3.0]

mean_vector3 = [2.5, -3.0]

covariance_matrix1 = np.array([[1.0, 0.2],
                               [0.2, 3.2]])

covariance_matrix2 = np.array([[1.6, -0.8],
                               [-0.8, 1.0]])

covariance_matrix3 = np.array([[1.6, 0.8],
                               [0.8, 1.0]])

x1 = np.random.multivariate_normal(mean_vector1, covariance_matrix1, N1)
x2 = np.random.multivariate_normal(mean_vector2, covariance_matrix2, N2)
x3 = np.random.multivariate_normal(mean_vector3, covariance_matrix3, N3)

X = np.concatenate((x1,x2,x3))

# One-hot representation
y1 = np.concatenate([[[1,0,0]] for _ in range(N1)])
y2 = np.concatenate([[[0,1,0]] for _ in range(N2)])
y3 = np.concatenate([[[0,0,1]] for _ in range(N3)])

y = np.concatenate((y1,y2,y3))

model = Model()

errors = []
learning_rate = 1e-3

epochs = 1000
losses = []
for _ in range(epochs):
    losses.append(model.train(X, y))

print("Weights\n{}".format(model.weights))
print("Biases\n{}".format(model.bias))

plt.figure(1)
plt.subplot(211, title='Error vs Iteration')
plt.plot(losses)

y_labels = np.concatenate((np.zeros(N1),np.ones(N1),np.ones(N1)*2)).astype(np.int8)

confusion_matrix = np.zeros([3, 3])
misclassified = []
classified = []
count = 0

for i, x in enumerate(X):
    pred_class = model.predict(x.reshape(1,-1))[0]
    confusion_matrix[pred_class][y_labels[i]] += 1
    if pred_class == y_labels[i]:
        classified.append(x)
        count += 1
    else:
        misclassified.append(x)

# Decision boundaries

x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(212, title="Decision Boundaries")

plt.contourf(xx, yy, Z, 2, alpha=0.4, cmap=plt.cm.coolwarm)

for c in classified:
    if c in x1:
        color = 'b'
    elif c in x2:
        color = 'silver'
    else:
        color = 'r'

    plt.scatter(c[0], c[1], c=color)

for mc in misclassified:
    if mc in x1:
        color = 'b'
    elif mc in x2:
        color = 'silver'
    else:
        color = 'r'

    plt.scatter(mc[0], mc[1], c=color, edgecolor='black')

print("Confusion matrix:\npredicted/actual")
print(confusion_matrix)
print("Accuracy:", count/(N1 + N2 + N3))

plt.show()


