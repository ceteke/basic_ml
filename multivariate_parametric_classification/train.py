import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import math
class Gaussian:
    def __init__(self, mean_vector,  covariance_matrix, dimension, class_prior=None):
        '''
        :param mean_vector: numpy column vector. ie. numpy matrix with shape [d,1]
        :param covariance_matrix: numpy matrix with shape [d, d]
        :param dimension: dimension (ie feature size)
        '''
        self.dimension = dimension
        self.mean_vector = mean_vector
        self.covariance_matrix = np.asmatrix(covariance_matrix)
        self.class_prior = class_prior

        self.__first_term = 1.0 / (
            math.pow(2*math.pi, self.dimension/2) *
            math.sqrt(np.linalg.norm(self.covariance_matrix))
        ) # Get rid of unnecessary computation

        self.__cov_inv = np.linalg.inv(self.covariance_matrix)

    def get_discriminant(self, x):
        g = -(self.dimension/(2*math.log(2*math.pi)) + 1/2*(math.log(np.linalg.norm(self.covariance_matrix))))
        mid = float(np.matmul(np.matmul((x-self.mean_vector).T, np.linalg.inv(self.covariance_matrix)), x-self.mean_vector))
        mid = -1/2*mid
        g = g + mid + math.log(self.class_prior)
        return g

    def __call__(self, x):

        diff = x - self.mean_vector
        exp_term = math.exp(float(
            np.matmul(np.matmul(diff, self.__cov_inv), diff.T)))

        result = self.__first_term * exp_term

        return result

    def sample(self, N):
        return np.random.multivariate_normal(self.mean_vector, self.covariance_matrix, N)

class GaussianEstimator:
    def __init__(self, dimension, total_samples):
        self.dimension = dimension
        self.total_samples = total_samples
        self.estimated_gaussians = []

    def fit(self, X, y):
        classes = {}

        for x, i in zip(X, y):
            if i in classes:
                class_data = classes[i]
                classes[i]  = np.concatenate((class_data, [x]))
            else:
                classes[i] = np.array([x])

        for c in classes.keys():
            class_samples = classes[c]
            mean = self._estimate_mean(class_samples)
            covariance = self._estimate_covariance(class_samples, mean)
            class_prior = class_samples.shape[0] / self.total_samples
            gaussian = Gaussian(mean, covariance, self.dimension, class_prior)
            self.estimated_gaussians.append(gaussian)

    def get_mean(self):
        return np.concatenate([[g.mean_vector] for g in self.estimated_gaussians])

    def print_covariances(self):
        for i, g in enumerate(self.estimated_gaussians):
            print(i + 1)
            print(g.covariance_matrix)

    def predict(self, x):
        discriminants = []
        for g in self.estimated_gaussians:
            discriminants.append(g.get_discriminant(x))
        pred_class = np.argmax(discriminants)
        return pred_class

    def predict_multiple(self, X):
        predicted = []
        for x in X:
            pred = self.predict(x)
            predicted.append(pred)
        return np.array(predicted)

    def _estimate_mean(self, X):
        return np.mean(X, axis=0)

    def _estimate_covariance(self, X, mean):
        # Naive implementation
        N = X.shape[0]
        s = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                sum = 0.0
                for t in range(N):
                    sum += (X[t][i] - mean[i]) * (X[t][j] - mean[j])
                sum /= N
                s.append(sum)

        s = np.array(s).reshape(self.dimension, self.dimension)
        return s


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

g1 = Gaussian(mean_vector1, covariance_matrix1, 2)
g2 = Gaussian(mean_vector2, covariance_matrix2, 2)
g3 = Gaussian(mean_vector3, covariance_matrix3, 2)

x1 = g1.sample(N1)
x2 = g2.sample(N2)
x3 = g3.sample(N3)

y1 = [0] * N1
y2 = [1] * N2
y3 = [2] * N3

total_samples = N1 + N2 + N3

X = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))

estimator = GaussianEstimator(2, total_samples)
estimator.fit(X, y)

count = 0
print("Mean vectors")
print(estimator.get_mean())

print("Covariance matrices:")
estimator.print_covariances()

confusion_matrix = np.zeros([3, 3])
misclassified = []
classified = []
for i, x in enumerate(X):
    pred_class = estimator.predict(x)
    confusion_matrix[pred_class][y[i]] += 1
    if pred_class == y[i]:
        classified.append(x)
        count += 1
    else:
        misclassified.append(x)

# Decision boundaries

x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = estimator.predict_multiple(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, 25, alpha=0.4, cmap="RdYlBu")

for c in classified:
    if c in x1:
        color = 'r'
    elif c in x2:
        color = 'y'
    else:
        color = 'b'

    plt.scatter(c[0], c[1], c=color)

for mc in misclassified:
    if mc in x1:
        color = 'r'
    elif mc in x2:
        color = 'y'
    else:
        color = 'b'

    plt.scatter(mc[0], mc[1], c=color, edgecolor='black')

print("Confusion matrix:\npredicted/actual")
print(confusion_matrix)
print("Accuracy:", count/total_samples)

plt.show()
