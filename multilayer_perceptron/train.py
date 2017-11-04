import numpy as np
import matplotlib.pyplot as pyplt

class Data:
    def __init__(self, data_str):
        self.data_str = data_str
        self.num_sample = 0
        self.dataset = {}
        self.colors = ['b', 'g', 'r', 'm']
        self._parse_data_str(data_str)

    def _parse_data_str(self, data_str):
        for i, dt in enumerate(data_str.split(';')):
            if len(dt) == 0: break
            cls = dt.split('$')
            mean_vector = np.array(cls[0].split(',')).astype(np.float)
            covariance = np.array([cls[1].split(','), list(reversed(cls[1].split(',')))]).astype(np.float)
            num = int(cls[2])
            class_id = int(cls[3])
            self.num_sample += num

            x = np.random.multivariate_normal(mean_vector, covariance, num)
            if class_id-1 in self.dataset:
                self.dataset[class_id-1] = np.concatenate([self.dataset[class_id-1], x])
            else:
                self.dataset[class_id-1] = x

    def plot_training(self, plt):
        for i in self.dataset.keys():
            plt.scatter(self.dataset[i][:,0],self.dataset[i][:,1],c=self.colors[i])

    def dump(self):
        X = np.concatenate([self.dataset[i] for i in self.dataset.keys()])
        y = []

        for i in self.dataset.keys():
            one_hot = [0] * len(self.dataset)
            one_hot[i] = 1
            y.append(np.repeat([one_hot], len(self.dataset[i]), axis=0))

        y = np.concatenate(y)

        return X, y

    def get_training_data(self, mini_batch=16, shuffle=True):
        X, y = self.dump()

        if shuffle:
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]

        X = X.reshape(-1, mini_batch, X.shape[1])
        y = y.reshape(-1, mini_batch, y.shape[1])

        return X, y


class MLP:
    def __init__(self, layer_size, num_feats, output_size, learning_rate):
        self.num_feats = num_feats
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layer_size = layer_size
        self._init_parameters()

    def _init_parameters(self):
        self.input_weights = np.random.normal(0,1e-3,(self.num_feats + 1, self.layer_size))
        self.output_weights = np.random.normal(0,1e-3,(self.layer_size + 1, self.output_size))
        self.i_grads = np.ones((self.num_feats + 1, self.layer_size))
        self.o_grads = np.ones((self.layer_size + 1, self.output_size))

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:,np.newaxis]

    def _sigmoid(self, x):
        try:
            result = 1 / (1 + np.exp(-x))
        except RuntimeWarning:
            result = np.zeros(x.shape)
        return result

    def _add_bias(self, x):
        x_biased = np.ones([x.shape[0], x.shape[1]+1])
        x_biased[:, :-1] = x
        return x_biased

    def _forward(self, x):
        x_biased = self._add_bias(x)
        z = self._sigmoid(np.matmul(x_biased, self.input_weights))
        z_biased = self._add_bias(z)
        return np.matmul(z_biased, self.output_weights), z_biased, x_biased, z

    def train(self, X, y):
        # Forward pass and error calculation
        pre_soft, z_biased, x_biased, z = self._forward(X)
        y_hat = self._softmax(pre_soft)
        J = -np.einsum('ij, ij->i', y, np.log(y_hat))

        # Update equations
        output_delta = np.matmul((y - y_hat).T, z_biased).T

        inner = np.matmul(y - y_hat, self.output_weights[:-1, :].T)
        input_delta = np.matmul(x_biased.T, inner * z * (1-z))

        self.output_weights += (self.learning_rate / np.sqrt(self.o_grads)) * output_delta
        self.input_weights += (self.learning_rate / np.sqrt(self.i_grads)) * input_delta

        self.o_grads += np.square(output_delta)
        self.i_grads += np.square(input_delta)

        return np.sum(J)

    def predict(self, X):
        return np.argmax(self._forward(X)[0], axis=1)

data_str = '2,2$0.8,-0.6$50$1;-4,-4$0.4,0$50$1;-2,2$0.8,0.6$50$2;4,-4$0.4,0$50$2;-2,-2$0.8,-0.6$50$3;4,4$0.4,0$50$3;2,-2$0.8,0.6$50$4;-4,4$0.4,0$50$4;'
data = Data(data_str)

mlp = MLP(20,2,4,0.5)
errors = []
epochs = 1000

for e in range(epochs):
    X_train, y_train = data.get_training_data()
    epoch_error = []
    for i in range(len(X_train)):
        batch_error = mlp.train(X_train[i], y_train[i])
        epoch_error.append(batch_error)
    errors.append(np.mean(batch_error))

confusion_matrix = np.zeros((len(data.dataset), len(data.dataset)))
correct = 0
classified = []
misclassified = []


for i in data.dataset.keys():
    predictions = mlp.predict(data.dataset[i])
    for j, p in enumerate(predictions):
        confusion_matrix[i][p] += 1
        if p == i:
            correct += 1
            classified.append((data.dataset[i][j], i))
        else:
            misclassified.append((data.dataset[i][j], i))

print(confusion_matrix)
print('Accuracy:', (correct/400)*100)

# Decision boundaries
X, _ = data.dump()

x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

pyplt.subplot(212, title="Decision Boundaries")

pyplt.contourf(xx, yy, Z, 3, alpha=0.2, linestyles='solid', cmap=pyplt.cm.coolwarm)

for x, c in classified:
    pyplt.scatter(x[0], x[1], c=data.colors[c])

for x, c in misclassified:
    pyplt.scatter(x[0], x[1], c=data.colors[c], edgecolor='black')

pyplt.subplot(211, title='Error vs Iteration')
pyplt.plot(errors)
pyplt.show()
