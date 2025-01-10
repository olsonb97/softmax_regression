import numpy as np
from plot import SoftmaxPlot

class SoftmaxModel(SoftmaxPlot):
    """Softmax Regression Model"""
    def __init__(self, weights=None, bias=None):
        super().__init__()
        self.weights = weights
        self.bias = bias

    def _calculate_logits(self, x, w, b):
        return x @ w + b
    
    def _calculate_probabilities(self, z):
        stable_logits = z - np.max(z, axis=1, keepdims=True)
        numerator = np.exp(stable_logits)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def _calculate_gradients(self, x, y, p):
        d_w = (-1/x.shape[0]) * x.T @ (y - p)
        d_b = (-1/x.shape[0]) * np.sum(y - p, axis=0)
        return d_w, d_b

    def _adjust_parameters(self, d_w, d_b, learning_rate):
        self.weights = self.weights - (learning_rate * d_w)
        self.bias = self.bias - (learning_rate * d_b)
    
    def _calculate_loss(self, p, y):
        epsilon = 1e-10 # Prevent log(0)
        coefficient = (-1/y.shape[0])
        factor = np.sum(y * np.log(p + epsilon))
        return coefficient * factor
    
    def _initialize_sets(self, dataset, labels):
        num_features = dataset.shape[1]
        num_classes = len(np.unique(labels))
        xavier = np.sqrt(2 / (num_features + num_classes))
        if self.weights is None:
            self.weights = np.random.uniform(-xavier, xavier, size=(num_features, num_classes)).astype(np.float32)
        if self.bias is None:
            self.bias = np.random.uniform(size=(num_classes,)).astype(np.float32)
        
    def _shuffle_data(self, dataset, labels):
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        return dataset[indices], labels[indices]

    def _process_batch(self, x_batch, y_batch, w, b, eta):
        z = self._calculate_logits(x_batch, w, b)
        p = self._calculate_probabilities(z)
        dw, db = self._calculate_gradients(x_batch, y_batch, p)
        self._adjust_parameters(dw, db, eta)
        return self._calculate_loss(p, y_batch)

    def train(
            self,
            dataset,
            labels,
            epochs,
            batches=1,
            learning_rate=0.01,
            decay_rate=0.001,
            shuffle=True,
            plot=True,
            print_interval=None
    ):
        self._initialize_sets(dataset, labels)
        x = dataset
        y = np.eye(len(np.unique(labels)))[labels]
        losses = np.zeros(shape=(epochs,))
        learning_rates = np.zeros(shape=(epochs,))

        try:
            for epoch in range(epochs):
                x, y = self._shuffle_data(x, y) if shuffle else (x, y)
                eta = learning_rate / (1 + decay_rate * epoch)
                batch_size = x.shape[0] // batches
                epoch_loss = 0
                w = self.weights
                b = self.bias

                for i in range(batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    x_batch = x[start:end]
                    y_batch = y[start:end]
                    epoch_loss += self._process_batch(x_batch, y_batch, w, b, eta)

                losses[epoch] = epoch_loss / batches
                learning_rates[epoch] = eta

                if (epoch + 1) % print_interval == 0 and print_interval:
                    print(f"Epoch: {epoch+1}/{epochs}, Loss: {losses[epoch]:.6f}, Learning Rate: {eta:.6f}")
        except KeyboardInterrupt:
            pass

        if plot:
            self.plot_training(
                loss=losses,
                epochs=epochs,
                learning_rates=learning_rates,
                weights=self.weights,
                bias=self.bias
            )
            self.show()

    def predict(self, data):
        x = data
        w = self.weights
        b = self.bias
        z = self._calculate_logits(x, w, b)
        p = self._calculate_probabilities(z)
        return np.argmax(p, axis=1)
    
    def get_accuracy(self, predicted_labels, true_labels):
        return np.mean(predicted_labels == true_labels, axis=0) * 100

    def test(self, data, labels):
        true_labels = labels
        predicted_labels = self.predict(data)
        return self.get_accuracy(predicted_labels, true_labels)
    
    def save(self, filename="./params.npz"):
        np.savez_compressed(
            file=filename,
            weights=self.weights,
            bias=self.bias
        )

    def load(self, filename="./params.npz"):
        params = np.load(filename)
        self.weights = params["weights"]
        self.bias = params["bias"]
