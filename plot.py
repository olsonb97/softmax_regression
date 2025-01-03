import matplotlib.pyplot as plt

class SoftmaxPlot:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.axes = [
            self.fig.add_subplot(2, 2, i+1) for i in range(4)
        ]

    def plot_loss(self, loss, epochs):
        self.axes[0].plot(epochs, loss)
        self.axes[0].set_title("Loss Minimization")
        self.axes[0].set_xlabel("Epochs")
        self.axes[0].set_ylabel("Loss")

    def plot_learning_rate(self, learning_rate, epochs):
        self.axes[1].plot(epochs, learning_rate)
        self.axes[1].set_title("Learning Rate Decay")
        self.axes[1].set_xlabel("Epochs")
        self.axes[1].set_ylabel("Learn Rate")

    def plot_weight_convergence(self, weights):
        for class_idx in range(weights.shape[1]):
            self.axes[2].plot(
                range(1, weights.shape[0]+1),
                weights[:, class_idx],
                marker='o', linestyle='-', label=f"Class {class_idx + 1} Weights"
            )
        self.axes[2].set_title("Weight Convergence")
        self.axes[2].set_xlabel("Feature")
        self.axes[2].set_ylabel("Weight Values")
        self.axes[2].legend()

    def plot_bias_convergence(self, bias):
        for class_idx, bias_value in enumerate(bias):
            self.axes[3].plot(
                class_idx+1, 
                bias_value, 
                marker='o', linestyle='-', label=f"Class {class_idx + 1} Bias"
            )
        
        self.axes[3].set_title("Bias Convergence")
        self.axes[3].set_xlabel("Class")
        self.axes[3].set_ylabel("Bias Values")
        self.axes[3].set_xticks(range(1, len(bias)+1))
        self.axes[3].legend()
    
    def show(self):
        plt.tight_layout()
        plt.show()