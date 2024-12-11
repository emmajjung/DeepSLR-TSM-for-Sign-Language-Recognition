import matplotlib.pyplot as plt
import numpy as np

# Function to plot data
def plot_learning_rate_vs_accuracy(data, title):
    learning_rates = list(data.keys())
    accuracy_values = [np.mean(accuracies) for accuracies in data.values()]

    # Sort the data for better visualization
    sorted_indices = np.argsort(learning_rates)
    learning_rates = np.array(learning_rates)[sorted_indices]
    accuracy_values = np.array(accuracy_values)[sorted_indices]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates, accuracy_values, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.xscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel('Learning Rate (log scale)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Input data: dict of learning rates vs the accuracy

learning_rate_data_10 = {
    0.00001: [59.72],
    0.00003: [70.83],
    0.000045: [75.00],
    0.000048: [75.00],
    0.000049: [81.94],
    0.00005: [84.72],
    0.000051: [75.00],
    0.000055: [77.78],
    0.00008: [80.56],
    0.0001: [80.50],
    0.0005: [100],
    0.001: [44.44]
}

plot_learning_rate_vs_accuracy(learning_rate_data_10, 'Learning Rate vs Accuracy (10 Epochs)')

learning_rate_data_15 = {
    0.000049: [76.36],
    0.00005: [81.94],
    0.000051: [76.39],
    0.000055: [84.72]
}

plot_learning_rate_vs_accuracy(learning_rate_data_15, 'Learning Rate vs Accuracy (15 Epochs)')


learning_rate_data_20 = {
    0.000049: [77.78],
    0.00005: [81.94],
    0.000051: [79.17],
    0.000055: [79.10]
}
plot_learning_rate_vs_accuracy(learning_rate_data_20, 'Learning Rate vs Accuracy (20 Epochs)')


learning_rate_data_25 = {
    0.000049: [100],
    0.00005: [77.78],
    0.000051: [100],
    0.000055: [79.17]
}
plot_learning_rate_vs_accuracy(learning_rate_data_25, 'Learning Rate vs Accuracy (25 Epochs)')