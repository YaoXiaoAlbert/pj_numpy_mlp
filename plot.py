import matplotlib.pyplot as plt
import seaborn as sns

# 参数热力图
def plot_heatmap(weights, layer_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Heatmap for {layer_name}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    plt.show()

#  权重直方图
def plot_histogram(weights, layer_name):
    plt.figure(figsize=(6, 4))
    plt.hist(weights.flatten(), bins=50, alpha=0.75)
    plt.title(f"Histogram of weights in {layer_name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 偏置项可视化
def plot_biases(biases, layer_name):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(biases)), biases)
    plt.title(f"Biases in {layer_name}")
    plt.xlabel("Neuron")
    plt.ylabel("Bias value")
    plt.show()