import numpy as np
import matplotlib.pyplot as plt

def image_grid(images, cols, rows):
    plt.figure(figsize=[18, 6])
    for i in range(cols*rows):
        plt.subplot(cols, rows, i+1)
        plt.imshow(images[i])
    plt.show()

def plot_loss(loss):
    plt.plot(np.arange(0, len(loss)), loss)
    plt.show()