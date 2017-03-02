import numpy as np

from matplotlib import lines
from matplotlib import pyplot as plt

VMIN = -0.5
VMAX = +0.5


def image_grid(images, cols, rows):
    plt.figure(figsize=[18, 6])
    for i in range(cols*rows):
        plt.subplot(cols, rows, i+1)
        plt.imshow(images[i], interpolation='none', vmin=VMIN, vmax=VMAX)
    plt.ioff()
    plt.show()

def image_plot(titles):
    fig, axes = plt.subplots(1, len(titles))

    img = []

    def update(images):
        if len(img) == 0:
            for i in range(len(titles)):
                axes[i].set_title(titles[i])
                img.append(axes[i].imshow(images[i],
                    interpolation='none', vmin=VMIN, vmax=VMAX))
            fig.colorbar(img[0])
        else:
            for i in range(len(titles)):
                img[i].set_data(images[i])
                axes[i].figure.canvas.draw()
                axes[i].figure.canvas.flush_events()

    plt.ion()
    plt.show()

    return update

def loss_plot():
    fig, axes = plt.subplots()

    lin1, = axes.plot([], [], color='orange', label='training')
    lin2, = axes.plot([], [], color='blue', label='testing')

    axes.set_title('mean-squared-error per epoch')
    axes.set_xlabel('epoch')
    axes.set_ylabel('mse')
    axes.legend()

    plt.ion()
    plt.show()

    def update(train_loss, test_loss):
        x = np.arange(0, len(train_loss))

        lin1.set_data(x, train_loss)
        if len(test_loss) > 0:
            lin2.set_data(x, test_loss)

        axes.set_xticks(np.arange(0, np.max(x)+1, 1))
        axes.relim()
        axes.autoscale_view(scalex=False, scaley=True)
        axes.figure.canvas.draw()
        axes.figure.canvas.flush_events()

    return update