import matplotlib.pyplot as plt

def image_grid(images, cols, rows):
    plt.figure(figsize=[18, 6])
    for i in range(cols*rows):
        plt.subplot(cols, rows, i+1)
        plt.imshow(images[i])
    plt.show()

def loss_plot():
    fig, ax = plt.subplots()

    plt.ion()
    plt.show()

    def update(x, y):
        ax.plot(x, y, 'r')
        ax.figure.canvas.flush_events()

    return update