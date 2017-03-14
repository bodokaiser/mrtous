import argparse
import numpy as np
import os
import torch
import torch.nn

from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mrtous.network import Basic
from mrtous.dataset import MnibiteNative

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

VMIN = 0.0
VMAX = 1.0

def loss_plot():
    fig, axes = plt.subplots()

    lin1, = axes.plot([], [], color='orange', label='training')
    lin2, = axes.plot([], [], color='blue', label='testing')

    axes.set_title('MSE per Epoch')
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

def image_plot(title, subtitles, rows=1, cols=3):
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(title)

    grid = ImageGrid(fig, 111, (rows, cols), axes_pad=.1,
        cbar_mode='single', cbar_location='right', label_mode=1)

    imgs = []
    axes = []

    def update(images):
        if len(imgs) == 0:
            for i in range(len(images)):
                axis = grid[i]
                axis.set_axis_off()
                axes.append(axis)
                imgs.append(axis.imshow(images[i],
                    interpolation='none', vmin=VMIN, vmax=VMAX))

            grid[0].cax.colorbar(imgs[0])
            for i, subtitle in enumerate(subtitles):
                grid[i].set_title(subtitle)
        else:
            for i in range(len(images)):
                imgs[i].set_data(images[i])
                axes[i].set_aspect('auto')
                axes[i].figure.canvas.draw()
                axes[i].figure.canvas.flush_events()

    plt.ion()
    plt.show()

    return update

def threshold(image):
    value = np.mean(image) - 2*np.var(image)

    mask = image > value
    mask = torch.from_numpy(mask.astype(int))

    return Variable(mask).float()

def main(args):
    model = Basic()

    dataset = MnibiteNative(args.datadir, int(args.train))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    mr, us = dataset[120]
    fixed_inputs = Variable(torch.from_numpy(mr)).unsqueeze(0).unsqueeze(0)
    fixed_targets = Variable(torch.from_numpy(us)).unsqueeze(0).unsqueeze(0)

    test_losses = []
    train_losses = []

    optimizer = Adam(model.parameters())

    if args.show_loss:
        update_loss = loss_plot()
    if args.show_images:
        update_images = image_plot('Training Images', ['MRI', 'US', 'OUT'])

    for epoch in range(1, args.epochs+1):
        test_loss = 0
        train_loss = 0

        for mr, us in dataloader:
            if np.any(us.numpy()) and us.sum() > 100:
                mask = threshold(us.numpy())

                inputs = Variable(mr).unsqueeze(1)
                targets = Variable(us).unsqueeze(1)
                results = model(inputs)

                optimizer.zero_grad()
                loss = results[0].mul(mask).dist(targets[0].mul(mask), 2)
                loss.div_(mask.sum().data[0])
                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        if args.show_loss:
            update_loss(train_losses, test_losses)
        if args.show_images:
            update_images([
                fixed_inputs.data[0][0].numpy(),
                fixed_targets.data[0][0].numpy(),
                model(fixed_inputs).data[0][0].numpy(),
            ])

        print(f'testing (epoch: {epoch}, loss: {test_loss}')
        print(f'training (epoch: {epoch}, loss: {train_loss})')

    input('press enter to exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, nargs='?', default=11)
    parser.add_argument('--train', type=int, nargs='?', default=13)
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--show-loss', dest='show_loss', action='store_true')
    parser.add_argument('--show-images', dest='show_images', action='store_true')
    parser.set_defaults(show_loss=False, show_images=False)
    main(parser.parse_args())