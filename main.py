import os
import argparse
import numpy as np

from mrtous import dataset, transform, network
from torch import nn, optim, autograd
from torch.utils import data
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1

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

    grid = axes_grid1.ImageGrid(fig, 111, (rows, cols), axes_pad=.1,
        cbar_mode='single', cbar_location='right', label_mode=1)

    imgs = []
    axes = []

    def update(images):
        if len(imgs) == 0:
            for i in range(len(images)):
                axis = grid[i]
                axis.set_axis_off()
                axis.set_aspect('auto')
                axes.append(axis)
                imgs.append(axis.imshow(images[i],
                    interpolation='none', vmin=VMIN, vmax=VMAX))

            grid[0].cax.colorbar(imgs[0])
            for i, subtitle in enumerate(subtitles):
                grid[i].set_title(subtitle)
        else:
            for i in range(len(images)):
                imgs[i].set_data(images[i])
                axes[i].figure.canvas.draw()
                axes[i].figure.canvas.flush_events()

    plt.ion()
    plt.show()

    return update

def main(args):
    model = network.Basic()

    test_loader = data.DataLoader(dataset.MNIBITEFolder(
        map(lambda d: os.path.join(args.datadir, d), args.test)),
            shuffle=True, batch_size=128, num_workers=4)
    train_loader = data.DataLoader(dataset.MNIBITEFolder(
        map(lambda d: os.path.join(args.datadir, d), args.train)),
            shuffle=True, batch_size=128, num_workers=4)

    if args.show_images:
        image_loader = data.DataLoader(dataset.MNIBITENative(args.datadir,
            int(args.train[0]), transform.RegionCrop()), shuffle=True)

    test_losses = []
    train_losses = []

    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters())

    if args.show_loss:
        update_loss = loss_plot()
    if args.show_images:
        update_images = image_plot('training images',
            ['MRI', 'US', 'RE'])
    if args.show_patches:
        update_patches = image_plot('training and testing patches',
            ['MRI', 'US', 'RE'], rows=2)

    for epoch in range(1, args.epochs+1):
        test_loss = 0
        train_loss = 0

        for step, (mr, us) in enumerate(train_loader):
            inputs = autograd.Variable(mr).unsqueeze(1)
            targets = autograd.Variable(us).unsqueeze(1)
            results = model(inputs)

            optimizer.zero_grad()
            loss = criterion(results, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]

        train_patches = [
            inputs.data[0][0].numpy(),
            targets.data[0][0].numpy(),
            results.data[0][0].numpy(),
        ]
        train_losses.append(train_loss)

        for step, (mr, us) in enumerate(test_loader):
            inputs = autograd.Variable(mr).unsqueeze(1)
            targets = autograd.Variable(us).unsqueeze(1)
            results = model(inputs)

            loss = criterion(results, targets)
            test_loss += loss.data[0]

        test_patches = [
            inputs.data[0][0].numpy(),
            targets.data[0][0].numpy(),
            results.data[0][0].numpy(),
        ]
        test_losses.append(test_loss)

        if args.show_loss:
            update_loss(train_losses, test_losses)
        if args.show_images:
            for _, (mr, us) in enumerate(image_loader):
                if np.any(us.numpy()) and sum(us.numpy().shape[1:3]) > 30:
                    inputs = autograd.Variable(mr).unsqueeze(1)
                    targets = autograd.Variable(us).unsqueeze(1)
                    results = model(inputs)
                    break

            update_images([
                inputs.data[0][0].numpy(),
                targets.data[0][0].numpy(),
                results.data[0][0].numpy(),
            ])
        if args.show_patches:
            update_patches([
                *train_patches,
                *test_patches,
            ])

        print(f'testing (epoch: {epoch}, loss: {test_loss}')
        print(f'training (epoch: {epoch}, loss: {train_loss})')

    input('press enter to exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, nargs='+', default=['11'])
    parser.add_argument('--train', type=str, nargs='+', default=['13'])
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--show-loss', dest='show_loss', action='store_true')
    parser.add_argument('--show-images', dest='show_images', action='store_true')
    parser.add_argument('--show-patches', dest='show_patches', action='store_true')
    parser.set_defaults(show_loss=False, show_images=False, show_patches=False)
    main(parser.parse_args())