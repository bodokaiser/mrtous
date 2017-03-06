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

def image_plot(titles, subtitles):
    fig = plt.figure(1, (10, 10))
    grid = axes_grid1.ImageGrid(fig, 111, (3, 3), axes_pad=.1,
        cbar_mode='single', cbar_location='right', label_mode=1)

    imgs = []
    axes = []

    def update(images):
        if len(imgs) == 0:
            for i in range(len(titles)):
                for j in range(len(subtitles)):
                    axis = grid[3*i+j]
                    axis.set_axis_off()
                    axes.append(axis)
                    imgs.append(axis.imshow(images[i][j],
                        interpolation='none', vmin=VMIN, vmax=VMAX))
            grid[0].cax.colorbar(imgs[0])
            grid[0].set_title(subtitles[0])
            grid[1].set_title(subtitles[1])
            grid[2].set_title(subtitles[2])
            grid[3].set_title(titles[1], rotation='vertical', x=-.1, y=+.7)
            grid[6].set_title(titles[2], rotation='vertical', x=-.1, y=+.7)
        else:
            for i in range(len(titles)):
                for j in range(len(subtitles)):
                    imgs[3*i+j].set_data(images[i][j])
                    axes[3*i+j].figure.canvas.draw()
                    axes[3*i+j].figure.canvas.flush_events()

    plt.ion()
    plt.show()

    return update

def var(mr, us):
    inputs = autograd.Variable(mr).unsqueeze(1)
    targets = autograd.Variable(us).unsqueeze(1)
    return inputs, targets

def sample(loader):
    for _, (mr, us) in enumerate(loader):
        if np.any(us.numpy()) and sum(us.numpy().shape[1:3]) > 30:
            return var(mr, us)

def main(args):
    model = network.Basic()

    test_loader = data.DataLoader(dataset.MNIBITEFolder(
        map(lambda d: os.path.join(args.datadir, d), args.test)),
            shuffle=True, batch_size=128, num_workers=4)
    train_loader = data.DataLoader(dataset.MNIBITEFolder(
        map(lambda d: os.path.join(args.datadir, d), args.train)),
            shuffle=True, batch_size=128, num_workers=4)
    image_loader = data.DataLoader(dataset.MNIBITENative(args.datadir,
        int(args.train[0]), transform.RegionCrop()), shuffle=True)

    test_losses = []
    train_losses = []

    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters())

    if args.show_loss:
        update_loss = loss_plot()
    if args.show_image:
        update_image = image_plot(
            ['training image', 'training patch', 'testing patch'],
            ['MRI', 'US', 'RE'])

    for epoch in range(1, args.epochs+1):
        test_loss = 0
        train_loss = 0

        for step, (mr, us) in enumerate(train_loader):
            inputs, targets = var(mr, us)
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
            inputs, targets = var(mr, us)
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

        if args.show_image:
            inputs, targets = sample(image_loader)
            results = model(inputs)

            update_image([[
                inputs.data[0][0].numpy(),
                targets.data[0][0].numpy(),
                results.data[0][0].numpy(),
            ], train_patches, test_patches])

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
    parser.add_argument('--show-image', dest='show_image', action='store_true')
    main(parser.parse_args())