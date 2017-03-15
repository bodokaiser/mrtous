import argparse
import numpy as np
import os
import torch
import torch.nn

from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mrtous import io
from mrtous.network import Basic
from mrtous.dataset import Concat, MnibiteNative

from matplotlib import pyplot as plt

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

def threshold(images):
    value = images.mean() - 2*images.var()

    return Variable(images.gt(value).float())

def main(args):
    model = Basic()

    dataset = Concat([
        MnibiteNative(
            os.path.join(args.datadir, f'{int(i):02d}_mr.mnc'),
            os.path.join(args.datadir, f'{int(i):02d}_us.mnc'))
        for i in args.train
    ])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_losses = []
    train_losses = []

    optimizer = Adam(model.parameters())

    if args.show_loss:
        update_loss = loss_plot()

    for epoch in range(1, args.epochs+1):
        test_loss = 0
        train_loss = 0

        for mr, us in dataloader:
            mask = threshold(us)

            inputs = Variable(mr)
            targets = Variable(us)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = outputs.mul(mask).dist(targets.mul(mask), 2)
            loss.div_(mask.sum().data[0])
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        if args.show_loss:
            update_loss(train_losses, test_losses)
        if args.save_images:
            os.makedirs(args.outdir, exist_ok=True)

            io.imsave(os.path.join(args.outdir, f'{epoch:03d}_input.png'),
                inputs[0][0])
            io.imsave(os.path.join(args.outdir, f'{epoch:03d}_output.png'),
                outputs[0][0])
            io.imsave(os.path.join(args.outdir, f'{epoch:03d}_target.png'),
                targets[0][0])

        print(f'testing (epoch: {epoch}, loss: {test_loss}')
        print(f'training (epoch: {epoch}, loss: {train_loss})')

    input('press enter to exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='+', default=['11'])
    parser.add_argument('--train', nargs='+', default=['13'])
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--outdir', type=str, nargs='?', default='output')
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--batch_size', type=int, nargs='?', default=64)
    parser.add_argument('--show-loss', dest='show_loss', action='store_true')
    parser.add_argument('--save-images', dest='save_images', action='store_true')
    parser.set_defaults(show_loss=False, show_images=False)
    main(parser.parse_args())