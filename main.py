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
from mrtous.summary import SummaryWriter
from mrtous.dataset import Concat, MnibiteNative

def threshold(images):
    value = images.mean() - 2*images.var()

    return Variable(images.gt(value).float())

def main(args):
    model = Basic()

    loader = DataLoader(Concat([
        MnibiteNative(
            os.path.join(args.datadir, f'{int(i):02d}_mr.mnc'),
            os.path.join(args.datadir, f'{int(i):02d}_us.mnc'))
        for i in args.train]), batch_size=args.batch_size, shuffle=True)

    writer = SummaryWriter(os.path.join(args.outdir, 'loss.json'))

    optimizer = Adam(model.parameters())

    for epoch in range(1, args.epochs+1):
        train_loss = 0

        for step, (mr, us) in enumerate(loader):
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

            if args.save_loss:
                writer.write(epoch=epoch, step=step, loss=loss.data[0])

        if args.save_images:
            io.imsave(os.path.join(args.outdir, f'{epoch:03d}_input.png'),
                inputs[0][0])
            io.imsave(os.path.join(args.outdir, f'{epoch:03d}_output.png'),
                outputs[0][0])
            io.imsave(os.path.join(args.outdir, f'{epoch:03d}_target.png'),
                targets[0][0])

        print(f'training (epoch: {epoch}, loss: {train_loss}')

    input('press enter to exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='+', default=['11'])
    parser.add_argument('--train', nargs='+', default=['13'])
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--outdir', type=str, nargs='?', default='output')
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--batch_size', type=int, nargs='?', default=64)
    parser.add_argument('--save-loss', dest='save_loss', action='store_true')
    parser.add_argument('--save-images', dest='save_images', action='store_true')
    parser.set_defaults(show_loss=False, show_images=False)
    main(parser.parse_args())