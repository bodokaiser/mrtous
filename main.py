import argparse
import itertools
import numpy as np
import os
import skimage.io
import torch
import torch.nn
import torch.multiprocessing as mp

from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from mrtous.network import UNet, Basic
from mrtous.summary import SummaryWriter
from mrtous.dataset import Minc2, MnibiteNative
from mrtous.transform import CenterCrop, Normalize, ExpandDim, ToTensor

def threshold(images):
    value = images.mean() - 2*images.var()

    return Variable(images.gt(value).float())

def save_images(dirname, inputs, outputs, targets, epoch, step=0):
    fmt = lambda n: os.path.join(dirname, f'{step:04d}_{n}_{epoch:03d}.png')

    if inputs.is_cuda:
        inputs = inputs.cpu()
        outputs = outputs.cpu()
        targets = targets.cpu()

    skimage.io.imsave(fmt('input'), inputs[0][0].data.numpy())
    skimage.io.imsave(fmt('output'), outputs[0][0].data.numpy())
    skimage.io.imsave(fmt('target'), targets[0][0].data.numpy())

def train_epoch(args, epoch, model, writer, loader, optimizer):
    model.train()

    total_loss = 0

    for step, (mr, us) in enumerate(loader):
        if us.sum() < 200:
            continue

        if args.cuda:
            us = us.cuda()
            mr = mr.cuda()

        #mask = threshold(us)
        inputs = Variable(mr)
        targets = Variable(us)
        outputs = model(inputs)

        optimizer.zero_grad()
        #loss = outputs.mul(mask).dist(targets.mul(mask), 2)
        loss = outputs.dist(targets, 2)
        #loss.div_(mask.sum().data[0])
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

        if args.nsteps > 0 and step % args.nsteps == 0:
            if args.save_loss:
                writer.write(epoch=epoch, step=step, loss=loss.data[0])
            if args.save_images:
                save_images(args.outdir, inputs, outputs, targets, epoch, step)

    if args.nepochs > 0 and epoch % args.nepochs == 0:
        if args.save_loss:
            writer.write(epoch=epoch, loss=loss.data[0])
        if args.save_images:
            save_images(args.outdir, inputs, outputs, targets, epoch)

    print(f'training (epoch: {epoch}, loss: {total_loss})')

def main(args):
    #model = UNet()
    model = Basic()

    if args.cuda:
        model.cuda()

    if args.save_loss or args.save_images:
        os.makedirs(args.outdir, exist_ok=True)

    loader = itertools.chain(*[
        DataLoader(MnibiteNative(
            Minc2(os.path.join(args.datadir, f'{int(i):02d}_mr.mnc'), Compose([
                CenterCrop(320),
                Normalize(MnibiteNative.MR_VRANGE),
                ExpandDim(0),
                ToTensor(),
            ])),
            Minc2(os.path.join(args.datadir, f'{int(i):02d}_us.mnc'), Compose([
                CenterCrop(320),
                Normalize(MnibiteNative.US_VRANGE),
                ExpandDim(0),
                ToTensor(),
            ])),
        ), shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
    for i in range(1, 4)])

    writer = SummaryWriter(args.outdir, 'loss.json')

    optimizer = Adam(model.parameters())

    for epoch in range(1, args.epochs+1):
        train_epoch(args, epoch, model, writer, loader, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--test', nargs='+', default=['11'])
    parser.add_argument('--train', nargs='+', default=['13'])
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--outdir', nargs='?', default='output')
    parser.add_argument('--datadir', nargs='?', default='mnibite')
    parser.add_argument('--batch_size', type=int, nargs='?', default=1)
    parser.add_argument('--num_workers', type=int, nargs='?', default=1)
    parser.add_argument('--every-steps', type=int, dest='nsteps', default=10)
    parser.add_argument('--every-epochs', type=int, dest='nepochs', default=1)
    parser.add_argument('--save-loss', dest='save_loss', action='store_true')
    parser.add_argument('--save-images', dest='save_images', action='store_true')
    parser.set_defaults(cuda=False, show_loss=False, show_images=False)
    main(parser.parse_args())
