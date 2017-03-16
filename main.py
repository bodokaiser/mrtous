import argparse
import numpy as np
import os
import torch
import torch.nn
import torch.multiprocessing as mp

from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from mrtous import io
from mrtous.network import UNet
from mrtous.summary import SummaryWriter
from mrtous.dataset import Concat, Minc2, MnibiteNative
from mrtous.transform import Normalize, CenterCrop, ExpandDim

MR_VRANGE = [-32768, 32767]
US_VRANGE = [0, 255]

def threshold(images):
    value = images.mean() - 2*images.var()

    return Variable(images.gt(value).float())

def train_epoch(args, epoch, model, writer, loader, optimizer):
    model.train()

    total_loss = 0

    for step, (mr, us) in enumerate(loader):
        if us.sum() < 10:
            continue

        mask = threshold(us)
        inputs = Variable(mr)
        targets = Variable(us)
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = outputs.mul(mask).dist(targets.mul(mask), 2)
        loss.div_(mask.sum().data[0])
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

        if args.save_loss:
            writer.write(epoch=epoch, step=step, loss=loss.data[0])

    if args.save_images:
        io.imsave(os.path.join(args.outdir, f'{epoch:03d}_input.png'),
            inputs[0][0])
        io.imsave(os.path.join(args.outdir, f'{epoch:03d}_output.png'),
            outputs[0][0])
        io.imsave(os.path.join(args.outdir, f'{epoch:03d}_target.png'),
            targets[0][0])

    print(f'training (epoch: {epoch}, loss: {total_loss}, pid: {pid})')

def main(args):
    model = UNet()

    if args.save_loss or args.save_images:
        os.makedirs(args.outdir, exist_ok=True)

    loader = DataLoader(Concat([
        MnibiteNative(
            Minc2(os.path.join(args.datadir, f'{int(i):02d}_mr.mnc'), Compose([
                Normalize(MR_VRANGE),
                CenterCrop(320),
                ExpandDim(2),
                ToTensor(),
            ])),
            Minc2(os.path.join(args.datadir, f'{int(i):02d}_us.mnc'), Compose([
                Normalize(US_VRANGE),
                CenterCrop(320),
                ExpandDim(2),
                ToTensor(),
            ]))) for i in args.train]),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    writer = SummaryWriter(args.outdir, 'loss.json')
    
    optimizer = Adam(model.parameters())

    for epoch in range(1, args.epochs+1):
        train_epoch(args, epoch, model, writer, loader, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='+', default=['11'])
    parser.add_argument('--train', nargs='+', default=['13'])
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--outdir', type=str, nargs='?', default='output')
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--batch_size', type=int, nargs='?', default=1)
    parser.add_argument('--num_workers', type=int, nargs='?', default=2)
    parser.add_argument('--save-loss', dest='save_loss', action='store_true')
    parser.add_argument('--save-images', dest='save_images', action='store_true')
    parser.set_defaults(show_loss=False, show_images=False)
    main(parser.parse_args())