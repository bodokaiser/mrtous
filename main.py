import numpy as np
import time
import torch

from argparse import ArgumentParser

from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Lambda

from visdom import Visdom

from mrtous.network import One, Two, UNet
from mrtous.dataset import MNIBITE
from mrtous.transform import ToTensor, Clip, HistNormalize

input_transform = Compose([
    Clip(-25000, -20000),
    HistNormalize(),
    ToTensor(),
])
target_transform = Compose([
    HistNormalize(),
    ToTensor(),
])

def main(args):
    if args.model == 'one':
        Net = One
    if args.model == 'two':
        Net = Two
    if args.model == 'unet':
        Net = UNet

    model = Net()
    model.train()

    if args.cuda:
        model = model.cuda()
    if args.state:
        model.load_state_dict(torch.load(args.state))

    loader = DataLoader(MNIBITE(args.datadir, input_transform, target_transform))
    optimizer = SGD(model.parameters(), 1e-4)

    if args.vis_steps > 0:
        vis = Visdom(port=args.vis_port)

    for epoch in range(1, args.num_epochs+1):
        epoch_loss = []

        for step, (mr, us) in enumerate(loader):
            if us.sum() < 1e3:
                continue

            inputs = Variable(mr)
            targets = Variable(us)

            if args.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            outputs = outputs.mul(targets.gt(0).float())

            optimizer.zero_grad()
            loss = outputs.dist(targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            if args.log_steps > 0 and step & args.log_steps == 0:
                print(f'epoch: {epoch}, step: {step}, loss: {sum(epoch_loss)/len(epoch_loss)}')
            if args.vis_steps > 0 and step % args.vis_steps == 0:
                title = f'(epoch: {epoch}, step: {step})'
                vis.image(inputs[0][0].cpu().data, opts=dict(title=f'input {title}'))
                vis.image(outputs[0][0].cpu().data, opts=dict(title=f'output {title}'))
                vis.image(targets[0][0].cpu().data, opts=dict(title=f'target {title}'))
            if args.save_steps > 0 and step % args.save_steps == 0:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                torch.save(model.state_dict(), f'{args.model}-{stamp}.pth')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', choices=['one', 'two', 'unet'], required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='action')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--vis-port', type=int, default=3000)
    parser_train.add_argument('--vis-steps', type=int, default=0)
    parser_train.add_argument('--log-steps', type=int, default=0)
    parser_train.add_argument('--save-steps', type=int, default=0)

    main(parser.parse_args())