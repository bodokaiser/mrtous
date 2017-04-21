import numpy as np

from argparse import ArgumentParser

from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Lambda

from visdom import Visdom

from mrtous.network import Simple
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
    model = Simple()
    model.train()

    loader = DataLoader(MNIBITE(args.datadir, input_transform, target_transform),
        num_workers=args.num_workers)
    optimizer = SGD(model.parameters(), 1e-4)

    if args.vis_steps > 0:
        vis = Visdom(port=args.vis_port)

    for epoch in range(1, args.num_epochs+1):
        epoch_loss = []

        for step, (mr, us) in enumerate(loader):
            if us.sum() < 1e4:
                continue

            inputs = Variable(mr)
            targets = Variable(us)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = outputs.dist(targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            if step % args.vis_steps == 0:
                title = f'(epoch: {epoch}, step: {step})'
                vis.image(mr[0][0], opts=dict(title=f'input {title}'))
                vis.image(us[0][0], opts=dict(title=f'target {title}'))
                vis.image(outputs[0][0].data, opts=dict(title=f'output {title}'))

        print(f'epoch: {epoch}, loss: {sum(epoch_loss)/len(epoch_loss)}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')

    subparsers = parser.add_subparsers(dest='action')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--vis-port', type=int, default=3000)
    parser_train.add_argument('--vis-steps', type=int, default=0)

    main(parser.parse_args())