import time
import torch

from argparse import ArgumentParser

from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Lambda

from mrtous.session import Trainer
from mrtous.network import One, Two, UNet
from mrtous.dataset import MNIBITE
from mrtous.transform import ToTensor, Clip, HistNormalize

input_transform = Compose([
    Clip(-32000, 0),
    HistNormalize(),
    ToTensor(),
])
target_transform = Compose([
    HistNormalize(),
    ToTensor(),
])

def test_epoch(args, epoch, model, loader, crit1, crit2, trainer):
    model.eval()

    for step, (mr, us) in enumerate(loader):
        if us.sum() < 1e3:
            continue

        inputs = Variable(mr, volatile=True)
        targets = Variable(us, volatile=True)

        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        outputs = outputs.mul(targets.gt(0).float())

        loss1 = crit1(outputs, targets)
        loss2 = crit2(outputs, targets)
        loss = loss1 + loss2

        trainer.l1meter.add(loss1.data[0])
        trainer.l2meter.add(loss2.data[0])

        if args.log_steps > 0 and step % args.log_steps == 0:
            trainer.log_losses(epoch, step)
        if args.vis_steps > 0 and step % args.vis_steps == 0:
            trainer.vis_images(epoch, step, dict(
                input=inputs[0][0],
                output=outputs[0][0],
                target=targets[0][0]))

def train_epoch(args, epoch, model, loader, crit1, crit2, optimizer, trainer):
    model.train()

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

        loss1 = crit1(outputs, targets)
        loss2 = crit2(outputs, targets)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainer.l1meter.add(loss1.data[0])
        trainer.l2meter.add(loss2.data[0])

        if args.log_steps > 0 and step % args.log_steps == 0:
            trainer.log_losses(epoch, step)
        if args.vis_steps > 0 and step % args.vis_steps == 0:
            trainer.vis_images(epoch, step, dict(
                input=inputs[0][0],
                output=outputs[0][0],
                target=targets[0][0]))

def main(args):
    if args.model == 'one':
        Net = One
    if args.model == 'two':
        Net = Two
    if args.model == 'unet':
        Net = UNet

    model = Net()
    crit1 = L1Loss()
    crit2 = MSELoss()

    if args.cuda:
        model = model.cuda()
        crit1 = crit1.cuda()
        crit2 = crit2.cuda()
    if args.state:
        model.load_state_dict(torch.load(args.state))

    trainer = Trainer(args, model)
    optimizer = Adam(model.parameters())

    test_loader = DataLoader(MNIBITE(args.testdir, input_transform, target_transform))
    train_loader = DataLoader(MNIBITE(args.traindir, input_transform, target_transform))

    for epoch in range(args.num_epochs):
        train_epoch(args, epoch, model, train_loader, crit1, crit2, optimizer, trainer)
        trainer.vis_losses(epoch)
        trainer.reset()

        test_epoch(args, epoch, model, test_loader, crit1, crit2, trainer)
        trainer.vis_losses(epoch)
        trainer.reset()

        if args.save_epochs > 0 and epoch % args.save_epochs == 0:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f'{args.name}-{stamp}.pth')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', choices=['one', 'two', 'unet'], required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='action')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--name', default='mrtous')
    parser_train.add_argument('--testdir')
    parser_train.add_argument('--traindir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=60)
    parser_train.add_argument('--vis-port', type=int, default=3000)
    parser_train.add_argument('--vis-steps', type=int, default=0)
    parser_train.add_argument('--log-steps', type=int, default=0)
    parser_train.add_argument('--save-epochs', type=int, default=0)

    main(parser.parse_args())