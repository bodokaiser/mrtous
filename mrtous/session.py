import numpy as np

from torchnet.meter import AverageValueMeter

from visdom import Visdom

class Trainer:

    def __init__(self, args, model):
        self.name = args.name
        self.model = model
        self.l1win = None
        self.l2win = None
        self.l1meter = AverageValueMeter()
        self.l2meter = AverageValueMeter()
        self.visdom = Visdom(port=args.vis_port) if args.vis_steps > 0 else None

    @property
    def mode(self):
        return 'training' if self.model.training else 'testing'

    @property
    def losses(self):
        return self.l1meter.value()[0], self.l2meter.value()[1]

    def reset(self):
        self.l1meter.reset()
        self.l2meter.reset()

    def log_losses(self, epoch, step):
        l1, l2 = self.losses
        message = f'{self.name} is {self.mode} (epoch: {epoch}, step: {step}) '
        message += f'l1 average: {l1}, l2 average: {l2}'
        print(message)

    def vis_losses(self, epoch):
        l1, l2 = self.losses
        x, y1, y2 = np.array([epoch]), np.array([l1]), np.array([l2])

        if self.l1win is None or self.l2win is None:
            opt = dict(xlabel='epochs', xtickstep=1, ylabel='mean loss',
                width=900)
            self.l1win = self.visdom.line(X=x, Y=y1, opts=dict(
                title=f'l1 loss ({self.name})', **opt))
            self.l2win = self.visdom.line(X=x, Y=y2, opts=dict(
                title=f'l2 loss ({self.name})', **opt))
        else:
            n = '1' if self.model.training else '2'
            self.visdom.updateTrace(X=x, Y=y1, win=self.l1win, name=n)
            self.visdom.updateTrace(X=x, Y=y2, win=self.l2win, name=n)

    def vis_images(self, epoch, step, images):
        title = f'({self.name}, epoch: {epoch}, step: {step})'
        for key, image in images.items():
            self.visdom.image(image.cpu().data, env=self.mode,
                opts=dict(title=f'{key} {title}'))