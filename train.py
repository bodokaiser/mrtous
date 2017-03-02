import argparse
import torch

from mrtous import dataset, transform, network
from mrtous import session, visualize
from torch.utils.data import DataLoader

def main(args):
    model = network.Simple()

    test_loader = DataLoader(dataset.MNIBITE(args.datadir, args.test,
        transform.RegionCrop()), shuffle=True)
    train_loader = DataLoader(dataset.MNIBITE(args.datadir, args.train,
        transform.RegionCrop()), shuffle=True)

    test_loss = []
    train_loss = []

    if args.plot_loss:
        loss_plot_fn = visualize.loss_plot()
    if args.plot_image:
        image_plot_fn = visualize.image_plot(['mr', 'us', 're', 'us-re'])

    def test_fn(inputs, targets, results, loss):
        test_loss.append(loss)

        if args.plot_loss:
            loss_plot_fn(test_loss, train_loss)
        print(f'testing (epoch: {len(test_loss)}, loss: {loss}')

    def train_fn(inputs, targets, results, epoch, loss):
        train_loss.append(loss)

        if args.plot_image:
            image_plot_fn([
                inputs.data[0][0].numpy(),
                targets.data[0][0].numpy(),
                results.data[0][0].numpy(),
                targets.data[0][0].numpy()-results.data[0][0].numpy(),
            ])
        print(f'training (epoch: {epoch}, loss: {loss})')

        session.evaluate(model, test_loader, test_fn)

    session.train(model, train_loader, args.epochs, train_fn)

    if args.checkpoint is not None:
        torch.save(model.state_dict(), args.checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, nargs='?', default=11)
    parser.add_argument('--train', type=int, nargs='?', default=13)
    parser.add_argument('--epochs', type=int, nargs='?', default=20)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--checkpoint', type=str, nargs='?', dest='checkpoint')
    parser.add_argument('--plot-loss', dest='plot_loss', action='store_true')
    parser.add_argument('--plot-image', dest='plot_image', action='store_true')
    parser.set_defaults(plot_loss=False, plot_image=False)
    main(parser.parse_args())