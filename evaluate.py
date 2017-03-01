import argparse

from mrtous import dataset, transform, network
from mrtous import evaluate, visualize
from torch.utils.data import DataLoader

def main(args):
    model = network.Simple()
    loader = DataLoader(
        dataset.MNIBITE(args.datadir, args.train, transform.RegionCrop()),
        shuffle=True)

    loss_timeline = []
    loss_plot_fn = visualize.loss_plot()
    image_plot_fn = visualize.image_plot(['mr', 'us', 're', 'us-re'])

    def train_fn(inputs, targets, results, epoch, loss):
        loss_timeline.append(loss)
        loss_plot_fn(loss_timeline)
        image_plot_fn([
            inputs.data[0][0].numpy(),
            targets.data[0][0].numpy(),
            results.data[0][0].numpy(),
            targets.data[0][0].numpy()-results.data[0][0].numpy(),
        ])
        print(f'epoch: {epoch}, loss: {loss}')

    evaluate.train(model, loader, args.epochs, train_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, nargs='?')
    parser.add_argument('--epochs', type=int, nargs='?')
    parser.add_argument('--datadir', type=str, nargs='?')
    parser.set_defaults(epochs=30, train=13, datadir='mnibite')
    main(parser.parse_args())