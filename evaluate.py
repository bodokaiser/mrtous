import argparse
import torch

from mrtous import dataset, transform, network
from mrtous import session, visualize
from torch.utils.data import DataLoader

def main(args):
    model = network.Simple()
    model.load_state_dict(torch.load(args.checkpoint))

    loader = DataLoader(dataset.MNIBITE(args.datadir, args.dataset,
        transform.RegionCrop()))

    losses = []

    if args.plot_loss:
        loss_plot_fn = visualize.loss_plot()
    if args.plot_image:
        image_plot_fn = visualize.image_plot(['mr', 'us', 're', 'us-re'])

    def evaluate_fn(inputs, targets, results, loss):
        losses.append(loss)

        if args.plot_loss:
            loss_plot_fn(losses, [])
        if args.plot_image:
            image_plot_fn([
                inputs.data[0][0].numpy(),
                targets.data[0][0].numpy(),
                results.data[0][0].numpy(),
                targets.data[0][0].numpy()-results.data[0][0].numpy(),
            ])

        print(f'step: {len(losses)}, loss: {loss})')

    session.evaluate(model, loader, evaluate_fn, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, nargs='?', default=11)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--checkpoint', type=str, dest='checkpoint')
    parser.add_argument('--plot-loss', dest='plot_loss', action='store_true')
    parser.add_argument('--plot-image', dest='plot_image', action='store_true')
    parser.set_defaults(plot_loss=False, plot_image=False)
    main(parser.parse_args())