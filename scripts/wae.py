"""
Ex.:
python wae.py --data ~/data/ml-20m/ --logdir ~/experiments/wae/logs/WAE/ \
    --config config/001.gin
"""
import gin
import tensorflow as tf
from argparse import ArgumentParser
from scipy.sparse import load_npz

from data import DataLoader
from util import prune_global

from models.tf import WAE, train
from models.slim import closed_form_slim


@gin.configurable
def build_model(x_train, Model=WAE, n_layers=3, randomize=False, eps=0.01, slim_path=None):
    """Build a wide auto-encoder model with initial weights based on the SLIM
    item-item matrix.
    First weight will be the sparse SLIM weights with some noise,
    the others will be the same weights again, scaled by `eps`.
    """
    if slim_path is not None:
        print('Reading pre-computed SLIM matrix from {}...'.format(slim_path))
        pruned_slim = load_npz(slim_path)
    else:
        print('Computing SLIM matrix')
        slim = closed_form_slim(x_train, l2_reg=500)
        pruned_slim = prune_global(slim)

    tf.reset_default_graph()
    model = Model([pruned_slim] * n_layers)

    return model


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='directory containing pre-processed dataset', type=str)
    parser.add_argument('--cap', help='cap train/val data for debugging', type=int, default=None)
    parser.add_argument('--logdir', help='log directory for tensorboard', type=str)
    parser.add_argument('--config', help='path to gin config file', type=str)
    args = parser.parse_args()

    # override keyword arguments in gin.configurable modules from config file
    gin.parse_config_file(args.config)

    print('Loading data...')
    loader = DataLoader(args.data)
    x_train = loader.load_data('train')
    x_val, y_val = loader.load_data('validation')

    print('Constructing model...')
    model = build_model(x_train)  # don't cap yet, we want realistic sparsities

    print('Training...')
    if args.cap:
        x_train = x_train[:args.cap]
        x_val, y_val = x_val[:args.cap], y_val[:args.cap]
    train(model, x_train, x_val, y_val, log_dir=args.logdir)
