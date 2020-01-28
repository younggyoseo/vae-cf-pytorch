"""
Ex.:
python wae.py --data ~/data/ml-20m/ --log-dir ~/experiments/wae/logs/WAE/ \
    --config config/001.gin
"""
import gin
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from scipy.sparse import csr_matrix, load_npz

from data import DataLoader
from util import prune_global

from models.tf import WAE, train
from models.slim import closed_form_slim


@gin.configurable
def build_model(x_train, Model=WAE, n_layers=3, eps=0.01, slim_path=None):

    if slim_path is not None:
        print('Reading pre-computed SLIM matrix from {}...'.format(slim_path))
        pruned_slim = load_npz(slim_path)
    else:
        print('Computing SLIM matrix')
        slim = closed_form_slim(x_train, l2_reg=500)
        pruned_slim = prune_global(slim)

    eye = csr_matrix(np.eye(pruned_slim.shape[0]))
    encoder_inits = [eye + eps * pruned_slim for _ in range(n_layers - 1)]
    inits = encoder_inits + [pruned_slim]

    tf.reset_default_graph()
    model = Model(inits)

    return model


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='directory containing pre-processed dataset', type=str)
    parser.add_argument('--log-dir', help='log directory for tensorboard', type=str)
    parser.add_argument('--config', help='path to gin config file', type=str)
    args = parser.parse_args()

    # override keyword arguments in gin.configurable modules from config file
    gin.parse_config_file(args.config)

    print('Loading data...')
    loader = DataLoader(args.data)
    x_train = loader.load_data('train')
    x_val, y_val = loader.load_data('validation')

    print('Constructing model...')
    model = build_model(x_train)

    print('Training...')
    train(model, x_train, x_val, y_val, log_dir=args.log_dir)
