import pickle
from typing import Union, Dict
from steingp.svgd import SVGD
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
import pandas as pd
from sklearn.metrics import roc_auc_score


def save_model(object: Union[Dict, SVGD], filename: str):
    if filename.endswith('.pkl'):
        pass
    else:
        filename = "{}.pkl".format(filename)
    with open("models/{}".format(filename), 'wb') as outfile:
        pickle.dump(object, outfile)
    print(f"Model saved as {filename}")


def load_model(filepath: str):
    with open(filepath, 'rb') as openfile:
        model = pickle.load(openfile)
    return model


def write_preds(X: np.ndarray,
                truth: np.ndarray,
                preds: np.ndarray,
                var: np.ndarray,
                writename: str,
                dataset: str,
                iter: int,
                mode: str,
                idx: str = None,
                header: bool = True,
                n_particle: int = None):
    assert np.all(X.ndim == truth.ndim == preds.ndim == var.ndim == 2)
    n_dims = X.shape[1]
    cols = [f'x{i}' for i in range(n_dims)]
    cols.extend(['true_y', 'pred_y', 'var_y'])
    data = np.hstack((X, truth, preds, var))
    data_df = pd.DataFrame(data, columns=cols)
    data_df['dataset'] = dataset
    if idx is not None:
        data_df['idx'] = idx
    if n_particle is not None:
        data_df['n_particles'] = n_particle
    data_df['iteration'] = iter
    data_df.to_csv(writename, index=False, mode=mode, header=header)
    print(f"Predictions successfully written to {writename}")


def rmse(truth: ndarray, predicted: ndarray):
    assert truth.ndim == predicted.ndim, f"Predictions ({predicted.ndim}) and ground truth ({truth.ndim}) dimensions differ."
    return np.mean(np.square(truth - predicted))


def auc(truth: ndarray, proba: ndarray):
    return roc_auc_score(truth, proba)


def accuracy(proba: ndarray, truth: ndarray):
    """
    Compute accuracy for a binary classification task.
    """
    preds = np.where(proba >= 0.5, 1, 0)
    acc = np.mean(preds == truth)
    return acc


def ece(mu: ndarray, truth: ndarray, n_bins: int = 10):
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu.squeeze()

    preds = np.where(mu < 0.5, 0, 1)
    accs = preds == truth.squeeze()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    cal_error = 0
    N = mu.shape[0]
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin_idxs = np.where(np.logical_and(mu >= bl, mu < bu))[0]
        if in_bin_idxs.shape[0] > 0:
            in_bin_accs = accs[in_bin_idxs].mean()
            in_bin_confs = mu[in_bin_idxs].mean()
            cal_error += np.abs(in_bin_accs - in_bin_confs) * (in_bin_idxs.shape[0] / N)
    return cal_error


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    ORANGEBACK = '\033[43m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def box_print(msg, col=bcolors.ENDC):
    """
    Small helper function to print messages to console in a centralised box.
    :param msg: Message to be placed in box
    :type msg: str
    """
    max_len = max(78, len(msg) + 10)
    print('{}'.format('-' * (max_len + 2)))
    print(f'|{col}{msg.center((max_len))}{bcolors.ENDC}|')
    print('{}'.format('-' * (max_len + 2)))


def cprint(msg, col=bcolors.ENDC):
    print(f'{col}{msg}{bcolors.ENDC}')


def hline():
    print('-' * 80)

def train_test_idx(n: int, rng: RandomState, train_prop: float = 0.7):
    train_idx = np.sort(
        rng.choice(np.arange(n),
                   np.floor(n * train_prop).astype(int),
                   replace=False))
    nidx = []
    for i in np.arange(n):
        if not np.isin(i, train_idx):
            nidx.append(i)
    test_idx = np.array(nidx)
    return train_idx, test_idx
