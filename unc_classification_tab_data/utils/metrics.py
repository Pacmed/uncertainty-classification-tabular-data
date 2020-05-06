import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_N_BINS = 10


def ood_detection_auc(ood_uncertainties: np.ndarray, test_uncertainties: np.ndarray) -> float:
    """ Calculate the AUC when using uncertainty to detect OOD.

        Parameters
        ----------
        ood_uncertainties: np.ndarray
            The predicted uncertainties for the OOD samples
        test_uncertainties: int
            The predicted uncertainties for the regular test set.

        Returns
        -------
        type: float
            The AUC-ROC score.
    """
    all_uncertainties = np.concatenate([ood_uncertainties, test_uncertainties])
    labels = np.concatenate([np.ones(len(ood_uncertainties)), np.zeros(len(test_uncertainties))])
    return roc_auc_score(labels, all_uncertainties)


def ece(y, y_pred, n_bins=DEFAULT_N_BINS):
    """Calculate the Expected Calibration Error.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.

    Returns
    -------
    ece: float
        The expected calibration error.

    """
    grouped = _get_binned_df(y, y_pred, n_bins)
    weighed_diff = abs(grouped['y_pred'] - grouped['y']) * grouped['weight']
    ece_score = weighed_diff.sum()
    return ece_score


def resolution(y, y_pred, n_bins=DEFAULT_N_BINS):
    """Calculate the resolution as specified by the brier score decomposition.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.

    Returns
    -------
    res: float
        The resolution.
    """
    base_rate = np.mean(y)
    grouped = _get_binned_df(y, y_pred, n_bins)
    res = (grouped['weight'] * (grouped['y'] - base_rate) ** 2).sum()
    return res


def reliability(y, y_pred, n_bins=DEFAULT_N_BINS):
    """Calculate the reliability as specified by the brier score decomposition. This is the same
    as the ECE, except for the squared term.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.

    Returns
    -------
    rel: float
        The reliability.
    """
    grouped = _get_binned_df(y, y_pred, n_bins)
    rel = (grouped['weight'] * (grouped['y_pred'] - grouped['y']) ** 2).sum()
    return rel


def uncertainty(y, y_pred=None):
    """Calculate the uncertainty as specified by the brier score decomposition. This is
    independent of the predicted probabilities, but the argument is included for coherence with
    other methods.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray (optional, unused)
        The predicted probabilities.
    n_bins: int
        The number of bins to use.

    Returns
    -------
    unc: float
        The uncertainty.
    """
    base_rate = np.mean(y)
    unc = base_rate * (1 - base_rate)
    return unc


def binned_brier_score(y, y_pred, n_bins=DEFAULT_N_BINS):
    """Calculate the 'binned' brier score.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray (optional, unused)
        The predicted probabilities.
    n_bins: int
        The number of bins to use.

    Returns
    -------
    bs: float
        The brier score.
    """
    bs = reliability(y, y_pred, n_bins) - resolution(y, y_pred, n_bins) + uncertainty(y, y_pred)
    return bs


def brier_skill_score(y, y_pred):
    """Calculate the brier skill score. The BSS is perfect when equal to one, a BSS of 0 means
    that there is no improvement to just predicting the average observation rate. If the BSS is
    negative, it is worse than just predicting the average observation rate.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray (optional, unused)
        The predicted probabilities.
    n_bins: int
        The number of bins to use.

    Returns
    -------
    bss: float
        The brier skill score.
    """
    # brier score of our probability predictions
    bs = binned_brier_score(y, y_pred)

    # brier score when always predicting the mean observation rate
    base_y_pred = np.ones(len(y_pred)) * np.mean(y)
    bs_base = binned_brier_score(y, base_y_pred)
    bss = 1 - bs / bs_base
    return bss


def cal(y, y_pred, step_size=25, window_size=100):
    """Calculate CAL/CalBin metric, similar to ECE, but with no fixed windows. Instead,
    a window is shifted to create many overlapping bins.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    step_size: int
        The steps between each sliding window. By default this is 1, so the window slides with one
        step at a time.
    window_size: int
        The size of the window. By default this is 100.

    Returns
    -------
    cal: float
        The CAL/CalBin metric for the given data.
    """
    differences, n_windows = 0, 0
    df = pd.DataFrame({'y': y, 'y_pred': y_pred})
    df = df.sort_values(by='y_pred', ascending=True)

    # slide a window and calculate the absolute calibration error per window position
    for i in range(0, len(y_pred) - window_size, step_size):
        mean_y = np.mean(df['y'][i:i + window_size])
        mean_y_pred = np.mean(df['y_pred'][i:i + window_size])
        differences += abs(mean_y - mean_y_pred)
        n_windows += 1

    # the cal score is the average calibration error of all windows.
    cal_score = differences / n_windows
    return cal_score


def _get_binned_df(y, y_pred, n_bins):
    """Calculate a dataframe with average observations, predictions and the weight
    (bincount/totalcount) per bin. The bins are assumed to be of fixed size.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins
    """
    n = len(y_pred)
    bins = np.arange(0.0, 1.0, 1.0 / n_bins)
    bins_per_prediction = np.digitize(y_pred, bins)

    df = pd.DataFrame({'y_pred': y_pred,
                       'y': y,
                       'pred_bins': bins_per_prediction})

    # calculate the mean y and predicted probabilities per bin
    binned = df.groupby('pred_bins').mean()

    # calculate the number of items per bin
    binned_counts = df.groupby('pred_bins')['y'].count()

    # calculate the proportion of data per bin
    binned['weight'] = binned_counts / n
    return binned


def average_y(y, y_pred):
    return y.mean()
