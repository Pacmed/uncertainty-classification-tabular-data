import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools
import unc_classification_tab_data.utils.general_utils as gen_utils


class ResultContainer:
    """Class that handles storing the uncertainties and predictions given to a test set over
    multiple random seeds.
    """

    def __init__(self):
        """The uncertainties and predictions are stored separately in dictionaries, with as keys
        the method names. """
        self.uncertainties = defaultdict(list)
        self.predictions = defaultdict(list)

    def add_results(self, y_pred, name, y_pred_val=None, calibrate=False):
        """Add results of a single run of a single method.

        Parameters
        ----------
        y_pred: np.ndarray
           The predicted probabilities on the test set.
        name: str
            The method name.
        y_pred_val: np.ndarray, default None
            The predicted probabilities on the validation set, used for calibration
        calibrate: bool, default False
            Whether to calibrate the probabilities (based on the validation data) or not.

       """
        if not calibrate:
            # always use entropy for uncertainty
            self.uncertainties[name].append(gen_utils.entropy(y_pred, axis=1))
            self.predictions[name].append(y_pred[:, 1])
        elif calibrate:
            # platt scale the probabilities
            y_pred_calibrated = gen_utils.platt_scale(y_pred[:, 1], y_pred_val[:, 1],
                                                      y_pred_val)
            self.uncertainties[name].append(gen_utils.entropy(y_pred, axis=1))
            self.predictions[name].append(y_pred_calibrated)


class UncertaintyAnalyzer:
    def __init__(self, y, y_pred_dict, unc_dict, metrics, min_size, step_size):
        """Class that handles analyzing and plotting metrics, when excluding data points based
        on uncertainty. We do this in one class because calculating multiple metrics at once is
        much faster than doing the operations (sorting etc.) one at a time.

        Parameters
        ----------
        y: List[np.ndarray]
            A list of arrays that contain the true labels. It is a list because there might be
            different test sets over different seeds/cv splits.
        y_pred_dict: Dict[List[np.ndarray]]
            A dictionary, with different prediction method as keys (e.g. "BNN", "Logistic
            Regression"). The dictionary maps keys to lists of arrays that contain the predicted
            probabilities corresponding to this method. Again, it is a list because there might
            be variation over seeds/cv splits. It is assumed that the list is in the same order
            as the list in y, so the nth list of predictions in y_pred_dict corresponds to the
            nth list of labels in y.
        unc_dict: Dict[List[np.ndarray]]
            The same as y_pred_dict, except that it contains the predicted uncertainties instead
            of the predicted probabilities.
        metrics: List[function]
            A list of metrics that we want to calculate. All metrics should take labels and
            probabilities in this order: f(y, y_pred).
        min_size: int
            The minimum size of the dataset for which we want to evaluate the metrics.
        step_size: int
            The number of items that we want to add at each increment.
        """
        self.y = y
        self.y_pred = y_pred_dict
        self.uncertainty_dict = unc_dict
        self.loss_dict: defaultdict = defaultdict(list)
        self.loss_std_dict: defaultdict = defaultdict(list)
        self.xs = dict()
        self.metrics = metrics
        self.min_size = min_size
        self.step_size = step_size
        self._calculate_incremental_metrics(metrics)

    def plot_incremental_metric(self, metric, title=None, methods=None, alpha=0.1,
                                key_mapping=None, legend=True):
        """Plot how a metric changes when adding more uncertain points. Do this for multiple
        methods, such as Bayesian NN and KNN.

        Parameters
        ----------
        metric: str
            The name of the metric (metric.__name__).
        title: str (optional)
            The title of the plot.
        methods: List[str] (optional)
            A list with names of methods for which we want to see the metric (e.g. "KNN").
        alpha: float
            The shade of the standard deviation in the plot.
        key_mapping: dict (optional)
            A dictionary that maps method names to "prettier" names for display in the plot.
        """

        sns.set_style('whitegrid')
        if not methods:  # if no methods are specified, plot all.
            methods = self.uncertainty_dict.keys()
        sns.set_palette("Set1", len(methods))
        marker = itertools.cycle(('X', 'o', '^', 'v', 'd', '*', '.'))
        for m in methods:
            x_values = np.array(self.xs[m][metric]) / len(self.y[0])
            met = np.array(self.loss_dict[m][metric])
            std = np.array(self.loss_std_dict[m][metric])
            if key_mapping is not None:
                label = key_mapping[m]
            else:
                label = m
            plt.plot(x_values, met, label=label, marker=next(marker), markersize=8)
            plt.fill_between(x_values,
                             met - std,
                             met + std,
                             alpha=alpha)
        plt.xlabel("Fraction of included data points \n(most certain points are included first)")
        if legend:
            plt.legend()
        if title:
            plt.title(title)
        else:
            plt.title(metric)
        plt.tight_layout()

    def _calculate_incremental_metrics(self, metrics):
        """Calculate the specified metrics for an increasing number of uncertain points. This to
        be used later in plotting etc. It is done for each method and each specified metric.
        Apart from the mean, the standard deviation over seeds/runs is stored.

        Parameters
        ----------
        metrics: List[function]
            A list of metrics that we want to calculate. All metrics should take labels and
            probabilities in this order: f(y, y_pred).
        """
        for key in self.uncertainty_dict.keys():
            xs, loss_dict, score_std = get_incremental_loss(self.y, self.y_pred[key],
                                                            self.uncertainty_dict[key],
                                                            metrics, min_size=self.min_size,
                                                            step_size=self.step_size)
            self.xs[key] = xs
            self.loss_dict[key] = loss_dict
            self.loss_std_dict[key] = score_std


def get_incremental_loss(y_test, y_pred, uncertainty,
                         metrics, min_size, step_size):
    """Parameters
    ----------
    y_test: List[np.ndarray]
        A list of arrays with the true labels.
    y_pred: List[np.ndarray]
        A list of arrays with the predicted probabilities.
    uncertainty: List[np.ndarray]
        A list of arrays with the predicted uncertainties.
    metrics: List[function]
        A list of metrics to calculate. All metrics should take labels and
        probabilities in this order: f(y, y_pred).
    min_size: int
        The minimum size of the dataset for which we want to evaluate the metrics.
    step_size: int
        The number of items that we want to add at each increment.

    Returns
    -------
    xs: DefaultDict[List[int]]
        The positions on the x-axis where the metric is (and could be) calculated.
    score: DefaultDict[List[float]]
        The average of the metric on each subset of the data, for each metric separately stored
        in the dictionary.
    score_std: DefaultDict[List[float]]
        The standard deviation of the metric on each subset of the data, for each metric
        separately stored in the dictionary.
    """
    score, score_std, xs = defaultdict(list), defaultdict(list), defaultdict(list)
    number_of_seeds = len(uncertainty)
    for metric in metrics:
        # loop over each seed, take the average and standard deviation in the end.
        list_of_scores = []
        for n in range(number_of_seeds):
            df = pd.DataFrame({
                'y_pred': y_pred[n],
                'minimum_y_pred': pd.DataFrame([1 - y_pred[n], y_pred[n]]).min(),
                'uncertainty': uncertainty[n],
                'y': y_test[n]
            })

            temp_score, temp_xs = [], []

            # incrementally include more data points, sorted from certain to uncertain.
            sorted_df = df.sort_values(by='uncertainty')
            for i in range(min_size, len(sorted_df), step_size):
                temp = sorted_df[:i]
                temp_score += [metric(temp['y'], temp['y_pred'])]
                temp_xs += [i]
            list_of_scores += [np.array(temp_score)]

        # save mean and standard deviation of the metrics in a dictionary.
        score[metric.__name__] = list(np.array(list_of_scores).mean(axis=0))
        score_std[metric.__name__] = list(np.array(list_of_scores).std(axis=0))
        xs[metric.__name__] = list(range(min_size, len(sorted_df), step_size))
    return xs, score, score_std
