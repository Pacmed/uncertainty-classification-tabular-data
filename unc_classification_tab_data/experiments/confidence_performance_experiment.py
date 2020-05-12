import numpy as np
import os
import matplotlib.pyplot as plt
from unc_classification_tab_data.utils import data_utils, visualizing_utils, metrics
from unc_classification_tab_data.models.bootstrapped_classifier import BootstrappedClassifier
from unc_classification_tab_data.models.nn_ensemble import NNEnsemble
from unc_classification_tab_data.models.vae import VAE
import unc_classification_tab_data.utils.modeling_utils as gen_utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import unc_classification_tab_data.models.model_constants as model_const
from tqdm import tqdm
import argparse
from typing import Tuple


def run_confidence_performance_experiment() -> Tuple[visualizing_utils.ResultContainer,
                                                     visualizing_utils.ResultContainer]:
    """Store and return the uncertainties and predictions for the several methods with and
    without using class weighting.

    Returns
    -------
    type: Tuple[visualizing_utils.ResultContainer, visualizing_utils.ResultContainer]
        The uncertainties and predictions for several methods, over several random seeds.
    """
    rh_no_cw = visualizing_utils.ResultContainer()
    for seed in tqdm(range(model_const.N_SEEDS)):
        np.random.seed(seed)
        # Train Bootstrapped Logistic Regression
        lr_ensemble_model = BootstrappedClassifier(LogisticRegression(
            **model_const.DEFAULT_LR_PARAMS),
            n=model_const.DEFAULT_ENSEMBLE_SIZE,
            bootstrap_size=len(train_data))
        lr_ensemble_model.fit(X_train, y_train)
        y_pred = lr_ensemble_model.predict_proba(X_test)
        rh_no_cw.add_results(y_pred, 'Bootstrapped LR')

        # Train Single Logistic Regression
        lr_single_model = LogisticRegression(**model_const.DEFAULT_LR_PARAMS)
        lr_single_model.fit(X_train, y_train)
        y_pred_single = lr_single_model.predict_proba(X_test)
        rh_no_cw.add_results(y_pred_single, 'Single LR')

        # Train NN Ensemble, without class weight
        nn_model = NNEnsemble(model_const.DEFAULT_ENSEMBLE_SIZE,
                              model_const.DEFAULT_NN_PARAMS_NOCLASSWEIGHT)
        nn_model.train(X_train, y_train,
                       X_val, y_val,
                       training_params=model_const.DEFAULT_NN_TRAINING_PARAMS)

        random_ensemble_member = np.random.randint(0, model_const.DEFAULT_ENSEMBLE_SIZE)
        y_pred_single_nn = nn_model.get_single_predictions(X_test, random_ensemble_member)
        rh_no_cw.add_results(y_pred_single_nn, 'Single NN')

        y_pred_ensemble_nn = nn_model.predict_proba(X_test)
        rh_no_cw.add_results(y_pred_ensemble_nn, 'NN ensemble')

        # VAE
        vae = VAE(train_data=X_train,
                  val_data=X_val,
                  **model_const.DEFAULT_VAE_PARAMS)

        vae.train(n_epochs=model_const.DEFAULT_N_EPOCHS)
        vae_uncertainties = vae.get_reconstr_error(X_test)
        rh_no_cw.uncertainties['VAE'].append(vae_uncertainties)
        # Use predictions of single NN
        rh_no_cw.predictions['VAE'].append(y_pred_single_nn)

    rh_cw = visualizing_utils.ResultContainer()

    # Running the same experiments with using class weighting (and calibration)
    for seed in tqdm(range(model_const.N_SEEDS)):
        np.random.seed(seed)
        # Train Bootstrapped Logistic Regression
        lr_ensemble_model = BootstrappedClassifier(
            LogisticRegression(**model_const.DEFAULT_LR_PARAMS,
                               class_weight='balanced'),
            n=model_const.DEFAULT_ENSEMBLE_SIZE,
            bootstrap_size=len(train_data))
        lr_ensemble_model.fit(X_train, y_train)
        y_pred_lr = lr_ensemble_model.predict_proba(X_test)
        y_pred_val_lr = lr_ensemble_model.predict_proba(X_val)
        rh_cw.add_results(y_pred_lr, 'Bootstrapped LR', y_pred_val_lr, y_val, calibrate=True)

        # Train Single Logistic Regression
        lr_single_model = LogisticRegression(C=model_const.DEFAULT_LR_C, class_weight='balanced')
        lr_single_model.fit(X_train, y_train)
        y_pred_val_single = lr_single_model.predict_proba(X_val)
        y_pred_single = lr_single_model.predict_proba(X_test)
        rh_cw.add_results(y_pred_single, 'Single LR', y_pred_val_single, y_val, calibrate=True)

        # Train NN Ensemble, without class weight
        nn_model = NNEnsemble(model_const.DEFAULT_ENSEMBLE_SIZE, model_const.DEFAULT_NN_PARAMS)
        nn_model.train(X_train, y_train,
                       X_val, y_val,
                       training_params=model_const.DEFAULT_NN_TRAINING_PARAMS)

        random_ensemble_member = np.random.randint(0, model_const.DEFAULT_ENSEMBLE_SIZE)
        y_pred_single_nn = nn_model.get_single_predictions(X_test, random_ensemble_member)
        y_pred_single_nn_val = nn_model.get_single_predictions(X_val, random_ensemble_member)
        rh_cw.add_results(y_pred_single_nn, 'Single NN', y_pred_single_nn_val, y_val,
                          calibrate=True)

        y_pred_ensemble_nn = nn_model.predict_proba(X_test)
        y_pred_ensemble_nn_val = nn_model.predict_proba(X_val)
        rh_cw.add_results(y_pred_ensemble_nn, 'NN ensemble', y_pred_ensemble_nn_val, y_val,
                          calibrate=True)

        # VAE
        vae = VAE(train_data=X_train,
                  val_data=X_val,
                  **model_const.DEFAULT_VAE_PARAMS)

        vae.train(n_epochs=model_const.DEFAULT_N_EPOCHS)
        vae_uncertainties = vae.get_reconstr_error(X_test)
        rh_cw.uncertainties['VAE'].append(vae_uncertainties)
        # Use predictions of single NN
        rh_cw.predictions['VAE'].append(
            gen_utils.platt_scale(y_pred_single_nn[:, 1], y_pred_single_nn_val[:, 1], y_val)
        )
    return rh_cw, rh_no_cw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str, default="/data/processed/benchmark/inhospitalmortality",
                        help="The directory where the csv files are stored.")
    args = parser.parse_args()

    # Load data to use in experiments
    train_data, val_data, test_data = data_utils.load_data(args.data_dir)
    columns_to_use = data_utils.COLUMN_NAMES

    X_train = train_data[columns_to_use].values
    X_val = val_data[columns_to_use].values
    X_test = test_data[columns_to_use].values

    y_train = train_data['y'].values
    y_test = test_data['y'].values
    y_val = val_data['y'].values

    # Run experiments
    class_weight_results, no_class_weight_results = run_confidence_performance_experiment()

    # Which metrics to use, along with a 'prettier' name.
    metrics_to_use = dict([('AUC-ROC', roc_auc_score),
                           ('ECE', metrics.ece),
                           ('Fraction of positives', metrics.average_y)])
    step_size = int(len(X_test) / 10)  # steps of 10% of the data

    # which methods we want to plot
    methods = ['NN ensemble', 'Bootstrapped LR', 'Single NN', 'Single LR']#, 'VAE']

    # Make confidence-performance plots for both experiments and all metrics.
    for cw, rh in [('class_weight_', class_weight_results),
                   ("no_class_weight_", no_class_weight_results)]:

        analyzer = visualizing_utils.UncertaintyAnalyzer([y_test] * model_const.N_SEEDS,
                                                         rh.predictions,
                                                         rh.uncertainties,
                                                         metrics_to_use.values(),
                                                         min_size=step_size * 2 - 1,
                                                         # start at 20%
                                                         step_size=step_size,
                                                         )
        for metric_pretty_name, metric in metrics_to_use.items():
            analyzer.plot_incremental_metric(metric.__name__, methods=methods,
                                             title=metric_pretty_name)
            plt.savefig(os.path.join('plots', cw + metric.__name__ + "AA.png"),
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()
