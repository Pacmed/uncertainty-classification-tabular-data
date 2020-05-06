import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from unc_classification_tab_data.utils import data_utils
from unc_classification_tab_data.models.bootstrapped_classifier import BootstrappedClassifier
from unc_classification_tab_data.models.nn_ensemble import NNEnsemble
from unc_classification_tab_data.models.vae import VAE
import unc_classification_tab_data.utils.modeling_utils as gen_utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from unc_classification_tab_data.models.model_constants import *
import unc_classification_tab_data.utils.metrics as metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str, default="/data/processed/benchmark/inhospitalmortality",
                        help="The directory where the csv files are stored.")
    args = parser.parse_args()
    train_data, val_data, test_data = data_utils.load_data(args.data_dir)
    columns_to_use = data_utils.COLUMN_NAMES

    detect_auc, detect_auc_std = defaultdict(dict), defaultdict(dict)
    perf_auc, perf_auc_std = defaultdict(dict), defaultdict(dict)
    for ood_name, (column_name, ood_value) in data_utils.OOD_MAPPINGS.items():
        # Split all data splits into OOD and 'Non-OOD' data.
        train_ood, train_non_ood = data_utils.split_by_ood_name(train_data, column_name, ood_value)
        val_ood, val_non_ood = data_utils.split_by_ood_name(val_data, column_name, ood_value)
        test_ood, test_non_ood = data_utils.split_by_ood_name(test_data, column_name, ood_value)

        # Group all OOD splits together.
        all_ood = pd.concat([train_ood, val_ood, test_ood])

        ood_detect_aucs, aucs_on_subgroup = defaultdict(list), defaultdict(list)

        # Do the same experiments for a number of random seeds
        for random_seed in range(N_SEEDS):
            np.random.seed(random_seed)

            # Train Bootstrapped Logistic Regression on non-OOD data
            lr_model = BootstrappedClassifier(LogisticRegression(**DEFAULT_LR_PARAMS),
                                              n=DEFAULT_ENSEMBLE_SIZE,
                                              bootstrap_size=len(train_data))
            lr_model.fit(train_non_ood[columns_to_use].values, train_non_ood['y'].values)

            # Train NN Ensemble on non-OOD data
            nn_model = NNEnsemble(DEFAULT_ENSEMBLE_SIZE, DEFAULT_NN_PARAMS)
            nn_model.train(train_non_ood[columns_to_use].values, train_non_ood['y'].values,
                           val_non_ood[columns_to_use].values, val_non_ood['y'].values,
                           training_params=DEFAULT_NN_TRAINING_PARAMS)

            # Evaluate NN and LR ensembles
            for model_name, model in [
                ("NN ensemble", nn_model), ("Bootstrapped Logistic Regression",
                                            lr_model)]:
                non_ood_outputs = model.predict_proba(test_non_ood[columns_to_use].values)
                ood_outputs = model.predict_proba(all_ood[columns_to_use].values)

                test_uncertainties = gen_utils.entropy(non_ood_outputs, axis=1)
                ood_uncertainties = gen_utils.entropy(ood_outputs, axis=1)

                detection_auc = metrics.ood_detection_auc(ood_uncertainties, test_uncertainties)
                subgroup_auc = roc_auc_score(all_ood['y'], ood_outputs[:, 1])

                aucs_on_subgroup[model_name] += [subgroup_auc]
                ood_detect_aucs[model_name] += [detection_auc]

            # Train and evaluate VAE
            vae = VAE(train_data=train_non_ood[columns_to_use].values,
                      val_data=val_non_ood[columns_to_use].values,
                      **DEFAULT_VAE_PARAMS)

            vae.train(n_epochs=DEFAULT_N_EPOCHS)
            test_uncertainties = vae.get_reconstr_error(val_non_ood[columns_to_use].values)
            ood_uncertainties = vae.get_reconstr_error(all_ood[columns_to_use].values)
            vae_auc = metrics.ood_detection_auc(ood_uncertainties, test_uncertainties)
            ood_detect_aucs['VAE'] += [vae_auc]

        for model_name in ood_detect_aucs.keys():
            # Calculating the mean OOD detection AUC of the different models over the random seeds
            detect_auc[ood_name][model_name] = np.mean(ood_detect_aucs[model_name])

            # Calculating the mean subgroup AUC of the different models over the random seeds
            perf_auc[ood_name][model_name] = np.mean(aucs_on_subgroup[model_name])

            # Keeping track of standard deviations for the error bars
            perf_auc_std[ood_name][model_name] = np.std(aucs_on_subgroup[model_name])
            detect_auc_std[ood_name][model_name] = np.std(ood_detect_aucs[model_name])

    # transforming the nested dictionaries into dataframes for easy plotting.
    detect_df = pd.DataFrame.from_dict(detect_auc,
                                       orient='index').iloc[::-1]
    detect_std_df = pd.DataFrame.from_dict(detect_auc_std,
                                           orient='index').iloc[::-1]

    sns.set_palette("Set1", 10)
    sns.set_style('whitegrid')

    # Plot OOD detection AUC for all OOD groups
    detect_df.plot(kind='barh', alpha=0.9, xerr=detect_std_df, figsize=(7, 8), fontsize=12,
                   title="OOD "
                         "detection "
                         "AUC-ROC", xlim=(0.0, 0.9))

    plt.savefig(os.path.join('plots', 'OOD_experiment.png'), dpi=300, bbox_inches='tight', pad=0)
    plt.show(block=False)

    # Plot subgroup AUC for all OOD groups
    perf_df = pd.DataFrame.from_dict(perf_auc,
                                     orient='index').iloc[::-1]
    perf_std_df = pd.DataFrame.from_dict(perf_auc_std,
                                         orient='index').iloc[::-1]

    perf_df.plot(kind='barh', alpha=0.9, legend=True, fontsize=12, figsize=(3, 8),
                 xerr=perf_std_df,
                 title='subgroup AUC-ROC', xlim=(0.7, 0.9), yticks=[])
    plt.yticks([], [])
    plt.savefig(os.path.join('plots', 'OOD_experiment_subgroupAUCS.png'), dpi=300,
                bbox_inches='tight', pad=0)
    plt.show()
