DEFAULT_CLASS_WEIGHT_SETTING = True
DEFAULT_BATCH_NORM = False
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PAT = 2
DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_N_SAMPLES = 10  # the number of samples to use for the average reconstruction error
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_RECONSTR_ERROR_WEIGHT = 1e20
DEFAULT_OUTPUT_SIZE = 1
TINY_NUMBER = 1e-3

DEFAULT_ENSEMBLE_SIZE = 5

DEFAULT_LR_PARAMS = dict(C=0.001, class_weight='balanced', solver='lbfgs')

DEFAULT_MODEL_PARAMS = {'hidden_sizes': [100, 100],
                        'dropout_rate': 0.5,
                        'input_size': 714,
                        'batch_norm': False,
                        'class_weight': True}

DEFAULT_TRAINING_PARAMS = {'batch_size': 256,
                           'early_stopping': True,
                           'early_stopping_patience': 2}

DEFAULT_LATENT_DIM = 500
