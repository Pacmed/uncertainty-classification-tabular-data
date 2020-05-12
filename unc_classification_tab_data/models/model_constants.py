N_SEEDS = 5
INPUT_DIM = 714
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
DEFAULT_LR_C = 0.001
DEFAULT_LR_PARAMS = dict(C=DEFAULT_LR_C, solver='lbfgs')

DEFAULT_NN_PARAMS = {'hidden_sizes': [100, 100],
                     'dropout_rate': 0.5,
                     'input_size': INPUT_DIM,
                     'batch_norm': False,
                     'class_weight': True}

DEFAULT_NN_PARAMS_NOCLASSWEIGHT = {'hidden_sizes': [100, 100],
                                   'dropout_rate': 0.5,
                                   'input_size': INPUT_DIM,
                                   'batch_norm': False,
                                   'class_weight': False}

DEFAULT_NN_TRAINING_PARAMS = {'batch_size': DEFAULT_BATCH_SIZE,
                              'early_stopping': True,
                              'early_stopping_patience': 2}

DEFAULT_VAE_PARAMS = dict(input_dim=INPUT_DIM,
                          hidden_dims=[],
                          latent_dim=500,
                          learning_rate=DEFAULT_LEARNING_RATE,
                          batch_size=DEFAULT_BATCH_SIZE,
                          reconstr_error_weight=DEFAULT_RECONSTR_ERROR_WEIGHT)
