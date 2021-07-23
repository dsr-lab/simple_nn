import os
from functools import partial

import torch

from config import *
from simple_nn import SimpleNN
from nn_utils import Cifar10, DatasetLoader


def train_model(ds: DatasetLoader, weights_path):
    if MODEL_NEED_TRAINING:
        best_hyper_param = perform_model_tuning(ds)
        perform_model_training(ds, weights_path, best_hyper_param)


def perform_model_tuning(ds: DatasetLoader):
    best_hyper_param = run_model_hyper_parameters_selection(ds)
    return best_hyper_param


def perform_model_training(ds: DatasetLoader, weights_path, best_hyper_params):

    n_features = ds.get_n_features()
    hidden_width = best_hyper_params.get(HP_HIDDEN_WITH)
    hidden_layers = best_hyper_params.get(HP_HIDDEN_LAYERS)
    epochs = best_hyper_params.get(HP_EPOCHS)
    opt = best_hyper_params.get(HP_OPTIMIZER)

    model = SimpleNN(n_features, hidden_width, ds.num_classes, hidden_layers)
    best_epoch, best_valid_accuracy, best_params = \
        model.train_loop(train_dl=ds.get_full_trainin_dl(), valid_dl=None, epochs=epochs, partial_opt=opt, verbose=True)

    torch.save(best_params, os.path.join(weights_path, FILE_PATH_BEST_PARAMS))
    torch.save(best_hyper_params, os.path.join(weights_path, FILE_PATH_BEST_HYPERPARAMS))


def test_model(ds: DatasetLoader, weights_path):
    best_params, best_hyper_params = load_model_params(weights_path)

    n_features = ds.get_n_features()
    hidden_width = best_hyper_params.get(HP_HIDDEN_WITH)
    hidden_layers = best_hyper_params.get(HP_HIDDEN_LAYERS)

    print()

    model = SimpleNN(n_features, hidden_width, ds.num_classes, hidden_layers)
    model.load_state_dict(best_params)
    model.test(ds)


def load_model_params(weights_path):
    best_params = torch.load(os.path.join(weights_path, FILE_PATH_BEST_PARAMS))
    best_hyper_param = torch.load(os.path.join(weights_path, FILE_PATH_BEST_HYPERPARAMS))
    return best_params, best_hyper_param


def run_model_hyper_parameters_selection(ds: DatasetLoader):

    epochs = 30

    lrs = [1e-4, 1e-3]
    # betas = [0.9]
    opts = []
    opts += [partial(torch.optim.Adam, lr=lr) for lr in lrs]

    hidden_widths = [128]
    n_hidden_layers = [0, 1]

    best_valid_accuracy = 0
    best_hyper_params = []

    for opt in opts:
        for hw in hidden_widths:
            for hl in n_hidden_layers:
                model = SimpleNN(ds.get_n_features(), hw, ds.num_classes, hl)

                current_best_epoch, current_best_valid_accuracy, current_best_params = \
                    model.train_loop(
                        train_dl=ds.train_dl, valid_dl=ds.valid_dl, epochs=epochs, partial_opt=opt, verbose=True)

                if current_best_valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = current_best_valid_accuracy
                    best_hyper_params = \
                        {
                            HP_OPTIMIZER: opt,
                            HP_EPOCHS: current_best_epoch,
                            HP_HIDDEN_WITH: hw,
                            HP_HIDDEN_LAYERS: hl
                         }

                    print()
                    print(f'Found better params: {best_hyper_params}')
                    print()

    return best_hyper_params


def init_dataset():
    ds = Cifar10(batch_size=256)

    if MODEL_RUN_ON_DUMMY_DATASET:
        ds.init_dummy_features()
    else:
        ds.init_features()
    return ds


def file_system_setup():
    current_path = os.path.abspath(os.getcwd() + '/model')

    if not os.path.exists(current_path):
        os.makedirs(current_path)

    return current_path


def main():

    ds = init_dataset()

    current_path = file_system_setup()

    train_model(ds, current_path)

    test_model(ds, current_path)


if __name__ == "__main__":
    main()
