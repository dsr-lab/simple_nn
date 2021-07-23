import torch
import torch.nn.functional as F


from timeit import default_timer as timer

from nn_utils import number_of_correct_samples


class SimpleNN(torch.nn.Module):

    def __init__(self, n_features, hidden_width, n_classes, n_additional_hidden_layers=0):
        super(SimpleNN, self).__init__()
        self.first = torch.nn.Linear(n_features, hidden_width)
        self.activation = torch.relu
        self.last = torch.nn.Linear(hidden_width, n_classes)

        self.additional_hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_width, hidden_width) for i in range(n_additional_hidden_layers)]
        )

    def forward(self, x):
        x = self.first.forward(x)
        x = self.activation(x)

        for layer in self.additional_hidden_layers:
            x = layer.forward(x)
            x = self.activation(x)

        x = self.last.forward(x)

        return x

    def train_loop(self, train_dl, valid_dl, epochs, partial_opt, verbose=False):
        best_valid_accuracy = 0
        best_params = []
        best_epoch = -1

        optimizer = partial_opt(self.parameters())

        for epoch in range(epochs):
            # Train
            train_loss = 0
            n_train_samples = 0
            train_accuracy = 0

            for sample in train_dl:
                scores = self.forward(sample[0])
                loss = F.cross_entropy(scores, sample[1])

                # multiply the loss for the number of images in the current batch
                train_loss += loss.item() * sample[0].shape[0]
                n_train_samples += sample[0].shape[0]

                predictions = torch.argmax(scores, 1)
                train_accuracy += number_of_correct_samples(predictions, sample[1]).item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= n_train_samples
            train_accuracy /= n_train_samples

            with torch.no_grad():
                # Validation
                validation_loss = 0
                n_validation_samples = 0
                validation_accuracy = 0
                if valid_dl is not None:
                    for sample in valid_dl:
                        scores = self.forward(sample[0])
                        loss = F.cross_entropy(scores, sample[1])

                        validation_loss += loss.item() * sample[0].shape[0]
                        n_validation_samples += sample[0].shape[0]

                        predictions = torch.argmax(scores, 1)
                        validation_accuracy += number_of_correct_samples(predictions, sample[1]).item()

                    validation_loss /= n_validation_samples
                    validation_accuracy /= n_validation_samples

                if validation_accuracy > best_valid_accuracy or valid_dl is None:
                    best_valid_accuracy = validation_accuracy if valid_dl is not None else 0
                    best_params = self.state_dict()
                    best_epoch = epoch

            if epoch % 10 == 0 and verbose:
                print(f"Epoch {epoch}: training loss: {train_loss:.3f} - "
                      f"training accuracy: {train_accuracy:.3f} - "
                      f"validation loss: {validation_loss:.3f} -"
                      f"validation accuracy: {validation_accuracy:.3f}")

        if valid_dl is not None and verbose:
            print(f'Best epoch: {best_epoch}, best accuracy: {best_valid_accuracy:.3f}')

        return best_epoch, best_valid_accuracy, best_params

    def test(self, ds):
        n_test_samples = 0
        test_accuracy = 0
        start = timer()
        for sample in ds.test_dl:
            scores = self.forward(sample[0])
            predictions = torch.argmax(scores, 1)
            test_accuracy += number_of_correct_samples(predictions, sample[1]).item()
            n_test_samples += sample[0].shape[0]

        test_accuracy /= n_test_samples
        end = timer()

        print(f'Accuracy on test set: {test_accuracy}')
        print(f'Elapsed time: {end-start:.3f}')
